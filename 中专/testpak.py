import os
import sys
import glob
import random
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from train import render_set, evaluate, get_logger

def list_paks(bit_dir: str):
    return sorted(glob.glob(os.path.join(bit_dir, "pak_lane*_s*.pak")))

def drop_random_paks(paks, drop_pct: float, seed: int = 1337):
    random.seed(seed)
    total = len(paks)
    k = int(round(total * drop_pct))
    k = max(0, min(k, total))
    sel = set(random.sample(range(total), k)) if k > 0 else set()
    removed = []
    for i, fp in enumerate(paks):
        if i not in sel:
            continue
        try:
            os.remove(fp)
            removed.append(fp)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[drop_random_paks] remove failed: {fp} -> {e}")
    return removed, total, k

def main():
    parser = ArgumentParser(description="Drop a percentage of .pak groups and evaluate")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    # 与训练保持一致的附加参数
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    # 本脚本参数
    parser.add_argument("--drop_pct", type=float, default=0.20, help="fraction of .pak files to remove (default 0.2)")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--use_ec_head", type=int, default=0, help="1 to enable learning-based EC, 0 to disable (default 0)")

    args = parser.parse_args(sys.argv[1:])

    # 环境开关：启用 .pak 解包；是否使用学习型 EC
    os.environ.setdefault("HAC_PAK_ENABLE", "1")
    os.environ["HAC_USE_EC_HEAD"] = "1" if args.use_ec_head > 0 else "0"

    model_path = args.model_path
    bit_dir = os.path.join(model_path, "bitstreams")
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)

    if not os.path.isdir(bit_dir):
        logger.info(f"bitstreams dir not found: {bit_dir}")
        sys.exit(1)

    # 1) 列出并随机删除 drop_pct 的 .pak
    paks = list_paks(bit_dir)
    if len(paks) == 0:
        logger.info(f"no .pak found in: {bit_dir}")
        sys.exit(1)

    logger.info(f"Found .pak groups: {len(paks)}")
    removed, total, dropped = drop_random_paks(paks, args.drop_pct, args.seed)
    logger.info(f"DROP_PCT = {args.drop_pct:.3f} -> dropped {dropped}/{total} ({dropped/max(1,total):.3f})")
    logger.info(f"Removed files: {len(removed)}")
    if removed:
        logger.info("Examples:")
        for fp in removed[:10]:
            logger.info(f"  - {os.path.basename(fp)}")

    # 2) 构建模型与场景
    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    is_synth = os.path.exists(os.path.join(args.source_path, "transforms_train.json"))

    gaussians = GaussianModel(
        feat_dim=dataset.feat_dim,
        n_offsets=dataset.n_offsets,
        voxel_size=dataset.voxel_size,
        update_depth=dataset.update_depth,
        update_init_factor=dataset.update_init_factor,
        update_hierachy_factor=dataset.update_hierachy_factor,
        use_feat_bank=dataset.use_feat_bank,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        is_synthetic_nerf=is_synth,
    )
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    # 3) 解码
    logger.info("Decoding (group .pak enabled; EC head: %s)..." % ("ON" if args.use_ec_head>0 else "OFF"))
    info = gaussians.conduct_decoding(pre_path_name=bit_dir)
    logger.info(info)

    # 4) 渲染与评估
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    logger.info("Rendering test set...")
    _, visible_count = render_set(dataset.model_path, "test", -1, scene.getTestCameras(), gaussians, pipeline, background)
    logger.info("Rendering complete.")

    logger.info("Evaluating...")
    evaluate(dataset.model_path, visible_count=visible_count, wandb=None, logger=logger)
    logger.info("Done.")

if __name__ == "__main__":
    main()
