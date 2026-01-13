import os
import sys
import re
import glob
import random
import torch
from argparse import ArgumentParser

# 你工程内已有的模块
from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from train import render_set, evaluate, get_logger

# 联合丢包比例（锚点层面：同一 step-lane 的 offsets + scaling + 5xfeat 一起删除）
JOINT_DROP_PCT = 0.20
RAND_SEED = 1337

def group_offsets_bases(bit_dir: str):
    """
    枚举 offsets 基流（base）。返回列表 [(lane:int, step:int, base:str)]
    base 形如 .../offsets_lane{m}_s{s}.b（不含 _*.b chunk 后缀）
    优先通过现有 chunk 文件反推 base，兜底再找纯 base。
    """
    chunks = glob.glob(os.path.join(bit_dir, "offsets_lane*_s*_[0-9]*.b"))
    bases = {}
    for fp in chunks:
        base = re.sub(r'_\d+\.b$', '', fp)
        m = re.search(r'offsets_lane(\d+)_s(\d+)\.b$', base)
        if not m:
            m2 = re.search(r'offsets_lane(\d+)_s(\d+)$', base)
            if not m2:
                continue
            lane = int(m2.group(1)); step = int(m2.group(2))
            base = base + ".b"
        else:
            lane = int(m.group(1)); step = int(m.group(2))
        bases[(lane, step)] = base

    if not bases:
        pure_bases = glob.glob(os.path.join(bit_dir, "offsets_lane*_s*.b"))
        for fp in pure_bases:
            if re.search(r'_\d+\.b$', fp):
                continue
            m = re.search(r'offsets_lane(\d+)_s(\d+)\.b$', fp)
            if not m:
                continue
            lane = int(m.group(1)); step = int(m.group(2))
            bases[(lane, step)] = fp

    items = [(lane, step, base) for (lane, step), base in bases.items()]
    items.sort(key=lambda x: (x[1], x[0]))  # by step, lane
    return items

def remove_base_stream(bit_dir: str, lane: int, step: int):
    """
    删除同一 step-lane 的联合属性流：
      - offsets_lane{lane}_s{step}_*.b
      - scaling_lane{lane}_s{step}_*.b
      - feat_lane{lane}_cc{0..4}_s{step}_*.b
    返回删除的 chunk 文件列表。
    """
    removed = []
    # offsets
    for fp in glob.glob(os.path.join(bit_dir, f"offsets_lane{lane}_s{step}_*.b")):
        try:
            os.remove(fp); removed.append(fp)
        except FileNotFoundError:
            pass
    # scaling
    for fp in glob.glob(os.path.join(bit_dir, f"scaling_lane{lane}_s{step}_*.b")):
        try:
            os.remove(fp); removed.append(fp)
        except FileNotFoundError:
            pass
    # feat 5 段
    for cc in range(5):
        for fp in glob.glob(os.path.join(bit_dir, f"feat_lane{lane}_cc{cc}_s{step}_*.b")):
            try:
                os.remove(fp); removed.append(fp)
            except FileNotFoundError:
                pass
    return removed

def drop_joint_by_offsets(bit_dir: str, joint_pct: float, seed: int = 1337):
    """
    以 offsets 基流为抽样单位，删除 joint_pct 比例的 (lane,step) 对应的 offsets/scaling/5xfeat 联合属性流。
    返回：removed_files(list), stats(dict)
    """
    random.seed(seed)
    offsets_bases = group_offsets_bases(bit_dir)  # [(lane, step, base)]
    total_offsets_bases = len(offsets_bases)

    stats = {
        "total_offsets_bases": total_offsets_bases,
        "dropped_offsets_bases": 0,
        "dropped_scaling_bases_est": 0,  # = offsets 选中数
        "dropped_feat_bases_est": 0,     # = offsets 选中数 * 5
        "removed_chunk_files": 0,
    }
    removed = []

    if total_offsets_bases == 0:
        return removed, stats

    k = int(round(total_offsets_bases * joint_pct))
    k = max(1, min(k, total_offsets_bases))
    chosen = set(random.sample(range(total_offsets_bases), k))

    for idx, (lane, step, base) in enumerate(offsets_bases):
        if idx not in chosen:
            continue
        removed += remove_base_stream(bit_dir, lane, step)

    stats["dropped_offsets_bases"] = k
    stats["dropped_scaling_bases_est"] = k
    stats["dropped_feat_bases_est"] = k * 5
    stats["removed_chunk_files"] = len(removed)
    return removed, stats

def main():
    parser = ArgumentParser(description="EC test: joint drop (offsets+scaling+5xfeat) per step-lane")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    # 与训练脚本一致的附加参数
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    args = parser.parse_args(sys.argv[1:])

    # 日志
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)

    bit_dir = os.path.join(model_path, "bitstreams")
    if not os.path.isdir(bit_dir):
        logger.info(f"bitstreams dir not found: {bit_dir}")
        sys.exit(1)

    # 1) 联合丢包
    logger.info(f"Start joint dropping in: {bit_dir}")
    removed, stats = drop_joint_by_offsets(bit_dir, JOINT_DROP_PCT, RAND_SEED)
    logger.info(f"JOINT_DROP_PCT={JOINT_DROP_PCT:.3f}")
    logger.info(f"  offsets bases: total={stats['total_offsets_bases']}, dropped={stats['dropped_offsets_bases']} "
                f"({stats['dropped_offsets_bases']/max(1,stats['total_offsets_bases']):.3f})")
    logger.info(f"  scaling bases dropped (est.): {stats['dropped_scaling_bases_est']}")
    logger.info(f"  feat bases dropped (est.):    {stats['dropped_feat_bases_est']}")
    logger.info(f"  removed chunk files:          {stats['removed_chunk_files']}")
    if len(removed) > 0:
        logger.info("Examples:")
        for fp in removed[:10]:
            logger.info(f"  - {os.path.basename(fp)}")

    # 2) 实例化 GaussianModel 与 Scene
    dataset = lp.extract(args)
    pipeline = pp.extract(args)

    is_synthetic_nerf = os.path.exists(os.path.join(args.source_path, "transforms_train.json"))
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
        is_synthetic_nerf=is_synthetic_nerf,
    )
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    # 3) 解码（丢文件后，解码器会触发 EC）
    logger.info("Decoding (with EC if needed)...")
    log_info = gaussians.conduct_decoding(pre_path_name=bit_dir)
    logger.info(log_info)

    # 4) 渲染 test 集并评估
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    logger.info("Rendering test set...")
    t_list, visible_count = render_set(dataset.model_path, "test", -1, scene.getTestCameras(), gaussians, pipeline, background)
    logger.info("Rendering complete.")

    logger.info("Evaluating...")
    evaluate(dataset.model_path, visible_count=visible_count, wandb=None, logger=logger)
    logger.info("Done.")

if __name__ == "__main__":
    main()
