import os
import sys
import json
import time
import random
import shutil
import tempfile
import numpy as np
import torch
import glob
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torchvision
import torchvision.transforms.functional as tf
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel  # 调整为你的文件名
from gaussian_renderer import prefilter_voxel, render  # 添加导入
from train import get_logger  # 只需 get_logger，原 render_set 和 evaluate 可内联重写
from utils.loss_utils import ssim  # 若 ssim_fn 不需要，可用 utils 中的 ssim

lpips_fn = lpips.LPIPS(net='vgg').cuda()
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simulate_pak_drop(bit_dir: str, p_drop: float, seed: int, temp_dir: str):
    """
    复制 bitstreams 到 temp_dir，并随机删除 p_drop 比例的 .pak 文件（模拟组级丢失）。
    返回 temp_dir（已删除部分 .pak）。
    """
    shutil.copytree(bit_dir, temp_dir, dirs_exist_ok=True)

    pak_files = sorted(glob.glob(os.path.join(temp_dir, "*.pak")))
    if not pak_files:
        raise ValueError(f"No .pak files found in {bit_dir}")

    num_pak = len(pak_files)
    num_drop = max(1, int(round(num_pak * p_drop)))

    set_all_seeds(seed)
    drop_idx = random.sample(range(num_pak), num_drop)
    for i in drop_idx:
        os.remove(pak_files[i])

    return temp_dir


def eval_one(gaussians, scene, pipeline, bit_dir: str, model_path: str, iteration: int, p_drop: float, use_tr: bool,
             seed: int, logger, dataset):
    """
    执行一次评估：conduct_decoding（可能模拟 drop），render_set，计算指标。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if p_drop > 0:
            bit_dir = simulate_pak_drop(bit_dir, p_drop, seed, temp_dir)
            logger.info(f"Simulated drop {p_drop * 100:.1f}% .pak in temp {bit_dir}")

        # 设置 Transformer 开关
        os.environ['HAC_USE_EC_TRANSFORMER'] = '1' if use_tr else '0'
        os.environ['HAC_PAK_ENABLE'] = '1'  # 启用 .pak 解码

        # 解码
        log_info = gaussians.conduct_decoding(pre_path_name=bit_dir)
        logger.info(log_info)

        # 渲染测试集
        name = "test"
        views = scene.getTestCameras()
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(model_path, name, f"renders_drop{p_drop:.2f}_tr{int(use_tr)}_{iteration}")
        gt_path = os.path.join(model_path, name, f"gt_{iteration}")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        psnrs = []
        ssims = []
        lpipss = []
        t_list = []

        for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} (drop={p_drop}, tr={use_tr})")):
            torch.cuda.synchronize()
            t_start = time.time()

            visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=visible_mask)

            torch.cuda.synchronize()
            t_end = time.time()
            t_list.append(t_end - t_start)

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image.cuda(), 0.0, 1.0)

            # 保存图像（可选，调试用）
            torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
            torchvision.utils.save_image(gt, os.path.join(gt_path, f'{idx:05d}.png'))

            # 指标
            psnrs.append(psnr_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())
            ssims.append(ssim_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())
            lpipss.append(lpips_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())

        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        avg_lpips = np.mean(lpipss)
        avg_fps = 1.0 / np.mean(t_list[5:])  # 跳过前5个热身

        logger.info(
            f"[{name} drop={p_drop:.2f} tr={int(use_tr)}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FPS: {avg_fps:.4f}")

        return avg_psnr, avg_ssim, avg_lpips, avg_fps


def main():
    parser = ArgumentParser(
        description="评估 ECTransformer 在模拟丢失下的效果（比较 baseline vs. drop_no_tr vs. drop_tr）")
    lp = ModelParams(parser)  # 这会添加 --model_path 等标准参数
    pp = PipelineParams(parser)
    # 移除重复的 --model_path 添加
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--p_drop", type=float, default=0.2, help="模拟 .pak 丢失比例")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    args = parser.parse_args(sys.argv[1:])

    model_path = args.model_path  # 从 lp 提取
    bit_dir = os.path.join(model_path, "bitstreams")
    if not os.path.isdir(bit_dir):
        raise ValueError(f"bitstreams not found: {bit_dir}")

    logger = get_logger(model_path)
    set_all_seeds(args.seed)

    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    is_synth = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))

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
    ).cuda()

    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    # 加载 finetune 后的权重（包含 ec_transformer）
    ckpt = os.path.join(model_path, "mlp_ckpt_ec_finetuned.pth")
    if os.path.isfile(ckpt):
        gaussians.load_mlp_checkpoints(ckpt)
        logger.info(f"Loaded finetuned ckpt: {ckpt}")
    else:
        logger.warning(f"No finetuned ckpt found: {ckpt}, using random init for Transformer (may perform poorly)")

    # 评估三种模式
    t0 = time.time()

    # 1. Normal (no drop)
    logger.info("Eval Normal (no drop)")
    psnr_n, ssim_n, lpips_n, fps_n = eval_one(gaussians, scene, pipeline, bit_dir, model_path, args.iteration,
                                              p_drop=0.0, use_tr=False, seed=args.seed, logger=logger, dataset=dataset)

    # 2. Drop without Transformer
    logger.info("Eval Drop without Transformer")
    psnr_d_no, ssim_d_no, lpips_d_no, fps_d_no = eval_one(gaussians, scene, pipeline, bit_dir, model_path,
                                                          args.iteration, p_drop=args.p_drop, use_tr=False,
                                                          seed=args.seed, logger=logger, dataset=dataset)

    # 3. Drop with Transformer
    logger.info("Eval Drop with Transformer")
    psnr_d_tr, ssim_d_tr, lpips_d_tr, fps_d_tr = eval_one(gaussians, scene, pipeline, bit_dir, model_path,
                                                          args.iteration, p_drop=args.p_drop, use_tr=True,
                                                          seed=args.seed, logger=logger, dataset=dataset)

    # 总结
    logger.info("\nSummary:")
    logger.info(f"Normal: PSNR {psnr_n:.4f} | SSIM {ssim_n:.4f} | LPIPS {lpips_n:.4f} | FPS {fps_n:.4f}")
    logger.info(
        f"Drop {args.p_drop:.2f} no TR: PSNR {psnr_d_no:.4f} ({psnr_d_no - psnr_n:+.4f}) | SSIM {ssim_d_no:.4f} ({ssim_d_no - ssim_n:+.4f}) | LPIPS {lpips_d_no:.4f} ({lpips_d_no - lpips_n:+.4f}) | FPS {fps_d_no:.4f}")
    logger.info(
        f"Drop {args.p_drop:.2f} w/ TR: PSNR {psnr_d_tr:.4f} ({psnr_d_tr - psnr_n:+.4f}) | SSIM {ssim_d_tr:.4f} ({ssim_d_tr - ssim_n:+.4f}) | LPIPS {lpips_d_tr:.4f} ({lpips_d_tr - lpips_n:+.4f}) | FPS {fps_d_tr:.4f}")
    logger.info(
        f"TR Improvement over no TR: PSNR {psnr_d_tr - psnr_d_no:+.4f} | SSIM {ssim_d_tr - ssim_d_no:+.4f} | LPIPS {lpips_d_tr - lpips_d_no:+.4f}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
