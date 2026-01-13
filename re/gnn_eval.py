#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from scene.gaussian_model import GaussianModel
from gaussian_renderer import prefilter_voxel, render
from train import get_logger


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
        try:
            os.remove(pak_files[i])
        except FileNotFoundError:
            pass
    return temp_dir


@torch.no_grad()
def eval_one(gaussians, scene, pipeline, bit_dir: str, model_path: str, iteration: int,
             p_drop: float, use_gnn: bool, gnn_weights: str, gnn_only: bool, ec_iters: int,
             gnn_k: int, gnn_hidden: int, gnn_max_known: int, gnn_bq: int,
             seed: int, logger, dataset, save_images: bool = True, tag: str = ""):
    """
    执行一次评估：可选模拟 drop -> conduct_decoding（内部按环境变量启用 GNN/EC）-> 渲染 -> 指标。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if p_drop > 0:
            bit_dir = simulate_pak_drop(bit_dir, p_drop, seed, temp_dir)
            logger.info(f"Simulated drop {p_drop * 100:.1f}% .pak in temp {bit_dir}")

        # 按环境变量控制解码侧 GNN/EC
        os.environ['HAC_PAK_ENABLE'] = '1'
        if use_gnn:
            os.environ['HAC_GNN_INPAINT'] = '1'
            os.environ['HAC_GNN_ONLY'] = '1' if gnn_only else '0'
            os.environ['HAC_EC_ITERS'] = str(int(ec_iters))
            os.environ['HAC_GNN_FIT_STEPS'] = '0'  # 离线权重，解码时不再训练
            os.environ['HAC_GNN_K'] = str(int(gnn_k))
            os.environ['HAC_GNN_H'] = str(int(gnn_hidden))
            os.environ['HAC_GNN_MAX_KNOWN'] = str(int(gnn_max_known))
            os.environ['HAC_GNN_BQ'] = str(int(gnn_bq))
            if gnn_weights:
                os.environ['HAC_GNN_WEIGHTS'] = os.path.abspath(gnn_weights)
        else:
            os.environ['HAC_GNN_INPAINT'] = '0'
            os.environ['HAC_EC_ITERS'] = str(int(ec_iters))  # 允许只跑 EC

        # 解码（内部会加载 bitstreams/mlp_ckpt.pth）
        logger.info(f"Decoding with GNN={'ON' if use_gnn else 'OFF'}, EC iters={ec_iters} ...")
        log_info = gaussians.conduct_decoding(pre_path_name=bit_dir)
        logger.info(log_info)

        # 渲染测试集
        split_name = "test"
        views = scene.getTestCameras()
        if not views or len(views) == 0:
            raise RuntimeError("No test cameras found; please check dataset/source_path.")
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 输出目录
        render_dir = os.path.join(model_path, split_name, f"renders_drop{p_drop:.2f}_gnn{int(use_gnn)}_{iteration}{tag}")
        gt_dir = os.path.join(model_path, split_name, f"gt_{iteration}{tag}")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        psnrs, ssims, lpipss, t_list = [], [], [], []
        for idx, view in enumerate(tqdm(views, desc=f"Rendering {split_name} (drop={p_drop:.2f}, gnn={use_gnn})")):
            torch.cuda.synchronize()
            t_start = time.time()

            visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=visible_mask)

            torch.cuda.synchronize()
            t_end = time.time()
            t_list.append(t_end - t_start)

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image.cuda(), 0.0, 1.0)

            if save_images:
                torchvision.utils.save_image(rendering, os.path.join(render_dir, f'{idx:05d}.png'))
                torchvision.utils.save_image(gt, os.path.join(gt_dir, f'{idx:05d}.png'))

            # 指标
            psnrs.append(psnr_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())
            ssims.append(ssim_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())
            lpipss.append(lpips_fn(rendering.unsqueeze(0), gt.unsqueeze(0)).item())

        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))
        avg_lpips = float(np.mean(lpipss))
        # 跳过前5个热身
        if len(t_list) > 5:
            avg_fps = 1.0 / float(np.mean(t_list[5:]))
        else:
            avg_fps = 1.0 / float(np.mean(t_list))

        logger.info(f"[{split_name} drop={p_drop:.2f} gnn={int(use_gnn)}] "
                    f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FPS: {avg_fps:.4f}")

        return avg_psnr, avg_ssim, avg_lpips, avg_fps


def main():
    parser = ArgumentParser(description="评估 GNN inpainting 在模拟丢包下的效果（Normal / Drop-NoGNN / Drop-GNN）")
    # 与训练/渲染保持一致的标准参数
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--p_drop", type=float, default=0.2, help="模拟 .pak 丢失比例")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    # GNN 相关
    parser.add_argument("--use_gnn", type=int, default=1, help="1 启用 GNN 补洞（默认为 1）")
    parser.add_argument("--gnn_weights", type=str, default="", help="场景离线训练的 GNN 权重路径（建议提供）")
    parser.add_argument("--gnn_only", type=int, default=1, help="1 解码时只用 GNN，不跑 EC（默认 1）")
    parser.add_argument("--ec_iters", type=int, default=0, help="GNN 之后的 EC 精修迭代次数（默认 0）")
    parser.add_argument("--gnn_k", type=int, default=16)
    parser.add_argument("--gnn_hidden", type=int, default=256)
    parser.add_argument("--gnn_max_known", type=int, default=20000)
    parser.add_argument("--gnn_bq", type=int, default=1024)

    # 是否保存渲染图
    parser.add_argument("--save_images", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    model_path = args.model_path
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

    t0 = time.time()

    # 1) Normal（不丢包，不用 GNN）
    logger.info("Eval Normal (no drop, no GNN)")
    psnr_n, ssim_n, lpips_n, fps_n = eval_one(
        gaussians, scene, pipeline, bit_dir, model_path, args.iteration,
        p_drop=0.0, use_gnn=False, gnn_weights="", gnn_only=True, ec_iters=0,
        gnn_k=args.gnn_k, gnn_hidden=args.gnn_hidden, gnn_max_known=args.gnn_max_known, gnn_bq=args.gnn_bq,
        seed=args.seed, logger=logger, dataset=dataset,
        save_images=bool(args.save_images), tag="_normal"
    )

    # 2) Drop without GNN（丢包不用 GNN/EC）
    logger.info("Eval Drop without GNN")
    psnr_d_no, ssim_d_no, lpips_d_no, fps_d_no = eval_one(
        gaussians, scene, pipeline, bit_dir, model_path, args.iteration,
        p_drop=args.p_drop, use_gnn=False, gnn_weights="", gnn_only=True, ec_iters=0,
        gnn_k=args.gnn_k, gnn_hidden=args.gnn_hidden, gnn_max_known=args.gnn_max_known, gnn_bq=args.gnn_bq,
        seed=args.seed, logger=logger, dataset=dataset,
        save_images=bool(args.save_images), tag="_drop_nognn"
    )

    # 3) Drop with GNN（丢包用 GNN，默认不跑 EC；可通过 ec_iters>0 开启 GNN->EC 精修）
    logger.info("Eval Drop with GNN")
    psnr_d_gn, ssim_d_gn, lpips_d_gn, fps_d_gn = eval_one(
        gaussians, scene, pipeline, bit_dir, model_path, args.iteration,
        p_drop=args.p_drop, use_gnn=bool(args.use_gnn), gnn_weights=args.gnn_weights,
        gnn_only=bool(args.gnn_only), ec_iters=args.ec_iters,
        gnn_k=args.gnn_k, gnn_hidden=args.gnn_hidden, gnn_max_known=args.gnn_max_known, gnn_bq=args.gnn_bq,
        seed=args.seed, logger=logger, dataset=dataset,
        save_images=bool(args.save_images), tag="_drop_gnn"
    )

    # 总结
    logger.info("\nSummary:")
    logger.info(f"Normal:               PSNR {psnr_n:.4f} | SSIM {ssim_n:.4f} | LPIPS {lpips_n:.4f} | FPS {fps_n:.4f}")
    logger.info(f"Drop {args.p_drop:.2f} no GNN: PSNR {psnr_d_no:.4f} ({psnr_d_no - psnr_n:+.4f}) | "
                f"SSIM {ssim_d_no:.4f} ({ssim_d_no - ssim_n:+.4f}) | "
                f"LPIPS {lpips_d_no:.4f} ({lpips_d_no - lpips_n:+.4f}) | FPS {fps_d_no:.4f}")
    logger.info(f"Drop {args.p_drop:.2f} w/ GNN: PSNR {psnr_d_gn:.4f} ({psnr_d_gn - psnr_n:+.4f}) | "
                f"SSIM {ssim_d_gn:.4f} ({ssim_d_gn - ssim_n:+.4f}) | "
                f"LPIPS {lpips_d_gn:.4f} ({lpips_d_gn - lpips_n:+.4f}) | FPS {fps_d_gn:.4f}")
    logger.info(f"GNN improvement over no GNN: "
                f"PSNR {psnr_d_gn - psnr_d_no:+.4f} | SSIM {ssim_d_gn - ssim_d_no:+.4f} | LPIPS {lpips_d_gn - lpips_d_no:+.4f}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
