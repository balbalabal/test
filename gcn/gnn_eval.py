#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import shutil
import tempfile
import glob
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from tqdm import tqdm

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
    """复制 bitstreams 到 temp_dir，并随机删除 p_drop 比例的 .pak 文件"""
    shutil.copytree(bit_dir, temp_dir, dirs_exist_ok=True)
    pak_files = sorted(glob.glob(os.path.join(temp_dir, "*.pak")))
    if not pak_files:
        raise ValueError(f"No .pak files found in {bit_dir}")
    num_pak = len(pak_files)
    num_drop = max(1, int(round(num_pak * p_drop)))
    random.seed(seed)
    drop_idx = random.sample(range(num_pak), num_drop)
    for i in drop_idx:
        try:
            os.remove(pak_files[i])
        except FileNotFoundError:
            pass
    return temp_dir


@torch.no_grad()
def eval_drop_with_gnn(gaussians, scene, pipeline, bit_dir: str, model_path: str, iteration: int,
                       p_drop: float, gnn_weights: str, ec_iters: int,
                       gnn_backend: str, gnn_conv: str, gnn_mp: int, gnn_k: int, gnn_k2: int,
                       gnn_heads: int, gnn_hidden: int,
                       pred_max_known: int, pred_bq: int,
                       tau: float, alpha: float, conf_off: float, conf_fea: float, conf_sca: float,
                       force_safe: int, seed: int, logger, dataset, save_images: bool = True, tag: str = ""):

    with tempfile.TemporaryDirectory() as temp_dir:
        if p_drop > 0:
            bit_dir = simulate_pak_drop(bit_dir, p_drop, seed, temp_dir)
            logger.info(f"Simulated drop {p_drop * 100:.1f}% .pak in temp {bit_dir}")

        # 只评估 GNN
        os.environ['HAC_PAK_ENABLE'] = '1'
        os.environ['HAC_GNN_INPAINT'] = '1'
        os.environ['HAC_GNN_ONLY'] = '1'
        os.environ['HAC_EC_ITERS'] = str(int(ec_iters))
        os.environ['HAC_GNN_FIT_STEPS'] = '0'
        if force_safe is not None:
            os.environ['HAC_GNN_SAFE'] = str(int(force_safe))

        # 结构（CLI 优先；否则由权重 meta 在解码端自动对齐）
        if gnn_backend: os.environ['HAC_GNN_BACKEND'] = gnn_backend
        if gnn_conv:    os.environ['HAC_GNN_CONV'] = gnn_conv
        if gnn_mp is not None: os.environ['HAC_GNN_MP'] = str(int(gnn_mp))
        if gnn_k is not None:  os.environ['HAC_GNN_K']  = str(int(gnn_k))
        if gnn_k2 is not None: os.environ['HAC_GNN_K2'] = str(int(gnn_k2))
        if gnn_heads is not None: os.environ['HAC_GNN_HEADS'] = str(int(gnn_heads))
        if gnn_hidden is not None: os.environ['HAC_GNN_H'] = str(int(gnn_hidden))
        if pred_max_known is not None:
            os.environ['HAC_GNN_PRED_MAX_KNOWN'] = str(int(pred_max_known))
            os.environ['HAC_GNN_MAX_KNOWN'] = str(int(pred_max_known))
        if pred_bq is not None:
            os.environ['HAC_GNN_PRED_BQ'] = str(int(pred_bq))
            os.environ['HAC_GNN_BQ'] = str(int(pred_bq))

        # 置信/阈值
        if tau is not None:       os.environ['HAC_GNN_TAU'] = str(float(tau))
        if alpha is not None:     os.environ['HAC_GNN_ALPHA'] = str(float(alpha))
        if conf_off is not None:  os.environ['HAC_GNN_CONF_TH_OFF'] = str(float(conf_off))
        if conf_fea is not None:  os.environ['HAC_GNN_CONF_TH_FEA'] = str(float(conf_fea))
        if conf_sca is not None:  os.environ['HAC_GNN_CONF_TH_SCA'] = str(float(conf_sca))

        if gnn_weights:
            os.environ['HAC_GNN_WEIGHTS'] = os.path.abspath(gnn_weights)

        # 解码
        logger.info(f"Decoding with GNN=ON, EC iters={ec_iters} ...")
        _ = gaussians.conduct_decoding(pre_path_name=bit_dir)

        # 渲染
        views = scene.getTestCameras()
        if not views or len(views) == 0:
            raise RuntimeError("No test cameras found; please check dataset/source_path.")
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_dir = os.path.join(model_path, "test", f"renders_drop{p_drop:.2f}_gnn1_{iteration}{tag}")
        gt_dir = os.path.join(model_path, "test", f"gt_{iteration}{tag}")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        psnrs, ssims, lpipss, t_list = [], [], [], []
        for idx, view in enumerate(tqdm(views, desc=f"Rendering test (drop={p_drop:.2f}, gnn=True)")):
            torch.cuda.synchronize(); t0 = time.time()
            vm = prefilter_voxel(view, gaussians, pipeline, background)
            out = render(view, gaussians, pipeline, background, visible_mask=vm)
            torch.cuda.synchronize(); t1 = time.time()
            t_list.append(t1 - t0)

            img = torch.clamp(out["render"], 0.0, 1.0)
            gt  = torch.clamp(view.original_image.cuda(), 0.0, 1.0)
            if save_images:
                torchvision.utils.save_image(img, os.path.join(render_dir, f'{idx:05d}.png'))
                torchvision.utils.save_image(gt,  os.path.join(gt_dir,    f'{idx:05d}.png'))

            psnrs.append(psnr_fn(img.unsqueeze(0), gt.unsqueeze(0)).item())
            ssims.append(ssim_fn(img.unsqueeze(0), gt.unsqueeze(0)).item())
            lpipss.append(lpips_fn(img.unsqueeze(0), gt.unsqueeze(0)).item())

        psnr = float(np.mean(psnrs))
        ssim = float(np.mean(ssims))
        lp   = float(np.mean(lpipss))
        if len(t_list) > 5:
            fps = 1.0 / float(np.mean(t_list[5:]))
        else:
            fps = 1.0 / float(np.mean(t_list))

        logger.info(f"[test drop={p_drop:.2f} gnn=1] PSNR: {psnr:.4f} | SSIM: {ssim:.4f} | LPIPS: {lp:.4f} | FPS: {fps:.4f}")
        return psnr, ssim, lp, fps


def main():
    parser = ArgumentParser(description="只评估 Drop + GNN")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--p_drop", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    parser.add_argument("--gnn_weights", type=str, required=True)
    parser.add_argument("--ec_iters", type=int, default=0)
    parser.add_argument("--force_safe", type=int, default=None, help="1 强制 SAFE（诊断兜底）")

    # 结构/资源
    parser.add_argument("--gnn_backend", type=str, default=None, choices=["vanilla", "pyg"])
    parser.add_argument("--gnn_conv", type=str, default=None, choices=["edge", "gat"])
    parser.add_argument("--gnn_mp", type=int, default=None)
    parser.add_argument("--gnn_k", type=int, default=None)
    parser.add_argument("--gnn_k2", type=int, default=None)
    parser.add_argument("--gnn_heads", type=int, default=None)
    parser.add_argument("--gnn_hidden", type=int, default=None)
    parser.add_argument("--pred_max_known", type=int, default=80000)
    parser.add_argument("--pred_bq", type=int, default=2048)

    # 置信/阈值
    parser.add_argument("--tau", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--conf_off", type=float, default=0.55)
    parser.add_argument("--conf_fea", type=float, default=0.40)
    parser.add_argument("--conf_sca", type=float, default=0.40)

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

    psnr, ssim, lp, fps = eval_drop_with_gnn(
        gaussians, scene, pipeline, bit_dir, model_path, args.iteration,
        p_drop=args.p_drop, gnn_weights=args.gnn_weights, ec_iters=args.ec_iters,
        gnn_backend=args.gnn_backend, gnn_conv=args.gnn_conv, gnn_mp=args.gnn_mp,
        gnn_k=args.gnn_k, gnn_k2=args.gnn_k2, gnn_heads=args.gnn_heads, gnn_hidden=args.gnn_hidden,
        pred_max_known=args.pred_max_known, pred_bq=args.pred_bq,
        tau=args.tau, alpha=args.alpha, conf_off=args.conf_off, conf_fea=args.conf_fea, conf_sca=args.conf_sca,
        force_safe=args.force_safe, seed=args.seed, logger=logger, dataset=dataset,
        save_images=bool(args.save_images), tag="_drop_gnn_only"
    )

    logger.info(f"\nSummary(GNN only): PSNR {psnr:.4f} | SSIM {ssim:.4f} | LPIPS {lp:.4f} | FPS {fps:.4f}")


if __name__ == "__main__":
    main()
