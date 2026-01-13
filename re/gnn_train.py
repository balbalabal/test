#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import time
import argparse
import torch
import torch.nn.functional as F

# 项目依赖
from scene.gaussian_model import GaussianModel
from utils.gnn_inpaint import GNNInpaintor  # vanilla 后端
# PYG 后端按需导入（训练时才会用）
try:
    from utils.pyg_inpaint import PYGInpaintor  # noqa: F401
    _PYG_AVAILABLE = True
except Exception:
    _PYG_AVAILABLE = False

# ------------------------
# 配置解析/超参探测
# ------------------------

def parse_cfg_args(cfg_path: str) -> dict:
    """
    兼容两种格式：
      1) 纯 JSON
      2) argparse.Namespace(...) 字符串：仅提取简单值（数值/布尔/字符串），忽略对象/类等复杂项
    """
    try:
        s = open(cfg_path, "r").read().strip()
    except Exception:
        return {}

    # 先尝试 JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 再尝试 Namespace(...) 轻量解析
    if s.startswith("Namespace(") and s.endswith(")"):
        body = s[len("Namespace("):-1]
        patt = re.compile(r"([a-zA-Z_]\w*)=([+-]?\d+\.\d+|[+-]?\d+|True|False|'[^']*'|\"[^\"]*\")")
        out = {}
        for k, v in patt.findall(body):
            try:
                out[k] = ast.literal_eval(v)
            except Exception:
                out[k] = v.strip("'").strip('"')
        return out

    return {}


def infer_log2_from_table_rows(T: int, res_list, dim: int, lmin=10, lmax=22) -> int:
    """
    给定参数表总行数 T、分辨率列表 res_list、维度 dim，反推 log2_hashmap_size。
    """
    best_l, best_err = None, None
    for l in range(lmin, lmax + 1):
        S = sum(min(2**l, (r**dim)) for r in res_list)
        err = abs(S - T)
        if best_err is None or err < best_err:
            best_l, best_err = l, err
        if err == 0:
            return l
    print(f"[auto] 警告：无法精确匹配 T={T}，采用最接近的 log2={best_l} (误差={best_err})")
    return best_l


def autodetect_hyper_from_mlp_ckpt(mlp_ckpt: str, cfg: dict) -> dict:
    """
    从 MLP ckpt 中自动对齐关键超参：
      - n_offsets, feat_dim（opacity_mlp.2.weight）
      - n_features_per_level（encoding_xyz.params 的列数）
      - log2/log2_2D（encoding_*.params 的行数 + 默认分辨率列表）
    """
    ckpt = torch.load(mlp_ckpt, map_location="cpu")

    # 1) n_offsets, feat_dim
    w = ckpt["opacity_mlp"]["2.weight"]  # [n_offsets, feat_dim]
    n_offsets_ckpt = int(w.shape[0])
    feat_dim_ckpt = int(w.shape[1])

    # 2) mix_3D2D encoding 参数矩阵形状
    enc_sd = ckpt["encoding_xyz"]
    s3 = enc_sd["encoding_xyz.params"].shape  # [T3, D]
    s2 = enc_sd["encoding_xy.params"].shape   # [T2, D]
    D_ckpt = int(s3[1])                       # n_features_per_level

    # 3) 使用 GaussianModel 里默认分辨率列表
    RES3 = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514)
    RES2 = (130, 258, 514, 1026)
    log2_3 = infer_log2_from_table_rows(int(s3[0]), RES3, dim=3)
    log2_2 = infer_log2_from_table_rows(int(s2[0]), RES2, dim=2)

    cfg = dict(cfg)
    cfg["n_offsets"] = n_offsets_ckpt
    cfg["feat_dim"] = feat_dim_ckpt
    cfg["n_features"] = D_ckpt
    cfg["log2"] = log2_3
    cfg["log2_2D"] = log2_2
    cfg["use_2D"] = True
    print(f"[auto] 使用 ckpt 对齐超参: n_offsets={n_offsets_ckpt}, feat_dim={feat_dim_ckpt}, "
          f"n_features={D_ckpt}, log2={log2_3}, log2_2D={log2_2}")
    return cfg


def build_gaussian_model(cfg: dict, device="cuda"):
    """
    按 cfg 构建 GaussianModel；关键哈希编码超参来自 cfg_args 或 ckpt 自动对齐。
    """
    gm = GaussianModel(
        feat_dim=int(cfg.get("feat_dim", 50)),
        n_offsets=int(cfg.get("n_offsets", 5)),
        voxel_size=float(cfg.get("voxel_size", 0.01)),
        update_depth=int(cfg.get("update_depth", 3)),
        update_init_factor=int(cfg.get("update_init_factor", 100)),
        update_hierachy_factor=int(cfg.get("update_hierachy_factor", 4)),
        use_feat_bank=bool(cfg.get("use_feat_bank", False)),
        n_features_per_level=int(cfg.get("n_features", 2)),
        log2_hashmap_size=int(cfg.get("log2", 19)),
        log2_hashmap_size_2D=int(cfg.get("log2_2D", 17)),
        use_2D=bool(cfg.get("use_2D", True)),
        decoded_version=False,
    ).to(device)
    return gm


@torch.no_grad()
def decode_full_scene(gm: GaussianModel, bitstreams_dir: str, mlp_ckpt: str):
    """
    加载 MLP+Encoding 权重 -> 用 bitstreams 完整解码 -> 返回场景锚点与属性。
    """
    assert os.path.isfile(mlp_ckpt), f"MLP权重不存在: {mlp_ckpt}"
    assert os.path.isdir(bitstreams_dir), f"比特流目录不存在: {bitstreams_dir}"
    gm.load_mlp_checkpoints(mlp_ckpt)
    log_info = gm.conduct_decoding(pre_path_name=bitstreams_dir)
    print(log_info)

    x = gm._anchor.detach()
    feat = gm._anchor_feat.detach()
    scale = gm._scaling.detach()
    offsets = gm._offset.detach()
    mask = gm._mask.detach()
    return x, feat, scale, offsets, mask


def build_group_ids_by_morton(x: torch.Tensor, voxel_size: float, bitstreams_dir: str):
    """
    按 Morton 顺序把锚点等分成 G 组，其中 G = .pak 文件个数（近似训练时的打包组数）。
    """
    import glob
    try:
        from utils.gpcc_utils import calculate_morton_order
        has_morton = True
    except Exception:
        has_morton = False

    pak_files = glob.glob(os.path.join(bitstreams_dir, "*.pak"))
    G = len(pak_files) if len(pak_files) > 0 else 128

    if has_morton:
        x_cpu = x.detach().cpu()
        gi = torch.round(x_cpu / voxel_size).to(torch.int32).cuda()
        order = calculate_morton_order(gi)  # cuda tensor of indices
        order_cpu = order.detach().cpu()
    else:
        # 退化：按原序均分
        order_cpu = torch.arange(x.shape[0], dtype=torch.long)

    N = x.shape[0]
    group_ids = torch.empty(N, dtype=torch.long)
    for rank, idx in enumerate(order_cpu):
        g = min(G - 1, (rank * G) // N)
        group_ids[idx] = g
    return group_ids  # cpu long tensor

# ------------------------
# CLI
# ------------------------

def make_argparser():
    p = argparse.ArgumentParser("Scene-specific GNN inpainting offline trainer (self-supervised).")
    # 路径/基础
    p.add_argument("--model_path", type=str, required=True, help="输出目录（内含 cfg_args）")
    p.add_argument("--bitstreams", type=str, default=None, help="比特流目录，默认=model_path/bitstreams")
    p.add_argument("--mlp_ckpt", type=str, required=True, help="conduct_encoding 保存的 MLP权重路径")
    p.add_argument("--save", type=str, default=None, help="GNN权重保存路径，默认=model_path/scene_gnn.pth")

    # 训练超参
    p.add_argument("--steps", type=int, default=10000, help="自监督总步数")
    p.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    p.add_argument("--k", type=int, default=24, help="kNN 邻居数（16~32）")
    p.add_argument("--hidden", type=int, default=128, help="GNN 隐层宽度")
    p.add_argument("--max_known", type=int, default=30000, help="参与训练的已知锚点上限（控显存）")
    p.add_argument("--batch_q", type=int, default=2048, help="每步查询批大小（可 2048~4096）")

    # 可观测性/稳定性
    p.add_argument("--log_every", type=int, default=50, help="每多少步打印一次 loss")
    p.add_argument("--val_every", type=int, default=200, help="每多少步做一次轻量验证")
    p.add_argument("--val_ratio", type=float, default=0.1, help="验证抽样比例（0 关闭验证）")
    p.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪阈值（<=0 关闭）")
    p.add_argument("--cosine_decay", type=int, default=1, help="1 使用余弦退火调度；0 不使用")
    p.add_argument("--group_rate", type=float, default=0.2, help="每步按组采样时，用于查询的组比例")

    # GNN 结构超参（命令行直传，评估端会从 ckpt meta 自动读取）
    p.add_argument("--gnn_backend", type=str, default="pyg", choices=["vanilla", "pyg"], help="GNN 后端")
    p.add_argument("--gnn_conv", type=str, default="edge", choices=["edge", "gat"], help="PYG 模式卷积类型")
    p.add_argument("--gnn_mp", type=int, default=1, help="消息传递层数（1 或 2）")
    p.add_argument("--gnn_k2", type=int, default=0, help="第二层 k（<=0 表示与 --k 相同）")
    p.add_argument("--gnn_heads", type=int, default=2, help="GATv2 多头数（仅 --gnn_conv gat 生效）")

    p.add_argument("--seed", type=int, default=42)

    return p

# ------------------------
# main
# ------------------------

def main():
    args = make_argparser().parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_path = os.path.abspath(args.model_path)
    bitstreams_dir = os.path.abspath(args.bitstreams or os.path.join(model_path, "bitstreams"))
    save_path = os.path.abspath(args.save or os.path.join(model_path, "scene_gnn.pth"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cfg_path = os.path.join(model_path, "cfg_args")
    cfg = parse_cfg_args(cfg_path)
    if not cfg:
        print(f"[WARN] 解析 {cfg_path} 失败，将依赖自动探测/默认超参")

    # 从 ckpt 自动对齐哈希编码与 MLP 相关关键超参
    try:
        cfg = autodetect_hyper_from_mlp_ckpt(args.mlp_ckpt, cfg)
    except Exception as e:
        print(f"[WARN] 无法从 {args.mlp_ckpt} 自动对齐关键超参：{e}，将使用 cfg/默认值，可能导致形状不匹配")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gm = build_gaussian_model(cfg, device=device)

    # 完整解码 -> 获取自监督GT
    t0 = time.time()
    x, feat, scale, offsets, mask = decode_full_scene(gm, bitstreams_dir, args.mlp_ckpt)
    N = x.shape[0]
    known_mask = torch.ones(N, dtype=torch.bool, device=device)
    print(f"[info] decoded anchors: {N}, feat_dim={feat.shape[1]}, n_offsets={offsets.shape[1]}")

    # 组 id 构造（Morton 等分，组数=.pak 数量）
    group_ids = build_group_ids_by_morton(x, gm.voxel_size, bitstreams_dir)

    # 实例化 inpaintor（命令行参数直传）
    backend = args.gnn_backend.lower()
    if backend == "pyg":
        # from utils.pyg_inpaint import PYGInpaintor
        gnn = PYGInpaintor(
            k=args.k,
            hidden=args.hidden,
            device=device,
            mp_iters=args.gnn_mp,
            conv_type=args.gnn_conv,
            k2=args.gnn_k2,
            heads=args.gnn_heads,
        )
    else:
        # from utils.gnn_inpaint import GNNInpaintor
        gnn = GNNInpaintor(k=args.k, hidden=args.hidden, device=device)

    print(f"[train] backend={backend}, conv={args.gnn_conv}, mp={args.gnn_mp}, "
          f"k={args.k}, k2={(args.gnn_k2 if args.gnn_k2 > 0 else args.k)}, "
          f"hidden={args.hidden}, heads={args.gnn_heads}")

    # 训练
    print(f"[train] start self-supervised fit: steps={args.steps}, lr={args.lr}")
    gnn.fit(
        gm=gm,
        x_all=x,
        known_mask=known_mask,
        feat=feat,
        scale=scale,
        offsets=offsets,
        mask=mask,
        steps=args.steps,
        lr=args.lr,
        k=args.k,
        max_known=args.max_known,
        batch_q=args.batch_q,
        log_every=args.log_every,
        val_every=args.val_every,
        val_ratio=args.val_ratio,
        clip_grad=args.clip_grad,
        cosine_decay=bool(args.cosine_decay),
        group_ids=group_ids,
        group_rate=args.group_rate,
    )
    t1 = time.time()
    print(f"[train] done, elapsed {t1 - t0:.1f}s")

    # 保存权重 + meta
    k2_eff = args.gnn_k2 if args.gnn_k2 > 0 else args.k
    torch.save({
        "state_dict": gnn.state_dict(),
        "backend": args.gnn_backend.lower(),
        "k": args.k,
        "hidden": args.hidden,
        "mp_iters": args.gnn_mp,
        "conv": (args.gnn_conv.lower() if args.gnn_backend.lower() == "pyg" else "edge"),
        "k2": (args.gnn_k2 if args.gnn_k2 is not None else args.k),
        "heads": (args.gnn_heads if args.gnn_backend.lower() == "pyg" else 1),
    }, save_path)
    print(f"[done] saved GNN weights to: {save_path}")

    print("评估端将优先读取权重 meta 自动构造 inpaintor；也可用 HAC_GNN_* 覆盖。")


if __name__ == "__main__":
    main()
