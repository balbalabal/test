import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from train import get_logger

# ---------- Utils ----------

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sample_drop_masks(N, p_feat=0.3, p_scal=0.0, p_offs=0.0, seed=0, device='cuda'):
    """锚点级独立伯努利缺失掩码"""
    g = torch.Generator(device=device); g.manual_seed(seed)
    miss_feat = (torch.rand((N, 5), generator=g, device=device) < p_feat).to(torch.bool)
    miss_scal = (torch.rand((N, 1), generator=g, device=device) < p_scal).to(torch.bool).squeeze(1)
    miss_offs = (torch.rand((N, 1), generator=g, device=device) < p_offs).to(torch.bool).squeeze(1)
    return miss_feat, miss_scal, miss_offs

def space_topL(pos_miss, pos_all_avail, L, AvC=8192):
    """分块空间 Top-L 候选；返回 (topd, topi) [B,L]"""
    B = pos_miss.shape[0]; Ma = pos_all_avail.shape[0]; device = pos_miss.device
    topd = torch.full((B, L), float('inf'), device=device)
    topi = torch.full((B, L), -1, dtype=torch.long, device=device)
    for as_ in range(0, Ma, AvC):
        ae = min(as_ + AvC, Ma)
        d = torch.cdist(pos_miss, pos_all_avail[as_:ae])  # [B,Ac]
        d_all = torch.cat([topd, d], dim=1)
        i_block = torch.arange(as_, ae, device=device, dtype=torch.long)
        i_all = torch.cat([topi, i_block.unsqueeze(0).expand(B, -1)], dim=1)
        newd, newk = torch.topk(d_all, k=L, largest=False)
        newi = torch.gather(i_all, 1, newk)
        topd, topi = newd, newi
        del d, d_all, i_block, i_all, newd, newk, newi
    return topd, topi

def build_mixture_sigma(mean_cc, scale_cc, mean_adj, scale_adj, prob_cc, eps=1e-9):
    """近似混合方差（按段取均值）"""
    s1 = torch.clamp(scale_cc, min=eps); s2 = torch.clamp(scale_adj, min=eps)
    p = torch.clamp(prob_cc, 1e-6, 1-1e-6)
    var = p*(s1**2) + (1-p)*(s2**2) + p*(1-p)*((mean_cc-mean_adj)**2)
    return torch.sqrt(var.mean(dim=1) + eps)

# interleave meta helpers（用于 joint/mix）
def splitmix64_numpy(x: np.ndarray) -> np.ndarray:
    x = np.uint64(x)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(30))
    x = (x * np.uint64(0xbf58476d1ce4e5b9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27))
    x = (x * np.uint64(0x94d049bb133111eb)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(31))
    return x

def lanes_for_range_numpy(start: int, count: int, M: int, seed: int):
    idx = (np.arange(start, start + count, dtype=np.uint64) ^ np.uint64(seed))
    h = splitmix64_numpy(idx)
    lanes = (h % np.uint64(M)).astype(np.int64)
    buckets = [np.where(lanes == m)[0] for m in range(M)]
    return buckets

def sample_joint_masks(N, meta, joint_pct, seed):
    """step-lane 组联合丢包：返回 miss_feat[N,5], miss_scal[N], miss_offs[N]"""
    M = int(meta["M"]); steps = int(meta["steps"]); SEED = int(meta["seed"]); MAX_bs = int(meta["max_batch_size"])
    rng = np.random.default_rng(seed)
    groups = []
    for s in range(steps):
        N_start, N_end = s*MAX_bs, min((s+1)*MAX_bs, N)
        if N_end <= N_start: continue
        buckets = lanes_for_range_numpy(N_start, N_end-N_start, M, SEED)
        for m in range(M):
            if len(buckets[m]) == 0: continue
            idx_global = (np.arange(N_start, N_end, dtype=np.int64))[buckets[m]]
            groups.append(idx_global)
    G = len(groups)
    if G == 0:
        miss_feat = torch.zeros((N,5), dtype=torch.bool, device='cuda')
        miss_scal = torch.zeros((N,), dtype=torch.bool, device='cuda')
        miss_offs = torch.zeros((N,), dtype=torch.bool, device='cuda')
        return miss_feat, miss_scal, miss_offs

    k = max(1, int(round(G * joint_pct)))
    sel = rng.choice(G, size=k, replace=False)
    miss_feat = torch.zeros((N,5), dtype=torch.bool, device='cuda')
    miss_scal = torch.zeros((N,), dtype=torch.bool, device='cuda')
    miss_offs = torch.zeros((N,), dtype=torch.bool, device='cuda')
    for g in sel:
        ids = torch.from_numpy(groups[g]).to('cuda', dtype=torch.long)
        miss_feat[ids,:] = True
        miss_scal[ids]   = True
        miss_offs[ids]   = True
    return miss_feat, miss_scal, miss_offs

# ---------- Finetune EC Head ----------

def main():
    parser = ArgumentParser(description="EC finetune for per-attribute random drop (robust)")
    # 工程参数
    lp = ModelParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    # 微调配置
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--min_lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--bs_anchor", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1337)

    # 缺失分布：默认 anchor（独立）各 20%；可混入 joint 提升稳健性
    parser.add_argument("--drop_mode", type=str, default="anchor", choices=["anchor","joint","mix"],
                        help="anchor: 各属性独立伯努利; joint: step-lane 联合; mix: 两者混合")
    parser.add_argument("--p_feat", type=float, default=0.20)
    parser.add_argument("--p_scal", type=float, default=0.20)
    parser.add_argument("--p_offs", type=float, default=0.20)
    parser.add_argument("--joint_pct", type=float, default=0.20,
                        help="joint 的组丢包率（mix 模式用作 joint 部分比例）")
    parser.add_argument("--mix_joint", type=float, default=0.30,
                        help="mix 模式下 joint 采样的概率（其余为 anchor）")

    # 损失权重
    parser.add_argument("--lambda_feat", type=float, default=1.0)
    parser.add_argument("--lambda_scal", type=float, default=1.0)
    parser.add_argument("--lambda_offs", type=float, default=0.7)

    # 预热解码
    parser.add_argument("--decode_first", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    # 日志
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)
    set_all_seeds(args.seed)

    # 模型/场景
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
    ).cuda()
    _ = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    # 解码预热（载入属性）
    if args.decode_first:
        bit_dir = os.path.join(model_path, "bitstreams")
        if not os.path.isdir(bit_dir):
            logger.info(f"[EC] bitstreams not found: {bit_dir}"); sys.exit(1)
        logger.info("[EC] conduct_decoding to preload attributes...")
        logger.info(gaussians.conduct_decoding(pre_path_name=bit_dir))

    # 加载 stage-1 权重（含 ec_head；gaussian_model.py 内已兼容部分加载）
    logger.info(f"[EC] load checkpoint: {args.checkpoint}")
    gaussians.load_mlp_checkpoints(args.checkpoint)

    # 冻结 backbone，只训 ec_head
    for p in gaussians.encoding_xyz.parameters(): p.requires_grad = False
    if hasattr(gaussians, "mlp_grid"):
        for p in gaussians.mlp_grid.parameters(): p.requires_grad = False
    if hasattr(gaussians, "mlp_deform"):
        for p in gaussians.mlp_deform.parameters(): p.requires_grad = False
    gaussians.ec_head.train()

    optim = torch.optim.AdamW(gaussians.ec_head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    # warmup + cosine
    def lr_lambda(step):
        if step < args.warmup:
            return max(1e-3, (step+1)/max(1,args.warmup))
        t = step - args.warmup
        T = max(1, args.iters - args.warmup)
        cos = 0.5 * (1 + np.cos(np.pi * t / T))
        return args.min_lr/args.lr + (1 - args.min_lr/args.lr) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # interleave meta（joint/mix 用）
    bit_dir = os.path.join(model_path, "bitstreams")
    meta_path = os.path.join(bit_dir, "interleave_meta.json")
    inter_meta = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            inter_meta = json.load(f)

    # 全量静态量
    with torch.no_grad():
        anchor = gaussians.get_anchor
        N = anchor.shape[0]
        fh_all = gaussians.calc_interp_feat(anchor)  # [N, D_fh]

        mean_all, scale_all, prob_all, mean_scal_all, scale_scal_all, mean_offs_all, scale_offs_all, \
        Q_feat_adj_all, Q_scal_adj_all, Q_offs_adj_all = torch.split(
            gaussians.get_grid_mlp(fh_all),
            split_size_or_sections=[gaussians.feat_dim, gaussians.feat_dim, gaussians.feat_dim,
                                    6, 6, 3 * gaussians.n_offsets, 3 * gaussians.n_offsets,
                                    1, 1, 1],
            dim=-1
        )
        scale_all = torch.clamp(scale_all, min=1e-9)

        gt_feat = gaussians._anchor_feat.detach()
        gt_scal = gaussians._scaling.detach()
        gt_offs = gaussians._offset.detach()
        gt_mask = gaussians._mask[:, :gaussians.n_offsets, :].detach().to(torch.bool)

        # 逐维 sigma（offsets），形状 [N, Koff, 3]
        sigma_off_pd_all = scale_offs_all.view(N, gaussians.n_offsets, 3).clamp_min(1e-9)

        fh_dim = fh_all.shape[1]
        ctx_dim_req = gaussians.ec_head.q_enc[0].in_features
        sig_dim = max(0, ctx_dim_req - fh_dim)
        logger.info(f"[EC] N={N}, fh_dim={fh_dim}, ec_head.ctx_dim={ctx_dim_req}, sig_dim={sig_dim}")

    # 训练超参
    L = int(os.getenv("HAC_EC_L", "256"))   # 独立缺失更稳
    Bm = int(os.getenv("HAC_EC_MB", "2048"))
    AvC = int(os.getenv("HAC_EC_AVC", "8192"))
    beta_huber = float(os.getenv("HAC_EC_BETA", "0.05"))
    clip_norm = float(os.getenv("HAC_EC_CLIP", "1.0"))
    nei_drop_p = float(os.getenv("HAC_EC_NEI_DROP", "0.2"))      # 邻居 dropout 概率
    sig_noise = float(os.getenv("HAC_EC_SIG_NOISE", "0.02"))     # 签名噪声 std（0~0.05）

    t0 = time.time()
    for it in range(1, args.iters+1):
        set_all_seeds(args.seed + it)

        # 缺失采样：anchor / joint / mix
        if args.drop_mode == "anchor":
            miss_feat, miss_scal, miss_offs = sample_drop_masks(
                N, p_feat=args.p_feat, p_scal=args.p_scal, p_offs=args.p_offs, seed=args.seed+it, device='cuda'
            )
        elif args.drop_mode == "joint" and inter_meta is not None:
            miss_feat, miss_scal, miss_offs = sample_joint_masks(N, inter_meta, args.joint_pct, args.seed+it)
        elif args.drop_mode == "mix" and inter_meta is not None and (random.random() < args.mix_joint):
            miss_feat, miss_scal, miss_offs = sample_joint_masks(N, inter_meta, args.joint_pct, args.seed+it)
        else:
            miss_feat, miss_scal, miss_offs = sample_drop_masks(
                N, p_feat=args.p_feat, p_scal=args.p_scal, p_offs=args.p_offs, seed=args.seed+it, device='cuda'
            )

        miss_any = miss_scal | miss_offs | miss_feat.any(dim=1)
        miss_idx_all = miss_any.nonzero(as_tuple=False).view(-1)
        if miss_idx_all.numel() == 0:
            scheduler.step(); continue
        if miss_idx_all.numel() > args.bs_anchor:
            sel = torch.randperm(miss_idx_all.numel(), device='cuda')[:args.bs_anchor]
            miss_idx_all = miss_idx_all[sel]

        # 逐元素加权 SmoothL1 累计
        feat_loss_sum = torch.tensor(0.0, device='cuda'); feat_w_sum = torch.tensor(0.0, device='cuda')
        scal_loss_sum = torch.tensor(0.0, device='cuda'); scal_w_sum = torch.tensor(0.0, device='cuda')
        offs_loss_sum = torch.tensor(0.0, device='cuda'); offs_w_sum = torch.tensor(0.0, device='cuda')

        with torch.no_grad():
            # 签名统计
            feat_mu_full = mean_all.view(N, 5, 10)
            feat_res_full = (gt_feat.view(N, 5, 10) - feat_mu_full)
            feat_res_mu = feat_res_full.mean(dim=2)
            feat_res_std = feat_res_full.std(dim=2)
            scal_res_full = (gt_scal - mean_scal_all)
            scal_res_mu  = scal_res_full.mean(dim=1, keepdim=True)
            scal_res_std = scal_res_full.std(dim=1, keepdim=True)
            off_pred = mean_offs_all.view(N, gaussians.n_offsets, 3)
            off_res_abs = (gt_offs - off_pred).abs()
            mt_full = gt_mask.expand_as(off_res_abs)
            num = (off_res_abs * mt_full).sum(dim=(1,2))
            den = mt_full.to(torch.float32).sum(dim=(1,2)).clamp_min(1.0)
            off_res_mu  = (num / den).unsqueeze(1)

        pos_all = anchor
        fea_est = gt_feat.clone()
        mean_scale_cat_all = torch.cat([mean_all, scale_all, prob_all], dim=-1)

        # ---------- feat：按 cc 段 ----------
        for cc in range(5):
            miss_mask_cc = miss_feat[:, cc]
            miss_idx_cc = (miss_mask_cc & torch.isin(torch.arange(N, device='cuda'), miss_idx_all)).nonzero(as_tuple=False).view(-1)
            if miss_idx_cc.numel() == 0: continue

            with torch.no_grad():
                mean_adj_all, scale_adj_all, prob_adj_all = gaussians.get_deform_mlp.forward(fea_est, mean_scale_cat_all, to_dec=cc)
                scale_adj_all = torch.clamp(scale_adj_all, min=1e-9)
                probs_cc = torch.stack([prob_all[:, cc*10:cc*10+10], prob_adj_all], dim=-1)
                probs_cc = torch.softmax(probs_cc, dim=-1)
                mu_cc_all = (probs_cc[..., 0] * mean_all[:, cc*10:cc*10+10] +
                             probs_cc[..., 1] * mean_adj_all)
                sigma_cc = build_mixture_sigma(mean_all[:, cc*10:cc*10+10], scale_all[:, cc*10:cc*10+10],
                                               mean_adj_all, scale_adj_all, probs_cc[..., 1])

            avail_idx_cc = (~miss_mask_cc).nonzero(as_tuple=False).view(-1)
            if avail_idx_cc.numel() == 0:
                pred = mu_cc_all[miss_idx_cc]
                err  = F.smooth_l1_loss(pred, gt_feat[miss_idx_cc, cc*10:cc*10+10], reduction='none', beta=beta_huber)
                w    = 1.0 / (sigma_cc[miss_idx_cc].unsqueeze(1) + 1e-9)
                feat_loss_sum += (w.repeat(1,10) * err).sum()
                feat_w_sum    += (w.sum() * 10)
                continue

            Btot = miss_idx_cc.numel()
            for ms in range(0, Btot, Bm):
                me = min(ms + Bm, Btot); ids = miss_idx_cc[ms:me]
                topd, topi = space_topL(pos_all[ids], pos_all[avail_idx_cc], L=L, AvC=AvC)
                nn_idx_L = avail_idx_cc[topi]; Lc = nn_idx_L.shape[1]

                # 签名（动态 12/6/0），可加轻微噪声增强
                if sig_dim >= 12:
                    idx_others = [k for k in range(5) if k != cc]
                    sig_miss = torch.cat([feat_res_mu[ids][:, idx_others], feat_res_std[ids][:, idx_others],
                                          scal_res_mu[ids], scal_res_std[ids], off_res_mu[ids], off_res_mu[ids]], dim=1)  # [B,12]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)][:, idx_others].view(me-ms, Lc, 4),
                                          feat_res_std[nn_idx_L.view(-1)][:, idx_others].view(me-ms, Lc, 4),
                                          scal_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          scal_res_std[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)   # [B,Lc,12]
                elif sig_dim >= 6:
                    idx_others = [k for k in range(5) if k != cc]
                    sig_miss = torch.cat([feat_res_mu[ids][:, idx_others], scal_res_mu[ids], off_res_mu[ids]], dim=1)  # [B,6]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)][:, idx_others].view(me-ms, Lc, 4),
                                          scal_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)   # [B,Lc,6]
                else:
                    sig_miss = torch.zeros((me-ms, sig_dim), device='cuda', dtype=torch.float32)
                    sig_nei  = torch.zeros((me-ms, Lc, sig_dim), device='cuda', dtype=torch.float32)

                if sig_noise > 0:
                    sig_miss = sig_miss + sig_noise * torch.randn_like(sig_miss)
                    sig_nei  = sig_nei  + sig_noise * torch.randn_like(sig_nei)

                ctx_q  = torch.cat([fh_all[ids], sig_miss], dim=1)
                ctx_kv = torch.cat([fh_all[nn_idx_L.view(-1)].view(me-ms, Lc, fh_dim), sig_nei], dim=2)

                # bias（邻居 dropout）
                dist = topd / (topd.mean(dim=1, keepdim=True) + 1e-9)
                sigma_nei = sigma_cc[nn_idx_L.view(-1)].view(me-ms, Lc)
                conf_nei  = torch.ones_like(sigma_nei)
                if nei_drop_p > 0:
                    drop_mask = (torch.rand((me-ms, Lc), device='cuda') < nei_drop_p)
                    conf_nei = conf_nei * (~drop_mask).float()
                if sig_dim > 0:
                    sig_miss_n = F.normalize(sig_miss + 1e-9, dim=1)
                    sig_nei_n  = F.normalize(sig_nei + 1e-9, dim=2)
                    sig_sim = (sig_nei_n * sig_miss_n.unsqueeze(1)).sum(dim=-1).clamp(0,1)
                else:
                    sig_sim = torch.ones_like(dist)
                bias_dict = {'dist': dist, 'sigma': sigma_nei, 'conf': conf_nei, 'sig_sim': sig_sim}

                r_hat = gaussians.ec_head.forward_feat_cc(ctx_q, ctx_kv, bias_dict)
                pred  = mu_cc_all[ids] + r_hat
                gt    = gt_feat[ids, cc*10:cc*10+10]
                err   = F.smooth_l1_loss(pred, gt, reduction='none', beta=beta_huber)
                w     = 1.0 / (sigma_cc[ids].unsqueeze(1) + 1e-9)
                feat_loss_sum += (w.repeat(1,10) * err).sum()
                feat_w_sum    += (w.sum() * 10)

        # ---------- scaling ----------
        miss_scal_idx = (miss_scal & torch.isin(torch.arange(N, device='cuda'), miss_idx_all)).nonzero(as_tuple=False).view(-1)
        if miss_scal_idx.numel() > 0:
            avail_scal_idx = (~miss_scal).nonzero(as_tuple=False).view(-1)
            sigma_scal = scale_scal_all.mean(dim=1)
            Btot = miss_scal_idx.numel()
            for ms in range(0, Btot, Bm):
                me = min(ms + Bm, Btot); ids = miss_scal_idx[ms:me]
                if avail_scal_idx.numel() == 0:
                    pred = mean_scal_all[ids]
                    err  = F.smooth_l1_loss(pred, gt_scal[ids], reduction='none', beta=beta_huber)
                    w    = 1.0 / (sigma_scal[ids].unsqueeze(1) + 1e-9)
                    scal_loss_sum += (w.repeat(1,6) * err).sum()
                    scal_w_sum    += (w.sum() * 6)
                    continue
                topd, topi = space_topL(pos_all[ids], pos_all[avail_scal_idx], L=L, AvC=AvC)
                nn_idx_L = avail_scal_idx[topi]; Lc = nn_idx_L.shape[1]
                if sig_dim >= 12:
                    sig_miss = torch.cat([feat_res_mu[ids], feat_res_std[ids],
                                          off_res_mu[ids], off_res_mu[ids]], dim=1)  # [B,12]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          feat_res_std[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)
                elif sig_dim >= 6:
                    sig_miss = torch.cat([feat_res_mu[ids], off_res_mu[ids]], dim=1)  # [B,6]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          off_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)
                else:
                    sig_miss = torch.zeros((me-ms, sig_dim), device='cuda', dtype=torch.float32)
                    sig_nei  = torch.zeros((me-ms, Lc, sig_dim), device='cuda', dtype=torch.float32)
                if sig_noise > 0:
                    sig_miss = sig_miss + sig_noise * torch.randn_like(sig_miss)
                    sig_nei  = sig_nei  + sig_noise * torch.randn_like(sig_nei)
                ctx_q  = torch.cat([fh_all[ids], sig_miss], dim=1)
                ctx_kv = torch.cat([fh_all[nn_idx_L.view(-1)].view(me-ms, Lc, fh_dim), sig_nei], dim=2)
                dist = topd / (topd.mean(dim=1, keepdim=True) + 1e-9)
                sigma_nei = sigma_scal[nn_idx_L.view(-1)].view(me-ms, Lc)
                conf_nei  = torch.ones_like(sigma_nei)
                if nei_drop_p > 0:
                    drop_mask = (torch.rand((me-ms, Lc), device='cuda') < nei_drop_p)
                    conf_nei = conf_nei * (~drop_mask).float()
                if sig_dim > 0:
                    sig_miss_n = F.normalize(sig_miss + 1e-9, dim=1)
                    sig_nei_n  = F.normalize(sig_nei + 1e-9, dim=2)
                    sig_sim = (sig_nei_n * sig_miss_n.unsqueeze(1)).sum(dim=-1).clamp(0,1)
                else:
                    sig_sim = torch.ones_like(dist)
                bias_dict = {'dist': dist, 'sigma': sigma_nei, 'conf': conf_nei, 'sig_sim': sig_sim}
                r_hat = gaussians.ec_head.forward_scal(ctx_q, ctx_kv, bias_dict)
                pred  = mean_scal_all[ids] + r_hat
                gt    = gt_scal[ids]
                err   = F.smooth_l1_loss(pred, gt, reduction='none', beta=beta_huber)
                w     = 1.0 / (sigma_scal[ids].unsqueeze(1) + 1e-9)
                scal_loss_sum += (w.repeat(1,6) * err).sum()
                scal_w_sum    += (w.sum() * 6)

        # ---------- offsets（仅 True 位；逐维 sigma 权重 + 可靠性） ----------
        miss_off_idx = (miss_offs & torch.isin(torch.arange(N, device='cuda'), miss_idx_all)).nonzero(as_tuple=False).view(-1)
        if miss_off_idx.numel() > 0:
            avail_off_idx = (~miss_offs).nonzero(as_tuple=False).view(-1)
            sigma_off_scalar = scale_offs_all.view(N, -1).mean(dim=1)  # 仅用于备选
            Btot = miss_off_idx.numel()
            for ms in range(0, Btot, Bm):
                me = min(ms + Bm, Btot); ids = miss_off_idx[ms:me]
                if avail_off_idx.numel() == 0:
                    pred = mean_offs_all.view(N, gaussians.n_offsets, 3)[ids]  # [B,Koff,3]
                    gt   = gt_offs[ids]
                    err  = F.smooth_l1_loss(pred, gt, reduction='none', beta=beta_huber)
                    mt   = gt_mask[ids]                                        # [B,Koff,1]
                    w_pd = 1.0 / (sigma_off_pd_all[ids] + 1e-9)               # [B,Koff,3]
                    offs_loss_sum += ((w_pd * err) * mt.to(err.dtype)).sum()
                    offs_w_sum    += ((w_pd * mt.to(w_pd.dtype)).sum())
                    continue

                topd, topi = space_topL(pos_all[ids], pos_all[avail_off_idx], L=L, AvC=AvC)
                nn_idx_L = avail_off_idx[topi]; Lc = nn_idx_L.shape[1]

                # 签名（动态 12/6/0）
                if sig_dim >= 12:
                    sig_miss = torch.cat([feat_res_mu[ids], feat_res_std[ids],
                                          scal_res_mu[ids], scal_res_std[ids]], dim=1)  # [B,12]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          feat_res_std[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          scal_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1),
                                          scal_res_std[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)
                elif sig_dim >= 6:
                    sig_miss = torch.cat([feat_res_mu[ids], scal_res_mu[ids]], dim=1)  # [B,6]
                    sig_nei  = torch.cat([feat_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 5),
                                          scal_res_mu[nn_idx_L.view(-1)].view(me-ms, Lc, 1)], dim=2)
                else:
                    sig_miss = torch.zeros((me-ms, sig_dim), device='cuda', dtype=torch.float32)
                    sig_nei  = torch.zeros((me-ms, Lc, sig_dim), device='cuda', dtype=torch.float32)

                if sig_noise > 0:
                    sig_miss = sig_miss + sig_noise * torch.randn_like(sig_miss)
                    sig_nei  = sig_nei  + sig_noise * torch.randn_like(sig_nei)

                ctx_q  = torch.cat([fh_all[ids], sig_miss], dim=1)
                ctx_kv = torch.cat([fh_all[nn_idx_L.view(-1)].view(me-ms, Lc, fh_dim), sig_nei], dim=2)

                # bias：距离/σ/可靠性（True 位比例）/签名 + 邻居 dropout
                dist = topd / (topd.mean(dim=1, keepdim=True) + 1e-9)
                nei_mt = gt_mask[nn_idx_L]                                              # [B,Lc,Koff,1]
                nei_ratio = nei_mt.to(torch.float32).mean(dim=(2,3)).clamp_min(0.1)     # [B,Lc]
                sigma_nei_scalar = sigma_off_pd_all[nn_idx_L.view(-1)].view(me-ms, Lc, gaussians.n_offsets, 3).mean(dim=(2,3))
                conf_nei  = nei_ratio
                if nei_drop_p > 0:
                    drop_mask = (torch.rand((me-ms, Lc), device='cuda') < nei_drop_p)
                    conf_nei = conf_nei * (~drop_mask).float()
                if sig_dim > 0:
                    sig_miss_n = F.normalize(sig_miss + 1e-9, dim=1)
                    sig_nei_n  = F.normalize(sig_nei + 1e-9, dim=2)
                    sig_sim = (sig_nei_n * sig_miss_n.unsqueeze(1)).sum(dim=-1).clamp(0,1)
                else:
                    sig_sim = torch.ones_like(dist)
                bias_dict = {'dist': dist, 'sigma': sigma_nei_scalar, 'conf': conf_nei, 'sig_sim': sig_sim}

                mt = gt_mask[ids]                                                       # [B,Koff,1]
                r_hat = gaussians.ec_head.forward_offs(ctx_q, ctx_kv, bias_dict, mt)
                pred  = mean_offs_all.view(N, gaussians.n_offsets, 3)[ids] + r_hat
                gt    = gt_offs[ids]
                err   = F.smooth_l1_loss(pred, gt, reduction='none', beta=beta_huber)   # [B,Koff,3]
                w_pd  = 1.0 / (sigma_off_pd_all[ids] + 1e-9)                            # [B,Koff,3]
                offs_loss_sum += ((w_pd * err) * mt.to(err.dtype)).sum()
                offs_w_sum    += ((w_pd * mt.to(w_pd.dtype)).sum())

        # 分属性平均 + 权重汇总
        feat_mean = (feat_loss_sum / feat_w_sum.clamp_min(1e-9)) if feat_w_sum > 0 else torch.tensor(0.0, device='cuda')
        scal_mean = (scal_loss_sum / scal_w_sum.clamp_min(1e-9)) if scal_w_sum > 0 else torch.tensor(0.0, device='cuda')
        offs_mean = (offs_loss_sum / offs_w_sum.clamp_min(1e-9)) if offs_w_sum > 0 else torch.tensor(0.0, device='cuda')
        loss = (args.lambda_feat * feat_mean + args.lambda_scal * scal_mean + args.lambda_offs * offs_mean)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gaussians.ec_head.parameters(), max_norm=clip_norm)
        optim.step(); scheduler.step()

        if it % 50 == 0:
            # 梯度范数观测
            with torch.no_grad():
                gn = 0.0
                for p in gaussians.ec_head.parameters():
                    if p.grad is not None:
                        gn += float(p.grad.detach().pow(2).sum().cpu())
                gn = float(np.sqrt(gn))
            cur_lr = optim.param_groups[0]['lr']
            logger.info(f"[EC] it={it:05d} | loss={loss.item():.4f} "
                        f"| feat={feat_mean.item():.4f} | scal={scal_mean.item():.4f} | offs={offs_mean.item():.4f} "
                        f"| lr={cur_lr:.4e} | grad={gn:.3f}")

    out_ck = os.path.join(model_path, "mlp_ckpt_ec_finetuned.pth")
    gaussians.save_mlp_checkpoints(out_ck)
    logger.info(f"[EC] finetune done. saved: {out_ck} | time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
