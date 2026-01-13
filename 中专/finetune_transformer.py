import os, sys, json, time, random, glob
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel  # 确保导入你修改的模型
from train import get_logger

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

def build_step_lane_groups(N, meta):
    M = int(meta["M"]); steps = int(meta["steps"]); SEED = int(meta["seed"]); MAX_bs = int(meta["max_batch_size"])
    groups = []
    for s in range(steps):
        N_start, N_end = s*MAX_bs, min((s+1)*MAX_bs, N)
        if N_end <= N_start: continue
        buckets = lanes_for_range_numpy(N_start, N_end-N_start, M, SEED)
        for m in range(M):
            if len(buckets[m]) == 0: continue
            ids = (np.arange(N_start, N_end, dtype=np.int64))[buckets[m]]
            groups.append(ids)
    return groups
def sample_group_masks(N, meta, p_group=0.2, seed=0):
    """
    按组(step-lane)整组丢失：随机抽 p_group 比例的 group，
    返回 miss_anchor: [N]，被抽中的组内所有锚点为 True。
    """
    rng = np.random.default_rng(seed)
    groups = build_step_lane_groups(N, meta)
    G = len(groups)
    miss_anchor = np.zeros(N, dtype=np.bool_)
    if G == 0:
        return miss_anchor
    k = max(1, int(round(G * p_group)))
    sel = rng.choice(G, size=min(k, G), replace=False)
    for g in sel:
        miss_anchor[groups[g]] = True
    return miss_anchor

def sample_attrjoint_masks(N, meta, p_feat=0.2, p_scal=0.2, p_offs=0.2, seed=0):
    rng = np.random.default_rng(seed)
    groups = build_step_lane_groups(N, meta)
    G = len(groups)
    miss = np.zeros((N,3), dtype=np.bool_)  # [feat,scal,offs] 标志
    if G == 0:
        return miss
    # feat: 对每个 cc 都是整锚点缺失（在训练中我们只训练整锚点，因此合并为 feat 缺失）
    kf = max(1, int(round(G * p_feat)))
    ks = max(1, int(round(G * p_scal)))
    ko = max(1, int(round(G * p_offs)))
    for g in rng.choice(G, size=min(kf,G), replace=False):
        miss[groups[g], 0] = True
    for g in rng.choice(G, size=min(ks,G), replace=False):
        miss[groups[g], 1] = True
    for g in rng.choice(G, size=min(ko,G), replace=False):
        miss[groups[g], 2] = True
    return miss

def main():
    parser = ArgumentParser(description="Finetune ECTransformer for group-level missing (attrjoint)")
    lp = ModelParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--bs_anchor", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--drop_mode", type=str, default="group", choices=["group", "attrjoint"])
    parser.add_argument("--p_group", type=float, default=0.20,
                        help="group 模式的整组丢包比例（与 .pak 一致）")
    parser.add_argument("--p_feat", type=float, default=0.2)
    parser.add_argument("--p_scal", type=float, default=0.2)
    parser.add_argument("--p_offs", type=float, default=0.2)
    parser.add_argument("--num_seeds", type=int, default=5, help="多 seed 数量，用于梯度累积")

    args = parser.parse_args(sys.argv[1:])

    os.environ.setdefault("HAC_PAK_ENABLE", "1")  # 让解码能解 .pak
    model_path = args.model_path
    logger = get_logger(model_path)
    set_all_seeds(args.seed)

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

    # 预解码（读入属性）
    bit_dir = os.path.join(model_path, "bitstreams")
    if not os.path.isdir(bit_dir):
        logger.info(f"[EC-T] bitstreams not found: {bit_dir}"); sys.exit(1)
    logger.info("[EC-T] conduct_decoding preload ...")
    logger.info(gaussians.conduct_decoding(pre_path_name=bit_dir))

    # 加载 stage-1 backbone 权重（含 ec_transformer 初始化）
    gaussians.load_mlp_checkpoints(args.checkpoint)
    # 冻结 backbone，只训 transformer
    for m in [gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color, gaussians.mlp_grid, gaussians.mlp_deform]:
        for p in m.parameters(): p.requires_grad = False
    for p in gaussians.encoding_xyz.parameters(): p.requires_grad = False
    for n, p in gaussians.named_parameters():
        if n.startswith("ec_transformer"):
            p.requires_grad = True

    optim = torch.optim.AdamW(gaussians.ec_transformer.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(step):
        if step < args.warmup:
            return (step+1)/max(1,args.warmup)
        t = step - args.warmup
        T = max(1, args.iters - args.warmup)
        cos = 0.5 * (1 + np.cos(np.pi * t / T))
        return args.min_lr/args.lr + (1 - args.min_lr/args.lr) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # 训练所需的全量静态量
    with torch.no_grad():
        anchor = gaussians.get_anchor
        N = anchor.shape[0]
        fh_all = gaussians.calc_interp_feat(anchor)  # [N,Fh]
        mean_all, scale_all, prob_all, mean_scal_all, scale_scal_all, mean_offs_all, scale_offs_all, \
        Q_feat_adj_all, Q_scal_adj_all, Q_offs_adj_all = torch.split(
            gaussians.get_grid_mlp(fh_all),
            split_size_or_sections=[gaussians.feat_dim, gaussians.feat_dim, gaussians.feat_dim,
                                    6, 6, 3 * gaussians.n_offsets, 3 * gaussians.n_offsets,
                                    1, 1, 1],
            dim=-1
        )
        scale_all = torch.clamp(scale_all, min=1e-9)
        feat_gt = gaussians._anchor_feat.detach()                      # [N,50]
        scal_gt = gaussians._scaling.detach()                          # [N,6]
        off_gt  = gaussians._offset.detach()                           # [N,n_off,3]
        mask_gt = gaussians._mask[:, :gaussians.n_offsets, :].detach() # [N,n_off,1]
        fh_dim = fh_all.shape[1]

        # 预计算 mu_feat_all
        mean_scale_cat_all = torch.cat([mean_all, scale_all, prob_all], dim=-1)
        mu_feat_all = torch.zeros((N, gaussians.feat_dim), device='cuda', dtype=torch.float32)
        for cc in range(5):
            mean_adj_all, scale_adj_all, prob_adj_all = gaussians.get_deform_mlp.forward(
                feat_gt, mean_scale_cat_all, to_dec=cc
            )
            probs_cc = torch.stack([prob_all[:, cc*10:cc*10+10], prob_adj_all], dim=-1)
            probs_cc = torch.softmax(probs_cc, dim=-1)
            mu_cc = probs_cc[...,0]*mean_all[:, cc*10:cc*10+10] + probs_cc[...,1]*mean_adj_all
            mu_feat_all[:, cc*10:cc*10+10] = mu_cc
        mu_scal_all = mean_scal_all
        mu_off_all  = mean_offs_all

        # 读取 interleave 元数据，供 attrjoint 随机丢组用
        meta = json.load(open(os.path.join(bit_dir, "interleave_meta.json"), "r"))

    # KNN 辅助
    def space_topK(pos_miss, pos_av, K):
        d = torch.cdist(pos_miss, pos_av)
        topd, topi = torch.topk(d, k=min(K, d.shape[1]), largest=False)
        return topd, topi

    # 建议：若上文未定义 K，这里用环境变量给个默认值；也可在循环外提前定义
    K = int(os.getenv("HAC_EC_K", "16"))
    num_seeds = getattr(args, "num_seeds", 5)  # 多 seed 梯度累积，默认 5

    t0 = time.time()
    for it in range(1, args.iters + 1):
        set_all_seeds(args.seed + it)

        # 累积打印用
        accum_loss = 0.0
        accum_l_feat = 0.0
        accum_l_scal = 0.0
        accum_l_offs = 0.0
        accum_l_mask = 0.0
        update_count = 0

        optim.zero_grad(set_to_none=True)

        for seed_offset in range(num_seeds):
            sub_seed = args.seed + it + seed_offset

            # 采样“组级（step-lane）整锚点缺失”
            if args.drop_mode == "group":
                miss_anchor = sample_group_masks(N, meta, args.p_group, seed=sub_seed)
            else:  # attrjoint
                miss = sample_attrjoint_masks(N, meta, args.p_feat, args.p_scal, args.p_offs, seed=sub_seed)
                miss_anchor = miss.any(axis=1)  # 仅在需要 attrjoint 时才用 union

            miss_idx_all = torch.from_numpy(np.where(miss_anchor)[0]).to('cuda', dtype=torch.long)
            avail_idx_all = torch.from_numpy(np.where(~miss_anchor)[0]).to('cuda', dtype=torch.long)
            if miss_idx_all.numel() == 0 or avail_idx_all.numel() == 0:
                continue  # 这个 seed 下无有效样本，跳过

            # 采样一个 batch
            if miss_idx_all.numel() > args.bs_anchor:
                perm = torch.randperm(miss_idx_all.numel(), device='cuda')[:args.bs_anchor]
                miss_idx = miss_idx_all[perm]
            else:
                miss_idx = miss_idx_all
            B = miss_idx.numel()

            # KNN（在 avail 内找邻居）
            pos_miss = anchor[miss_idx]
            pos_av = anchor[avail_idx_all]
            topd, topi = space_topK(pos_miss, pos_av, K)
            nn_idx = avail_idx_all[topi]  # [B,K]
            K_eff = nn_idx.shape[1]

            # 组装 Full token 序列
            input_dim = gaussians._ec_input_dim
            seq = torch.zeros((B, K_eff + 1, input_dim), device='cuda', dtype=torch.float32)

            # Query token
            pos_q = anchor[miss_idx]  # [B,3]
            fh_q = fh_all[miss_idx]  # [B,Fh]
            mu_f_q = mu_feat_all[miss_idx]  # [B,50]
            mu_s_q = mu_scal_all[miss_idx]  # [B,6]
            mu_o_q = mu_off_all[miss_idx]  # [B,3*n_off]

            off = 0
            seq[:, 0, off:off + 3] = 0.0;
            off += 3  # Δpos
            seq[:, 0, off:off + 1] = 0.0;
            off += 1  # dist
            seq[:, 0, off:off + 3] = pos_q;
            off += 3  # abs_pos
            fh_dim = fh_q.shape[1]
            seq[:, 0, off:off + fh_dim] = fh_q;
            off += fh_dim
            seq[:, 0, off:off + gaussians.feat_dim] = mu_f_q;
            off += gaussians.feat_dim
            seq[:, 0, off:off + 6] = mu_s_q;
            off += 6
            seq[:, 0, off:off + 3 * gaussians.n_offsets] = mu_o_q;
            off += 3 * gaussians.n_offsets
            off += gaussians.feat_dim + 6 + 3 * gaussians.n_offsets  # 残差（query=0）
            off += gaussians.n_offsets  # masks（query=0）

            # Neighbor tokens
            pos_nei = anchor[nn_idx.view(-1)].view(B, K_eff, 3)
            dpos = pos_nei - pos_q.unsqueeze(1)
            dist = topd / (torch.clamp(topd.mean(dim=1, keepdim=True), min=1e-6))

            fh_nei = fh_all[nn_idx.view(-1)].view(B, K_eff, fh_dim)
            mu_f_nei = mu_feat_all[nn_idx.view(-1)].view(B, K_eff, gaussians.feat_dim)
            mu_s_nei = mu_scal_all[nn_idx.view(-1)].view(B, K_eff, 6)
            mu_o_nei = mu_off_all[nn_idx.view(-1)].view(B, K_eff, 3 * gaussians.n_offsets)

            # f_nei = feat_gt[nn_idx.view(-1)].view(B, K_eff, gaussians.feat_dim)
            # s_nei = scal_gt[nn_idx.view(-1)].view(B, K_eff, 6)
            # o_nei = off_gt[nn_idx.view(-1)].view(B, K_eff, gaussians.n_offsets, 3).reshape(B, K_eff, -1)
            m_nei = mask_gt[nn_idx.view(-1)].view(B, K_eff, gaussians.n_offsets, 1).squeeze(-1)

            # 加 sigma 标准化（假设 scale_feat_all/scale_scal_all/scale_offs_all 已预计算，或用 abs(res).mean() 近似）
            sigma_f = torch.clamp(res_f.abs().mean(dim=[0, 1], keepdim=True) + 1e-6, min=1e-6)  # [1,1,feat_dim]
            sigma_s = torch.clamp(res_s.abs().mean(dim=[0, 1], keepdim=True) + 1e-6, min=1e-6)
            sigma_o = torch.clamp(res_o.abs().mean(dim=[0, 1], keepdim=True) + 1e-6, min=1e-6)
            res_f = res_f / sigma_f
            res_s = res_s / sigma_s
            res_o = res_o / sigma_o
            # 然后填 seq

            for k in range(K_eff):
                off = 0
                seq[:, 1 + k, off:off + 3] = dpos[:, k, :];
                off += 3
                seq[:, 1 + k, off:off + 1] = dist[:, k:k + 1];
                off += 1
                seq[:, 1 + k, off:off + 3] = pos_nei[:, k, :];
                off += 3
                seq[:, 1 + k, off:off + fh_dim] = fh_nei[:, k, :];
                off += fh_dim
                seq[:, 1 + k, off:off + gaussians.feat_dim] = mu_f_nei[:, k, :];
                off += gaussians.feat_dim
                seq[:, 1 + k, off:off + 6] = mu_s_nei[:, k, :];
                off += 6
                seq[:, 1 + k, off:off + 3 * gaussians.n_offsets] = mu_o_nei[:, k, :];
                off += 3 * gaussians.n_offsets
                seq[:, 1 + k, off:off + gaussians.feat_dim] = res_f[:, k, :];
                off += gaussians.feat_dim
                seq[:, 1 + k, off:off + 6] = res_s[:, k, :];
                off += 6
                seq[:, 1 + k, off:off + 3 * gaussians.n_offsets] = res_o[:, k, :];
                off += 3 * gaussians.n_offsets
                seq[:, 1 + k, off:off + gaussians.n_offsets] = m_nei[:, k, :];
                off += gaussians.n_offsets

            # 前向与损失（目标为残差）
            r_feat, r_scal, r_offs, m_prob = gaussians.ec_transformer(seq)
            # 目标残差
            t_feat = feat_gt[miss_idx] - mu_feat_all[miss_idx]
            t_scal = scal_gt[miss_idx] - mu_scal_all[miss_idx]
            t_offs = off_gt[miss_idx] - mu_off_all[miss_idx].view(B, gaussians.n_offsets, 3)
            t_masks = mask_gt[miss_idx].squeeze(-1)

            # L1 + BCE
            l_feat = F.smooth_l1_loss(r_feat, t_feat, beta=0.05)
            l_scal = F.smooth_l1_loss(r_scal, t_scal, beta=0.05)
            l_offs = F.smooth_l1_loss(r_offs, t_offs, beta=0.05)
            l_mask = F.binary_cross_entropy(m_prob, t_masks)
            loss = 1.5 * l_feat + 0.5 * l_scal + 2.0 * l_offs + 0.5 * l_mask

            # 累积梯度：每个 seed 的 loss / num_seeds
            (loss / num_seeds).backward()
            update_count += 1

            # 累积打印用
            accum_loss += loss.item() / num_seeds
            accum_l_feat += l_feat.item() / num_seeds
            accum_l_scal += l_scal.item() / num_seeds
            accum_l_offs += l_offs.item() / num_seeds
            accum_l_mask += l_mask.item() / num_seeds

        # 一步 update（只有在本迭代内至少有一个有效 seed 时才更新）
        if update_count > 0:
            torch.nn.utils.clip_grad_norm_(gaussians.ec_transformer.parameters(), max_norm=1.0)
            optim.step()
        scheduler.step()

        if it % 50 == 0:
            logger.info(
                f"[EC-T] it={it:05d} | loss={accum_loss:.4f} | lF={accum_l_feat:.3f} lS={accum_l_scal:.3f} lO={accum_l_offs:.3f} lM={accum_l_mask:.3f} | lr={optim.param_groups[0]['lr']:.4e}")

    # 保存
    out_ck = os.path.join(model_path, "mlp_ckpt_ec_finetuned.pth")
    gaussians.save_mlp_checkpoints(out_ck)
    logger.info(f"[EC-T] done. saved {out_ck} | time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
