# utils/pyg_inpaint.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅在可用时启用 PyG
try:
    from torch_geometric.nn import EdgeConv, GATv2Conv
    from torch_cluster import knn as _knn  # knn(x_src, x_dst, k) -> (row(dst), col(src))
    PYG_OK = True
    PYG_ERR = None
except Exception as e:
    PYG_OK = False
    PYG_ERR = e


def mlp(layers):
    mods = []
    for i in range(len(layers) - 1):
        mods.append(nn.Linear(layers[i], layers[i + 1]))
        if i != len(layers) - 2:
            mods.append(nn.ReLU(inplace=True))
    return nn.Sequential(*mods)


class PYGInpaintor(nn.Module):
    """
    PyTorch Geometric 版本的补洞器（无蒸馏）：
    - 两层可选 EdgeConv/GATv2Conv（bipartite kNN on 3D positions；k1/k2可配）
    - 节点特征：查询=先验均值（feat/scale/off），mask=0；已知=真实值
    - 残差回归 + 置信 gating/3σ剪裁 + 分项阈值回退 +（可选）插值融合
    - 训练：Charbonnier offsets + L1/BCE；无蒸馏
    """
    def __init__(self, k=16, hidden=128, device='cuda', mp_iters=1,
                 conv_type=None, k2=None, heads=None):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError(f"torch_geometric not available: {PYG_ERR}")

            # 参数优先级：入参 > 环境变量 > 默认

        def _env_int(name, default):
            v = os.getenv(name, None)
            return int(v) if v is not None else int(default)

        def _env_str(name, default):
            v = os.getenv(name, None)
            return (v if v is not None else default)

        self.k = int(k)
        # k2
        if k2 is not None:
            self.k2 = int(k2)
        else:
            self.k2 = _env_int("HAC_GNN_K2", self.k)
        # hidden
        self.hid = int(hidden)
        self.device = device
        # mp_iters
        if mp_iters is not None:
            self.mp_iters = int(mp_iters)
        else:
            self.mp_iters = _env_int("HAC_GNN_MP", 1)
        # conv_type
        if conv_type is not None:
            self.conv_type = conv_type.lower()
        else:
            self.conv_type = _env_str("HAC_GNN_CONV", "edge").lower()
        # heads
        if heads is not None:
            self.heads = int(heads)
        else:
            self.heads = _env_int("HAC_GNN_HEADS", 2)
        # GATv2 需要可整除，否则退化为单头
        if self.conv_type == "gat" and (self.hid % self.heads != 0):
            self.heads = 1

        self._built = False

    def _build(self, in_dim, D, K):
        """
        构建图卷积与回归头。in_dim 为节点特征维度，D/K 为场景维度。
        """
        if self.conv_type == "gat":
            # GATv2: 输入 in_dim -> 输出 hid；用 heads 分配每头维度
            out_h_each = max(1, self.hid // self.heads)
            self.conv1 = GATv2Conv(
                in_dim, out_h_each, heads=self.heads, add_self_loops=False,
                edge_dim=None, share_weights=False
            )
            self.conv2 = GATv2Conv(
                self.hid, out_h_each, heads=self.heads, add_self_loops=False,
                edge_dim=None, share_weights=False
            ) if self.mp_iters >= 2 else None
        else:
            # EdgeConv: 输入特征按照 [x_i || x_j - x_i] -> 2*in_dim
            self.conv1 = EdgeConv(mlp([2 * in_dim, self.hid, self.hid]), aggr='mean')
            self.conv2 = EdgeConv(mlp([2 * self.hid, self.hid, self.hid]), aggr='mean') if self.mp_iters >= 2 else None

        # 头部回归
        self.head_feat    = mlp([self.hid, self.hid, D])
        self.head_scale   = mlp([self.hid, self.hid, 6])
        self.head_offsets = mlp([self.hid, self.hid, 3 * K])
        self.head_mask    = mlp([self.hid, self.hid, K + 1])

        self.to(self.device)
        self._built = True

    @torch.no_grad()
    def _knn_bipartite(self, pos_k, pos_q, k):
        """
        构造“已知->查询”的有向边（bipartite kNN）。
        返回 edge_index: [2,E]，节点顺序为 [query(0..M-1), known(M..M+Nk-1)]
        """
        row, col = _knn(pos_k, pos_q, k)  # row: dst(query idx), col: src(known idx)
        M = pos_q.shape[0]
        src = col + M  # shift 到组合图中的已知节点
        dst = row      # 查询节点索引
        edge_index = torch.stack([src, dst], dim=0).contiguous()
        return edge_index

    def _compose_nodes(self, gm, x_q, x_k, feat_k, scale_k, off_k, mask_k):
        """
        统一节点特征：
        - 查询节点: [pos3, fh_H, m_f_D, m_s_6, m_o_3K, mask0_(K+1)]
        - 已知节点: [pos3, fh_H, feat_D, scale6, off_3K, mask_(K+1)]
        """
        D = gm.feat_dim
        K = gm.n_offsets
        H = gm.encoding_xyz.output_dim

        # 查询先验
        fh_q = gm.calc_interp_feat(x_q)               # [M,H]
        prior = gm.get_grid_mlp(fh_q)
        m_f, _, _, m_s, _, m_o, _, _, _, _ = torch.split(prior, [D, D, D, 6, 6, 3*K, 3*K, 1, 1, 1], dim=-1)
        mask_q = torch.zeros((x_q.shape[0], K + 1), device=x_q.device, dtype=torch.float32)

        # 已知真实
        fh_k = gm.calc_interp_feat(x_k)               # [Nk,H]
        # offsets -> [Nk, 3K]
        if off_k.dim() == 3 and off_k.shape[-1] == 3:
            off_k = off_k.view(off_k.shape[0], 3 * K)
        elif off_k.dim() == 2 and off_k.shape[1] == 3 * K:
            pass
        else:
            off_k = off_k.view(off_k.shape[0], -1)
        # mask -> [Nk, K+1] float（squeeze + pad/trim）
        mk = mask_k
        if mk.dtype not in (torch.float16, torch.float32, torch.float64):
            mk = mk.float()
        if mk.dim() == 3 and mk.shape[-1] == 1:
            mk = mk.squeeze(-1)
        elif mk.dim() == 1:
            mk = mk.view(-1, 1)
        if mk.dim() == 2:
            Ck = mk.shape[1]
            if Ck < (K + 1):
                pad = torch.zeros((mk.shape[0], (K + 1) - Ck), device=mk.device, dtype=mk.dtype)
                mk = torch.cat([mk, pad], dim=1)
            elif Ck > (K + 1):
                mk = mk[:, : (K + 1)]
        else:
            mk = torch.zeros((mask_k.shape[0], K + 1), device=mask_k.device, dtype=torch.float32)
        mask_k = mk

        # 拼接（查询在前，已知在后）
        xnode_q = torch.cat([x_q, fh_q, m_f, m_s, m_o, mask_q], dim=-1)  # [M, 3+H+D+6+3K+(K+1)]
        xnode_k = torch.cat([x_k, fh_k, feat_k, scale_k, off_k, mask_k], dim=-1)
        X = torch.cat([xnode_q, xnode_k], dim=0).contiguous()
        in_dim = X.shape[-1]
        return X, in_dim

    def _gating_and_blend(self, gm, x_q, d_feat, d_scale, d_off, d_mask,
                          feat_interp=None, scale_interp=None, off_interp=None, mask_interp=None):
        """
        残差 + 置信 gating + 3σ 剪裁 + 分项阈值回退 + （可选）插值融合
        """
        eps = 1e-6
        D = gm.feat_dim
        K = gm.n_offsets

        fh_q = gm.calc_interp_feat(x_q)
        prior = gm.get_grid_mlp(fh_q)
        m_f, s_f, _, m_s, s_s, m_o, s_o, _, _, _ = torch.split(prior, [D, D, D, 6, 6, 3*K, 3*K, 1, 1, 1], dim=-1)
        s_f = torch.clamp(s_f, min=eps)
        s_s = torch.clamp(s_s, min=eps)
        s_o = torch.clamp(s_o, min=eps)

        # 3σ 剪裁
        tau = float(os.getenv("HAC_GNN_TAU", "3.0"))
        d_feat = torch.clamp(d_feat, -tau * s_f, +tau * s_f)
        d_scale = torch.clamp(d_scale, -tau * s_s, +tau * s_s)
        d_off = torch.clamp(d_off, -tau * s_o, +tau * s_o)

        # 置信
        alpha = float(os.getenv("HAC_GNN_ALPHA", "0.5"))
        zf = (d_feat.abs() / s_f).mean(dim=-1)
        zs = (d_scale.abs() / s_s).mean(dim=-1)
        zo = (d_off.abs() / s_o).mean(dim=-1)
        cf = torch.exp(-alpha * zf).clamp(0, 1).unsqueeze(-1)
        cs = torch.exp(-alpha * zs).clamp(0, 1).unsqueeze(-1)
        co = torch.exp(-alpha * zo).clamp(0, 1).unsqueeze(-1)

        # 先验 + 残差
        feat_hat = m_f + cf * d_feat
        scale_hat = m_s + cs * d_scale
        off_hat = (m_o + co * d_off).view(-1, K, 3)
        mask_hat = torch.sigmoid(d_mask).view(-1, K + 1, 1)

        # 融合/回退（可选）
        use_blend = (feat_interp is not None)
        if use_blend:
            # 分项阈值
            T_off = float(os.getenv("HAC_GNN_CONF_TH_OFF", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            T_fea = float(os.getenv("HAC_GNN_CONF_TH_FEA", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            T_sca = float(os.getenv("HAC_GNN_CONF_TH_SCA", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            lo = (co.squeeze(-1) < T_off)
            lf = (cf.squeeze(-1) < T_fea)
            ls = (cs.squeeze(-1) < T_sca)

            # 正确广播形状
            cf2 = cf.view(-1, 1)   # [M,1] -> [M,D]
            cs2 = cs.view(-1, 1)   # [M,1] -> [M,6]
            co2 = co.view(-1, 1)   # [M,1] -> [M,C]
            co3 = co.view(-1, 1, 1)  # [M,1,1] -> [M,K,3]

            # 软融合
            feat_bl = cf2 * feat_hat + (1.0 - cf2) * feat_interp  # [M,D]
            scale_bl = cs2 * scale_hat + (1.0 - cs2) * scale_interp  # [M,6]
            off_bl = co3 * off_hat + (1.0 - co3) * off_interp  # [M,K,3]

            # mask 融合（公共前 C 个通道）
            C = mask_hat.shape[1]
            if mask_interp is not None:
                C = min(C, mask_interp.shape[1])
            mask_bl = mask_hat
            if C > 0:
                mask_bl = mask_hat.clone()
                mask_bl[:, :C, 0] = co2 * mask_hat[:, :C, 0] + (1.0 - co2) * mask_interp[:, :C]  # [M,C]

            # 硬回退（低置信）
            if lo.any():
                idx = lo.nonzero(as_tuple=False).view(-1)
                off_hat[idx] = off_interp[idx]
                if C > 0:
                    mask_hat[idx, :C, 0] = mask_interp[idx, :C]
            if lf.any():
                idx = lf.nonzero(as_tuple=False).view(-1)
                feat_hat[idx] = feat_interp[idx]
            if ls.any():
                idx = ls.nonzero(as_tuple=False).view(-1)
                scale_hat[idx] = scale_interp[idx]

            # 高置信 -> 软融合
            hi = (~lo).nonzero(as_tuple=False).view(-1)
            if hi.numel() > 0:
                off_hat[hi] = off_bl[hi]
                if C > 0:
                    mask_hat[hi, :C, 0] = mask_bl[hi, :C, 0]
            hf = (~lf).nonzero(as_tuple=False).view(-1)
            if hf.numel() > 0:
                feat_hat[hf] = feat_bl[hf]
            hs = (~ls).nonzero(as_tuple=False).view(-1)
            if hs.numel() > 0:
                scale_hat[hs] = scale_bl[hs]

        return feat_hat, scale_hat, off_hat, mask_hat

    def forward_once(self, gm, x_query, x_known,
                     feat_known, scale_known, offsets_known, mask_known,
                     apply_conf: bool = False):
        """
        构图：查询在前，已知在后；边从已知->查询（bipartite kNN）。
        """
        device = self.device
        x_query = x_query.to(device)
        x_known = x_known.to(device)
        feat_known = feat_known.to(device)
        scale_known = scale_known.to(device)
        offsets_known = offsets_known.to(device)
        mask_known = mask_known.to(device)

        D = gm.feat_dim
        K = gm.n_offsets

        X, in_dim = self._compose_nodes(gm, x_query, x_known, feat_known, scale_known, offsets_known, mask_known)
        if not self._built:
            self._build(in_dim, D, K)

        # 第一层边（k1）
        edge_index1 = self._knn_bipartite(x_known, x_query, self.k)
        h = self.conv1(X, edge_index1)
        # 第二层边（k2，可不同于 k1）
        if self.conv2 is not None:
            edge_index2 = self._knn_bipartite(x_known, x_query, self.k2)
            h = self.conv2(h, edge_index2)
            edge_interp = edge_index2
        else:
            edge_interp = edge_index1

        h_q = h[:x_query.shape[0]]

        # 回归头
        d_feat = self.head_feat(h_q)
        d_scale = self.head_scale(h_q)
        d_off = self.head_offsets(h_q)
        d_mask = self.head_mask(h_q)

        if not apply_conf:
            # 训练路径：直接返回 gated（无融合）
            feat_hat, scale_hat, off_hat, mask_hat = self._gating_and_blend(
                gm, x_query, d_feat, d_scale, d_off, d_mask, None, None, None, None
            )
            return feat_hat, scale_hat, off_hat, mask_hat

        # 推理路径：准备插值基线（按 edge_interp 计算距离加权插值）
        M = x_query.shape[0]
        src, dst = edge_interp[0], edge_interp[1]  # src: known+M, dst: query
        pos_all = torch.cat([x_query, x_known], dim=0)
        dq = pos_all[dst] - pos_all[src]
        r = torch.norm(dq, dim=-1) + 1e-6
        w = 1.0 / r
        w_sum = torch.zeros(M, device=device).index_add_(0, dst, w) + 1e-8
        w_norm = w / w_sum[dst]

        # 已知属性（真实值）
        feat_k = feat_known
        scale_k = scale_known
        # offsets -> [Nk, 3K]
        off_k = offsets_known
        if off_k.dim() == 3 and off_k.shape[-1] == 3:
            off_k = off_k.view(off_k.shape[0], 3 * K)
        elif off_k.dim() == 2 and off_k.shape[1] == 3 * K:
            pass
        else:
            off_k = off_k.view(off_k.shape[0], -1)
        # mask -> [Nk, K+1] float
        mk = mask_known
        if mk.dtype not in (torch.float16, torch.float32, torch.float64):
            mk = mk.float()
        if mk.dim() == 3 and mk.shape[-1] == 1:
            mk = mk.squeeze(-1)
        elif mk.dim() == 1:
            mk = mk.view(-1, 1)
        if mk.dim() == 2:
            Ck = mk.shape[1]
            if Ck < (K + 1):
                pad = torch.zeros((mk.shape[0], (K + 1) - Ck), device=mk.device, dtype=mk.dtype)
                mk = torch.cat([mk, pad], dim=1)
            elif Ck > (K + 1):
                mk = mk[:, : (K + 1)]
        else:
            mk = torch.zeros((mask_known.shape[0], K + 1), device=mask_known.device, dtype=torch.float32)
        mask_k = mk

        # 将 src 映射回已知子图索引
        src_k = src - M
        feat_interp = torch.zeros((M, D), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * feat_k[src_k])
        scale_interp = torch.zeros((M, 6), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * scale_k[src_k])
        off_interp = torch.zeros((M, 3 * K), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * off_k[src_k]).view(M, K, 3)
        C = K + 1
        mask_interp = torch.zeros((M, C), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * mask_k[src_k])
        # 兜底：若设置 HAC_GNN_SAFE=1，直接使用插值基线，避免坏权重拉崩
        if os.getenv("HAC_GNN_SAFE", "0") == "1":
            # 把 mask_interp 变为 [M,K+1,1]
            mask_safe = torch.zeros((M, K+1, 1), device=device)
            C_safe = min(mask_interp.shape[1], K+1)
            if C_safe > 0:
                mask_safe[:, :C_safe, 0] = mask_interp[:, :C_safe]
            return feat_interp, scale_interp, off_interp, mask_safe

        return self._gating_and_blend(gm, x_query, d_feat, d_scale, d_off, d_mask,
                                      feat_interp, scale_interp, off_interp, mask_interp)

    def predict(self, gm, x_all, known_mask, feat, scale, offsets, mask):
        """
        推理：对缺失锚点批量预测。限制库规模与查询分块避免 OOM。
        """
        device = self.device
        x_all = x_all.to(device)
        known_mask = known_mask.to(device)

        # 限制推理库规模（防 OOM）
        pred_max_known = int(os.getenv("HAC_GNN_PRED_MAX_KNOWN", "0"))
        if pred_max_known > 0 and known_mask.sum().item() > pred_max_known:
            keep = torch.where(known_mask)[0]
            perm = torch.randperm(keep.numel(), device=device)[:pred_max_known]
            sel = keep[perm]
            known_mask[:] = False
            known_mask[sel] = True

        x_k = x_all[known_mask]
        x_q = x_all[~known_mask]
        feat_k = feat[known_mask]
        scale_k = scale[known_mask]
        off_k = offsets[known_mask]
        mask_k = mask[known_mask]

        preds = []
        B = int(os.getenv("HAC_GNN_PRED_BQ", "2048"))
        for s in range(0, x_q.shape[0], B):
            xq = x_q[s:s + B]
            out = self.forward_once(gm, xq, x_k, feat_k, scale_k, off_k, mask_k, apply_conf=True)
            preds.append(out)

        if not preds:
            return (torch.zeros((0, gm.feat_dim), device=device),
                    torch.zeros((0, 6), device=device),
                    torch.zeros((0, gm.n_offsets, 3), device=device),
                    torch.zeros((0, gm.n_offsets + 1, 1), device=device))
        feat_hat = torch.cat([p[0] for p in preds], dim=0)
        scale_hat = torch.cat([p[1] for p in preds], dim=0)
        off_hat = torch.cat([p[2] for p in preds], dim=0)
        mask_hat = torch.cat([p[3] for p in preds], dim=0)
        return feat_hat, scale_hat, off_hat, mask_hat

    def fit(self, gm, x_all, known_mask, feat, scale, offsets, mask,
            steps=100, lr=1e-3, k=None, max_known=20000, batch_q=1024,
            log_every=50, val_every=200, val_ratio=0.1, clip_grad=1.0, cosine_decay=True,
            group_ids: torch.Tensor = None, group_rate: float = 0.2):
        """
        场景内自监督训练（组级伪丢包）：
        - L1/BCE + Charbonnier offsets（无蒸馏）
        """
        device = self.device
        self.to(device)
        self.train()
        if k is not None:
            self.k = k

        idx_known_all = torch.where(known_mask.to('cpu'))[0]
        if idx_known_all.numel() < (self.k * 2 + 8):
            return
        if idx_known_all.numel() > max_known:
            perm = torch.randperm(idx_known_all.numel())[:max_known]
            idx_known_all = idx_known_all[perm]

        x_k = x_all[idx_known_all].to(device)
        feat_k = feat[idx_known_all].to(device)
        scale_k = scale[idx_known_all].to(device)
        off_k = offsets[idx_known_all].to(device)
        msk_k = mask[idx_known_all].to(device)

        # 组级采样
        if (group_ids is not None) and (group_ids.numel() == known_mask.numel()):
            gid_sub = group_ids[idx_known_all]
            uniq = torch.unique(gid_sub)
            n_grp = uniq.numel()
            gq = max(1, int(n_grp * float(group_rate)))
            permg = torch.randperm(n_grp)[:gq]
            q_groups = uniq[permg]
            q_mask_sub = torch.isin(gid_sub, q_groups)
            pool_mask_sub = ~q_mask_sub
            q_idx_sub = torch.where(q_mask_sub)[0].to(device)
            pool_idx_sub = torch.where(pool_mask_sub)[0].to(device)
            if pool_idx_sub.numel() < (self.k + 8):
                n = x_k.shape[0]
                n_query = min(n // 4, 8000)
                perm = torch.randperm(n, device=device)
                q_idx_sub = perm[:n_query]
                pool_idx_sub = perm[n_query:]
        else:
            n = x_k.shape[0]
            n_query = min(n // 4, 8000)
            perm = torch.randperm(n, device=device)
            q_idx_sub = perm[:n_query]
            pool_idx_sub = perm[n_query:]

        x_q = x_k[q_idx_sub]
        x_pool = x_k[pool_idx_sub]
        feat_pool = feat_k[pool_idx_sub]
        scale_pool = scale_k[pool_idx_sub]
        off_pool = off_k[pool_idx_sub]
        msk_pool = msk_k[pool_idx_sub]

        # 懒构建触发
        if not self._built:
            with torch.no_grad():
                b0 = min(batch_q, x_q.shape[0])
                if b0 > 0:
                    _ = self.forward_once(gm, x_q[:b0], x_pool, feat_pool, scale_pool, off_pool, msk_pool, apply_conf=False)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps)) if cosine_decay else None

        # 验证子集
        has_val = val_ratio > 0 and x_q.shape[0] >= 32
        if has_val:
            n_val = max(8, int(x_q.shape[0] * val_ratio))
            v_idx = torch.randperm(x_q.shape[0], device=q_idx_sub.device)[:n_val]
            x_v = x_q[v_idx]
            gt_v_idx = q_idx_sub[v_idx]

        # 训练
        try:
            from tqdm import trange
            iters = trange(steps, desc="PYG fit", dynamic_ncols=True)
        except Exception:
            iters = range(steps)

        for it in iters:
            start = (it * batch_q) % max(1, x_q.shape[0])
            end = min(x_q.shape[0], start + batch_q)
            if end - start < 1:
                if hasattr(iters, "set_postfix"):
                    iters.set_postfix(skip="1")
                continue

            xq_b = x_q[start:end]
            feat_hat, scale_hat, off_hat, mask_hat = self.forward_once(
                gm, xq_b, x_pool, feat_pool, scale_pool, off_pool, msk_pool, apply_conf=False
            )
            gt_idx = q_idx_sub[start:end]
            feat_gt = feat_k[gt_idx]
            scale_gt = scale_k[gt_idx]
            off_gt = off_k[gt_idx]
            msk_gt = msk_k[gt_idx]

            # 基本损失（Charbonnier offsets）
            K = gm.n_offsets
            l_feat = F.l1_loss(feat_hat, feat_gt)
            l_scale = F.l1_loss(scale_hat, scale_gt)
            eps_c = 1e-3
            diff = off_hat - off_gt
            l_off_all = torch.sqrt(diff * diff + eps_c * eps_c)
            w_mask = torch.clamp(msk_gt[:, :K, :], 0, 1)
            l_off = (l_off_all * (1.0 + 2.0 * w_mask)).mean()
            l_mask = F.binary_cross_entropy(mask_hat, torch.clamp(msk_gt, 0, 1))

            loss = l_feat + 0.5 * l_scale + l_off + 0.5 * l_mask

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            opt.step()
            if sch is not None:
                sch.step()

            if hasattr(iters, "set_postfix"):
                iters.set_postfix(
                    feat=f"{l_feat.item():.4f}",
                    scale=f"{l_scale.item():.4f}",
                    off=f"{l_off.item():.4f}",
                    mask=f"{l_mask.item():.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )

            if (it + 1) % max(1, log_every) == 0 or (it + 1) == steps:
                if os.getenv("HAC_GNN_FIT_LOG", "1") == "1":
                    print(
                        f"[fit] it={it + 1}/{steps} | "
                        f"l_feat={l_feat.item():.4f} l_scale={l_scale.item():.4f} "
                        f"l_off={l_off.item():.4f} l_mask={l_mask.item():.4f} "
                        f"lr={opt.param_groups[0]['lr']:.2e}"
                    )

            if has_val and ((it + 1) % max(1, val_every) == 0 or (it + 1) == steps):
                self.eval()
                with torch.no_grad():
                    feat_v, scale_v, off_v, mask_v = self.forward_once(
                        gm, x_v, x_pool, feat_pool, scale_pool, off_pool, msk_pool, apply_conf=False
                    )
                    feat_gt_v = feat_k[gt_v_idx]
                    scale_gt_v = scale_k[gt_v_idx]
                    off_gt_v = off_k[gt_v_idx]
                    msk_gt_v = msk_k[gt_v_idx]
                    l1f = F.l1_loss(feat_v, feat_gt_v).item()
                    l1s = F.l1_loss(scale_v, scale_gt_v).item()
                    l1o = F.l1_loss(off_v, off_gt_v).item()
                    bcem = F.binary_cross_entropy(mask_v, torch.clamp(msk_gt_v, 0, 1)).item()
                    print(
                        f"[val] it={it + 1} | L1(feat)={l1f:.4f} L1(scale)={l1s:.4f} "
                        f"L1(off)={l1o:.4f} BCE(mask)={bcem:.4f}"
                    )
                self.train()
