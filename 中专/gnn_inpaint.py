# utils/gnn_inpaint.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(layers):
    mods = []
    for i in range(len(layers) - 1):
        mods.append(nn.Linear(layers[i], layers[i + 1]))
        if i != len(layers) - 2:
            mods.append(nn.ReLU(inplace=True))
    return nn.Sequential(*mods)


class GNNInpaintor(nn.Module):
    """
    轻量“消息传递+注意力”的补洞原型（不依赖 torch-geometric）
    - 懒构建：按实际拼接后的输入维度在运行时构建 MLP，避免手算维度出错
    - 设备一致：构建后立即 self.to(self.device)，避免 CPU/CUDA 不一致
    输入：
      - 查询点位置 x_query、hash 编码 fh_i、mlp_grid 先验(mean_feat/mean_scale/mean_offsets)
      - 邻居点位置/属性/编码
    输出：
      - 丢失锚点的 feat/scale/offsets/mask（预测残差 + 先验）
    """
    def __init__(self, k=16, hidden=128, device='cuda',k_far=None, mp_iters=None):
        super().__init__()
        self.k = k
        if k_far is None:
            env_v = os.getenv("HAC_GNN_KFAR", "")
            if env_v.strip() != "":
                try:
                    k_far = int(env_v)
                except Exception:
                    k_far = None
        self.k_far = (k // 2) if (k_far is None) else int(k_far)  # 远邻个数；默认 k//2
        self.hid = hidden
        self.device = device
        self.mp_iters = int(os.getenv("HAC_GNN_MP", "2")) if mp_iters is None else int(mp_iters)

        # 懒构建相关缓存
        self._built = False
        self.D_cur = None
        self.K_cur = None
        self.H_cur = None
        self.in_msg_dim = None
        self.in_att_dim = None
        self.in_q_dim = None
        self._dbg_printed = False  # 仅首次打印维度
        self.in_att2_dim = None

    def _build_from_dims(self, feat_dim, n_offsets, hash_dim,
                         in_msg_dim, in_att_dim, in_q_dim):
        self.D_cur = feat_dim
        self.K_cur = n_offsets
        self.H_cur = hash_dim
        self.in_msg_dim = in_msg_dim
        self.in_att_dim = in_att_dim
        self.in_q_dim = in_q_dim

        # 第一轮
        self.msg_mlp = mlp([in_msg_dim, self.hid, self.hid])
        self.att_mlp = mlp([in_att_dim, self.hid, 1])

        # 第二轮：注意力输入维度 = hid + H + 4（查询从 fh_q 换成 q_embed）
        in_att2_dim = self.hid + hash_dim + 4
        self.in_att2_dim = in_att2_dim
        self.msg_mlp2 = mlp([in_msg_dim, self.hid, self.hid])
        self.att_mlp2 = mlp([in_att2_dim, self.hid, 1])

        # q 投影：joint(hid+q_dim) -> hid
        self.q_proj = mlp([self.hid + in_q_dim, self.hid])

        head_in = self.hid + in_q_dim
        self.head_feat = mlp([head_in, self.hid, feat_dim])
        self.head_scale = mlp([head_in, self.hid, 6])
        self.head_offsets = mlp([head_in, self.hid, 3 * n_offsets])
        self.head_mask = mlp([head_in, self.hid, n_offsets + 1])

        self.to(self.device)
        self._built = True

    @torch.no_grad()
    def _knn(self, x_query, x_pool, k):
        """
        自适应内存kNN：
          - 查询分块 B：由 HAC_GNN_KNN_CAP 限制每块距离条目数（默认 5e7）
          - 库侧平铺：HAC_GNN_POOL_TILE 控制每个库tile大小（默认 20000）
          - 可选半精度距离：HAC_GNN_CDIST_HALF=1 时用 FP16 计算 cdist 再转回 FP32
        """
        import os
        cap = int(float(os.getenv("HAC_GNN_KNN_CAP", "5e7")))  # 每块最大距离条目数
        tile = int(os.getenv("HAC_GNN_POOL_TILE", "20000"))  # 库侧tile大小
        use_half = int(os.getenv("HAC_GNN_CDIST_HALF", "1")) > 0 and x_query.is_cuda

        Np = x_pool.shape[0]
        # 估计查询块 B，使得 B*Np <= cap
        B = max(256, min(4096, cap // max(1, Np)))

        all_idx, all_dist = [], []
        for s in range(0, x_query.shape[0], B):
            qe = x_query[s:s + B]  # [b,3]
            b = qe.shape[0]
            # 当前最佳 top-k 初始化
            best_val = torch.full((b, k), float('inf'), device=qe.device, dtype=torch.float32)
            best_idx = torch.full((b, k), -1, device=qe.device, dtype=torch.long)

            # 沿库侧平铺，流式合并 top-k
            for t in range(0, Np, tile):
                pool = x_pool[t:t + tile]  # [tN,3]
                # 距离计算（可选半精度）
                if use_half:
                    dist = torch.cdist(qe.half(), pool.half()).float()  # [b,tN]
                else:
                    dist = torch.cdist(qe, pool)  # [b,tN]
                val, idx = torch.topk(dist, k=min(k, pool.shape[0]), dim=-1, largest=False)  # [b,k]
                idx = idx + t

                # 合并当前 best 与本tile候选
                val_cat = torch.cat([best_val, val], dim=-1)  # [b, k + k_t]
                idx_cat = torch.cat([best_idx, idx], dim=-1)  # [b, k + k_t]
                val2, top_idx2 = torch.topk(val_cat, k=k, dim=-1, largest=False)
                rows = torch.arange(b, device=qe.device).unsqueeze(-1)
                idx2 = idx_cat.gather(-1, top_idx2)

                best_val, best_idx = val2, idx2
                # 及时释放中间张量，缓解显存峰值
                del dist, val, idx, val_cat, idx_cat, val2, top_idx2

            all_idx.append(best_idx)
            all_dist.append(best_val)
            # 流式阶段释放缓存
            if x_query.is_cuda:
                torch.cuda.empty_cache()

        return torch.cat(all_idx, dim=0), torch.cat(all_dist, dim=0)

    def _compute_prior(self, gm, x):
        """
        计算查询点处的 hash 编码与 mlp_grid 先验
        """
        fh = gm.calc_interp_feat(x)  # [M,H]
        D = gm.feat_dim
        K = gm.n_offsets
        out = gm.get_grid_mlp(fh)
        splits = [D, D, D, 6, 6, 3 * K, 3 * K, 1, 1, 1]
        mean, scale, prob, mean_s, scale_s, mean_off, scale_off, Qf, Qs, Qo = torch.split(out, splits, dim=-1)
        prior = {
            'fh': fh,
            'mean_feat': mean, 'scale_feat': scale, 'prob_feat': prob,
            'mean_scale': mean_s, 'scale_scale': scale_s,
            'mean_offsets': mean_off, 'scale_offsets': scale_off,
            'Q_feat': Qf, 'Q_scale': Qs, 'Q_offsets': Qo,
        }
        return prior

    def forward_once(self, gm, x_query, x_known,
                     feat_known, scale_known, offsets_known, mask_known,
                     apply_conf: bool = False):
        device = self.device
        x_query = x_query.to(device);
        x_known = x_known.to(device)
        feat_known = feat_known.to(device);
        scale_known = scale_known.to(device)
        offsets_known = offsets_known.to(device);
        mask_known = mask_known.to(device)

        K = gm.n_offsets;
        D = gm.feat_dim;
        H = gm.encoding_xyz.output_dim

        # 先验
        prior_q = self._compute_prior(gm, x_query)
        fh_q = prior_q['fh'];
        mean_feat_q = prior_q['mean_feat']
        mean_scale_q = prior_q['mean_scale'];
        mean_offsets_q = prior_q['mean_offsets']

        # 多尺度邻域（近邻 k + 远邻 k_far）
        k_near = int(self.k);
        k_far = int(max(0, self.k_far))
        k_total = k_near + k_far if k_far > 0 else k_near

        nbr_idx_all, dists_all = self._knn(x_query, x_known, k_total)  # [M,k_total]
        M, kt = nbr_idx_all.shape
        if kt < k_total:
            # 邻域不足时，退化为单尺度
            k_near = kt;
            k_far = 0

        idx_near = nbr_idx_all[:, :k_near]
        dst_near = dists_all[:, :k_near]
        if k_far > 0:
            idx_far = nbr_idx_all[:, k_near:]
            dst_far = dists_all[:, k_near:]
        # 取邻居特征
        fh_all = gm.calc_interp_feat(x_known)  # [N,H]
        fh_near = fh_all[idx_near]  # [M,k,H]
        feat_near = feat_known[idx_near]  # [M,k,D]
        scale_near = scale_known[idx_near]  # [M,k,6]
        off_near = offsets_known[idx_near].view(M, k_near, 3 * K)  # [M,k,3K]
        mask_near = mask_known[idx_near].squeeze(-1)  # [M,k,Cm]

        if k_far > 0:
            fh_far = fh_all[idx_far]
            feat_far = feat_known[idx_far]
            scale_far = scale_known[idx_far]
            off_far = offsets_known[idx_far].view(M, k_far, 3 * K)
            mask_far = mask_known[idx_far].squeeze(-1)

        # 相对位移与距离
        x_nbr_near = x_known[idx_near];
        dxyz_near = x_nbr_near - x_query.unsqueeze(1);
        r_near = torch.norm(dxyz_near, dim=-1, keepdim=True)
        if k_far > 0:
            x_nbr_far = x_known[idx_far];
            dxyz_far = x_nbr_far - x_query.unsqueeze(1);
            r_far = torch.norm(dxyz_far, dim=-1, keepdim=True)

        # 输入拼接
        msg_in_near = torch.cat([fh_near, feat_near, scale_near, off_near, mask_near, dxyz_near, r_near],
                                dim=-1)  # [M,k,*]
        att_in_near = torch.cat([fh_q.unsqueeze(1).expand(-1, k_near, -1), fh_near, dxyz_near, r_near],
                                dim=-1)  # [M,k,*]
        if k_far > 0:
            msg_in_far = torch.cat([fh_far, feat_far, scale_far, off_far, mask_far, dxyz_far, r_far], dim=-1)
            att_in_far = torch.cat([fh_q.unsqueeze(1).expand(-1, k_far, -1), fh_far, dxyz_far, r_far], dim=-1)

        # 懒构建（维度一致即可）
        need_build = (not self._built)
        need_build = need_build or (getattr(self, "in_msg_dim", None) != msg_in_near.size(-1))
        need_build = need_build or (getattr(self, "in_att_dim", None) != att_in_near.size(-1))
        need_build = need_build or (getattr(self, "in_q_dim", None) != (H + D + 6 + 3 * K))
        # 关键：第二轮注意力输入维度 = hid + H + 4
        need_build = need_build or (getattr(self, "in_att2_dim", None) != (self.hid + H + 4))
        need_build = need_build or (getattr(self, "K_cur", None) != K)
        need_build = need_build or (getattr(self, "D_cur", None) != D)
        if need_build:
            self._build_from_dims(
                feat_dim=D,
                n_offsets=K,
                hash_dim=H,
                in_msg_dim=msg_in_near.size(-1),
                in_att_dim=att_in_near.size(-1),
                in_q_dim=(H + D + 6 + 3 * K),
            )

        # 双路消息 + 注意力聚合（近邻 + 远邻），聚合后相加（保持 hid 维度不变）
        msg_near = self.msg_mlp(msg_in_near)  # [M,k,hid]
        att_near = self.att_mlp(att_in_near).squeeze(-1)  # [M,k]
        att_near = torch.softmax(att_near, dim=1)
        agg_near = torch.sum(att_near.unsqueeze(-1) * msg_near, dim=1)  # [M,hid]

        if k_far > 0:
            msg_far = self.msg_mlp(msg_in_far)
            att_far = self.att_mlp(att_in_far).squeeze(-1)
            att_far = torch.softmax(att_far, dim=1)
            agg_far = torch.sum(att_far.unsqueeze(-1) * msg_far, dim=1)  # [M,hid]
            agg = agg_near + agg_far
        else:
            agg = agg_near

        # 残差头
        q_in = torch.cat([fh_q, mean_feat_q, mean_scale_q, mean_offsets_q], dim=-1)
        # 第一轮聚合结果 agg 已有；q_in = [fh_q, mean_feat_q, mean_scale_q, mean_offsets_q]
        joint1 = torch.cat([agg, q_in], dim=-1)  # [M, hid + q_dim]

        # 第二轮（可开关）：用 joint1 投影成 q_embed，再按相同邻域重新做一轮消息+注意力
        if getattr(self, "mp_iters", 2) >= 2:
            q_embed = self.q_proj(joint1)  # [M,hid]
            # 近邻
            att2_in_near = torch.cat([q_embed.unsqueeze(1).expand(-1, k_near, -1), fh_near, dxyz_near, r_near], dim=-1)
            msg2_in_near = msg_in_near  # 邻居特征不变
            msg2_near = self.msg_mlp2(msg2_in_near)  # [M,k,hid]
            att2_near = self.att_mlp2(att2_in_near).squeeze(-1)  # [M,k]
            att2_near = torch.softmax(att2_near, dim=1)
            agg2_near = torch.sum(att2_near.unsqueeze(-1) * msg2_near, dim=1)  # [M,hid]
            # 远邻（可选）
            if k_far > 0:
                att2_in_far = torch.cat([q_embed.unsqueeze(1).expand(-1, k_far, -1), fh_far, dxyz_far, r_far], dim=-1)
                msg2_in_far = msg_in_far
                msg2_far = self.msg_mlp2(msg2_in_far)
                att2_far = self.att_mlp2(att2_in_far).squeeze(-1)
                att2_far = torch.softmax(att2_far, dim=1)
                agg2_far = torch.sum(att2_far.unsqueeze(-1) * msg2_far, dim=1)
                agg2 = agg2_near + agg2_far
            else:
                agg2 = agg2_near

            # 残差融合（保守）：第一轮与第二轮各占一半
            agg = 0.5 * agg + 0.5 * agg2

        # 进入残差头（关键：统一构造 joint，后面不再使用 joint1）
        joint = torch.cat([agg, q_in], dim=-1)
        d_feat = self.head_feat(joint)
        d_scale = self.head_scale(joint)
        d_off = self.head_offsets(joint)
        d_mask = self.head_mask(joint)

        if apply_conf:
            eps = 1e-6
            sigma_feat = torch.clamp(prior_q['scale_feat'], min=eps)
            sigma_scale = torch.clamp(prior_q['scale_scale'], min=eps)
            sigma_off = torch.clamp(prior_q['scale_offsets'], min=eps)

            tau = float(os.getenv("HAC_GNN_TAU", "3.0"))
            d_feat = torch.clamp(d_feat, -tau * sigma_feat, +tau * sigma_feat)
            d_scale = torch.clamp(d_scale, -tau * sigma_scale, +tau * sigma_scale)
            d_off = torch.clamp(d_off, -tau * sigma_off, +tau * sigma_off)

            alpha = float(os.getenv("HAC_GNN_ALPHA", "0.7"))
            z_f = (d_feat.abs() / sigma_feat).mean(dim=-1);
            z_s = (d_scale.abs() / sigma_scale).mean(dim=-1);
            z_o = (d_off.abs() / sigma_off).mean(dim=-1)
            c_f = torch.exp(-alpha * z_f).clamp(0, 1).unsqueeze(-1);
            c_s = torch.exp(-alpha * z_s).clamp(0, 1).unsqueeze(-1);
            c_o = torch.exp(-alpha * z_o).clamp(0, 1).unsqueeze(-1)

            # 距离权插值（合并近+远）
            d_all = dists_all
            w_all = 1.0 / (d_all + 1e-6)
            w_all = w_all / (w_all.sum(dim=1, keepdim=True) + 1e-8)  # [M,k_total]
            # 拼联合邻域特征
            if k_far > 0:
                feat_cat = torch.cat([feat_near, feat_far], dim=1)  # [M,k_total,D]
                scale_cat = torch.cat([scale_near, scale_far], dim=1)  # [M,k_total,6]
                off_cat = torch.cat([off_near, off_far], dim=1)  # [M,k_total,3K]
                mask_cat = torch.cat([mask_near, mask_far], dim=1)  # [M,k_total,Cm]
            else:
                feat_cat, scale_cat, off_cat, mask_cat = feat_near, scale_near, off_near, mask_near

            feat_interp = torch.sum(w_all.unsqueeze(-1) * feat_cat, dim=1)  # [M,D]
            scale_interp = torch.sum(w_all.unsqueeze(-1) * scale_cat, dim=1)  # [M,6]
            off_interp = torch.sum(w_all.unsqueeze(-1) * off_cat, dim=1)  # [M,3K]
            # Cm 可能是 K 或 K+1
            mask_interp = torch.sum(w_all.unsqueeze(-1) * mask_cat, dim=1)  # [M,Cm]

            # gating
            C_gate = min(d_mask.shape[-1], K)
            if C_gate > 0:
                d_mask[:, :C_gate] = d_mask[:, :C_gate] * c_o

            feat_hat_tmp = mean_feat_q + d_feat
            scale_hat_tmp = mean_scale_q + d_scale
            off_hat_tmp = mean_offsets_q + d_off
            mask_hat_tmp = torch.sigmoid(d_mask)

            T_conf = float(os.getenv("HAC_GNN_CONF_TH", "0.35"))

            # 通道数对齐
            C_pred = mask_hat_tmp.shape[-1]
            C_interp = mask_interp.shape[-1]
            C = min(C_pred, C_interp)

            # 软融合
            feat_hat_blend = c_f * feat_hat_tmp + (1.0 - c_f) * feat_interp
            scale_hat_blend = c_s * scale_hat_tmp + (1.0 - c_s) * scale_interp
            off_hat_blend = c_o * off_hat_tmp + (1.0 - c_o) * off_interp

            mask_hat_blend = mask_hat_tmp.clone()
            if C > 0:
                mask_hat_blend[:, :C] = c_o * mask_hat_tmp[:, :C] + (1.0 - c_o) * mask_interp[:, :C]

            # 硬回退
            low_conf = (c_o.squeeze(-1) < T_conf)
            if low_conf.any():
                idx = low_conf.nonzero(as_tuple=False).view(-1)
                feat_hat_tmp[idx] = feat_interp[idx]
                scale_hat_tmp[idx] = scale_interp[idx]
                off_hat_tmp[idx] = off_interp[idx]
                if C > 0:
                    mask_hat_tmp[idx, :C] = mask_interp[idx, :C]

                feat_hat_final = feat_hat_tmp
                scale_hat_final = scale_hat_tmp
                off_hat_final = off_hat_tmp
                mask_hat_final = mask_hat_tmp
                hi = (~low_conf).nonzero(as_tuple=False).view(-1)
                feat_hat_final[hi] = feat_hat_blend[hi]
                scale_hat_final[hi] = scale_hat_blend[hi]
                off_hat_final[hi] = off_hat_blend[hi]
                if C > 0:
                    mask_hat_final[hi, :C] = mask_hat_blend[hi, :C]
            else:
                feat_hat_final = feat_hat_blend
                scale_hat_final = scale_hat_blend
                off_hat_final = off_hat_blend
                mask_hat_final = mask_hat_blend

            feat_hat = feat_hat_final
            scale_hat = scale_hat_final
            off_hat = off_hat_final.view(M, K, 3)
            mask_hat = mask_hat_final.view(M, C_pred, 1)
            return feat_hat, scale_hat, off_hat, mask_hat

        # 无置信时的直接合成
        feat_hat = mean_feat_q + d_feat
        scale_hat = mean_scale_q + d_scale
        off_hat = (mean_offsets_q + d_off).view(M, K, 3)
        mask_hat = torch.sigmoid(d_mask).view(M, K + 1, 1)
        return feat_hat, scale_hat, off_hat, mask_hat

    @torch.no_grad()
    def smooth_missing(self, x_all, known_mask,
                       feat, scale, off, mask,
                       x_known=None, k=12, alpha=0.7):
        """
        对缺失锚点做一次“邻域距离插值”的保守平滑：new = alpha*pred + (1-alpha)*interp
        - 仅对缺失锚点生效；已知锚点保持不变
        - mask 只对前 K 通道平滑；其余通道保持不变
        """
        device = self.device
        x_all = x_all.to(device);
        known_mask = known_mask.to(device)
        feat = feat.to(device);
        scale = scale.to(device);
        off = off.to(device);
        mask = mask.to(device)

        miss = (~known_mask)
        if miss.sum().item() == 0:
            return feat, scale, off, mask

        x_miss = x_all[miss]
        if x_known is None:
            x_known = x_all[known_mask]

        # 邻居查询：只用已知锚点作为库
        idx, d = self._knn(x_miss, x_known, k)  # [M,k]
        w = 1.0 / (d + 1e-6)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # [M,k]

        # 收集已知属性
        feat_k = feat[known_mask]  # [Nk,D]
        scale_k = scale[known_mask]  # [Nk,6]
        off_k = off[known_mask]  # [Nk,K,3] 或 [Nk,3K]
        mask_k = mask[known_mask]  # [Nk,K+1,1] 或 [Nk,K+1]

        K = off.shape[1]  # 目标 K（off 为 [*,K,3]）

        # offsets: 展平为 [Nk,3K] 再 gather，再还原 [M,K,3]
        if off_k.dim() == 3 and off_k.shape[-1] == 3:
            off_k_flat = off_k.view(off_k.shape[0], 3 * K)
        elif off_k.dim() == 2 and off_k.shape[1] == 3 * K:
            off_k_flat = off_k
        else:
            off_k_flat = off_k.view(off_k.shape[0], -1)

        off_interp_flat = torch.sum(w.unsqueeze(-1) * off_k_flat[idx], dim=1)  # [M,3K]
        off_interp = off_interp_flat.view(-1, K, 3)  # [M,K,3]

        # feat/scale: 直接 gather 与加权
        feat_interp = torch.sum(w.unsqueeze(-1) * feat_k[idx], dim=1)  # [M,D]
        scale_interp = torch.sum(w.unsqueeze(-1) * scale_k[idx], dim=1)  # [M,6]

        # mask: 对齐到 [Nk,C] 再 gather；只融合前 C=min(C_pred,C_kn) 个通道
        mk = mask_k
        if mk.dtype not in (torch.float16, torch.float32, torch.float64):
            mk = mk.float()
        if mk.dim() == 3 and mk.shape[-1] == 1:
            mk = mk.squeeze(-1)  # [Nk,C]
        elif mk.dim() == 1:
            mk = mk.view(-1, 1)

        C_pred = mask.shape[1]
        C_kn = mk.shape[1] if mk.dim() == 2 else 0
        C = min(C_pred, C_kn)
        if C > 0:
            mask_interp = torch.sum(w.unsqueeze(-1) * mk[idx, :, :C], dim=1)  # [M,C]
        else:
            mask_interp = None

        # EMA 融合：只对缺失处
        feat_new = feat.clone();
        feat_new[miss] = alpha * feat[miss] + (1 - alpha) * feat_interp
        scale_new = scale.clone();
        scale_new[miss] = alpha * scale[miss] + (1 - alpha) * scale_interp
        off_new = off.clone();
        off_new[miss] = alpha * off[miss] + (1 - alpha) * off_interp

        mask_new = mask.clone()
        if C > 0:
            mask_new[miss, :C, 0] = alpha * mask[miss, :C, 0] + (1 - alpha) * mask_interp

        return feat_new, scale_new, off_new, mask_new

    def predict(self, gm, x_all, known_mask, feat, scale, offsets, mask):
        """
        对缺失锚点批量预测。若无缺失则返回空张量。
        """
        device = self.device
        x_all = x_all.to(device);
        known_mask = known_mask.to(device)

        # 可选：限制推理期的已知库容量（避免 OOM）
        pred_max_known = int(os.getenv("HAC_GNN_PRED_MAX_KNOWN", "0"))
        if pred_max_known > 0 and known_mask.sum().item() > pred_max_known:
            keep = torch.where(known_mask)[0]
            perm = torch.randperm(keep.numel(), device=device)[:pred_max_known]
            sel = keep[perm]
            known_mask[:] = False
            known_mask[sel] = True

        x_known = x_all[known_mask]
        x_miss = x_all[~known_mask]
        feat_k = feat[known_mask];
        scale_k = scale[known_mask]
        off_k = offsets[known_mask];
        mask_k = mask[known_mask]

        preds = []
        # 可选：查询分块大小（默认 4096 -> 可改小）
        B = int(os.getenv("HAC_GNN_PRED_BQ", "4096"))
        for s in range(0, x_miss.shape[0], B):
            xq = x_miss[s:s + B]
            # 推理期启用置信 gating + 3σ 剪裁
            out = self.forward_once(gm, xq, x_known, feat_k, scale_k, off_k, mask_k, apply_conf=True)
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
        场景内自适应（可选）：按 .pak 组级伪丢包采样，剩下当邻域库，最小监督优化
        - 训练可观测：tqdm 进度条 + 动态 loss 显示 + 定期验证
        - 稳定性：梯度裁剪（clip_grad），可选余弦退火
        - 偏置修正：offsets 的 L1 在 mask=1 的位置加权更大（只用前 K 通道）
        - 正则增强：插值蒸馏（默认 0.2）+ 先验一致性 z-score（默认 0.1）
          可通过环境变量覆盖：
            HAC_GNN_LAMBDA_DISTILL（默认 0.2）
            HAC_GNN_LAMBDA_PRIOR  （默认 0.1）
        """
        import os
        device = self.device
        self.to(device)
        self.train()
        if k is not None:
            self.k = k

        lam_distill = float(os.getenv("HAC_GNN_LAMBDA_DISTILL", "0.2"))
        lam_prior = float(os.getenv("HAC_GNN_LAMBDA_PRIOR", "0.1"))

        # 采样已知集合（限制规模，避免 OOM）
        idx_known_all = torch.where(known_mask.to('cpu'))[0]
        if idx_known_all.numel() < (self.k * 2 + 8):
            return  # 可用已知点太少，跳过训练
        if idx_known_all.numel() > max_known:
            perm = torch.randperm(idx_known_all.numel())[:max_known]
            idx_known_all = idx_known_all[perm]

        x_k = x_all[idx_known_all].to(device)
        feat_k = feat[idx_known_all].to(device)
        scale_k = scale[idx_known_all].to(device)
        off_k = offsets[idx_known_all].to(device)
        msk_k = mask[idx_known_all].to(device)

        # 组级采样：构造查询/库
        if (group_ids is not None) and (group_ids.numel() == known_mask.numel()):
            gid_sub = group_ids[idx_known_all]  # CPU long
            uniq = torch.unique(gid_sub)  # CPU
            n_grp = uniq.numel()
            gq = max(1, int(n_grp * float(group_rate)))
            permg = torch.randperm(n_grp)[:gq]  # CPU
            q_groups = uniq[permg]  # CPU

            q_mask_sub = torch.isin(gid_sub, q_groups)  # CPU
            pool_mask_sub = ~q_mask_sub  # CPU
            q_idx_sub = torch.where(q_mask_sub)[0]  # CPU
            pool_idx_sub = torch.where(pool_mask_sub)[0]  # CPU

            if pool_idx_sub.numel() < (self.k + 8):
                # 回退到随机划分（GPU）
                n = x_k.shape[0]
                n_query = min(n // 4, 8000)
                perm = torch.randperm(n, device=device)
                q_idx_sub = perm[:n_query]  # CUDA
                pool_idx_sub = perm[n_query:]  # CUDA
            else:
                # 把 CPU 索引搬到 GPU
                q_idx_sub = q_idx_sub.to(device)
                pool_idx_sub = pool_idx_sub.to(device)
        else:
            # 无组 id：回退到随机划分
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

        # 关键：在创建优化器前，用一个小批做一次干跑以触发懒构建
        if not self._built:
            with torch.no_grad():
                b0 = min(batch_q, x_q.shape[0])
                if b0 > 0:
                    _ = self.forward_once(gm, x_q[:b0], x_pool, feat_pool, scale_pool, off_pool, msk_pool,
                                          apply_conf=False)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps)) if cosine_decay else None

        # 验证子集
        has_val = val_ratio > 0 and x_q.shape[0] >= 32
        if has_val:
            n_val = max(8, int(x_q.shape[0] * val_ratio))
            v_idx = torch.randperm(x_q.shape[0], device=device)[:n_val]
            x_v = x_q[v_idx]
            gt_v_idx = q_idx_sub[v_idx]  # 用于从 feat_k/scale_k/off_k/msk_k 提取 GT

        # 进度条
        try:
            from tqdm import trange
            iters = trange(steps, desc="GNN fit", dynamic_ncols=True)
        except Exception:
            iters = range(steps)

        for it in iters:
            start = (it * batch_q) % max(1, x_q.shape[0])
            end = min(x_q.shape[0], start + batch_q)
            if end - start < 1:
                if hasattr(iters, "set_postfix"): iters.set_postfix(skip="1")
                continue

            xq_b = x_q[start:end]
            feat_hat, scale_hat, off_hat, mask_hat = self.forward_once(
                gm, xq_b, x_pool, feat_pool, scale_pool, off_pool, msk_pool, apply_conf=False
            )

            # 监督目标（已知真值）
            gt_idx = q_idx_sub[start:end]
            feat_gt = feat_k[gt_idx]
            scale_gt = scale_k[gt_idx]
            off_gt = off_k[gt_idx]
            msk_gt = msk_k[gt_idx]

            # 基本损失：offsets 在 mask=1 处权重大（只用前 K 通道）
            K = gm.n_offsets
            B = xq_b.shape[0]
            l_feat = F.l1_loss(feat_hat, feat_gt)
            l_scale = F.l1_loss(scale_hat, scale_gt)
            eps_c = 1e-3
            diff = off_hat - off_gt
            l_off_all = torch.sqrt(diff * diff + eps_c * eps_c)  # [B,K,3]
            w_mask = torch.clamp(msk_gt[:, :K, :], 0, 1)  # [B,K,1]
            l_off = (l_off_all * (1.0 + 2.0 * w_mask)).mean()
            l_mask = F.binary_cross_entropy(mask_hat, torch.clamp(msk_gt, 0, 1))

            # ===== 插值蒸馏（邻域距离权重） =====
            with torch.no_grad():
                # 用同一个 k 做插值
                nbr_idx, dists = self._knn(xq_b, x_pool, self.k)  # [B,k], [B,k]
                kk = nbr_idx.shape[1]
                feat_n = feat_pool[nbr_idx]  # [B,k,D]
                scale_n = scale_pool[nbr_idx]  # [B,k,6]
                off_n = off_pool[nbr_idx].view(B, kk, 3 * K)  # [B,k,3K]
                # 权重：1/(d+eps)
                w = 1.0 / (dists + 1e-6)
                w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # [B,k]
                # 插值结果
                feat_interp = torch.sum(w.unsqueeze(-1) * feat_n, dim=1)  # [B,D]
                scale_interp = torch.sum(w.unsqueeze(-1) * scale_n, dim=1)  # [B,6]
                off_interp = torch.sum(w.unsqueeze(-1) * off_n, dim=1).view(B, K, 3)  # [B,K,3]

            l_distill = lam_distill * (F.l1_loss(feat_hat, feat_interp) +
                                       F.l1_loss(scale_hat, scale_interp) +
                                       F.l1_loss(off_hat, off_interp))

            # ===== 先验一致性（z-score L1）=====
            with torch.no_grad():
                prior = self._compute_prior(gm, xq_b)
            eps = 1e-6
            sigma_feat = torch.clamp(prior['scale_feat'], min=eps)  # [B,D]
            sigma_scale = torch.clamp(prior['scale_scale'], min=eps)  # [B,6]
            sigma_off = torch.clamp(prior['scale_offsets'], min=eps).view(B, 3 * K)  # [B,3K]

            z_feat = ((feat_hat - prior['mean_feat']) / sigma_feat).abs().mean()
            z_scale = ((scale_hat - prior['mean_scale']) / sigma_scale).abs().mean()
            z_off = ((off_hat.view(B, 3 * K) - prior['mean_offsets']) / sigma_off).abs().mean()
            l_prior = lam_prior * (z_feat + z_scale + z_off)

            # 总损失
            loss = l_feat + 0.5 * l_scale + l_off + 0.5 * l_mask + l_distill + l_prior

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            opt.step()
            if sch is not None:
                sch.step()

            # 进度条尾部打印
            if hasattr(iters, "set_postfix"):
                iters.set_postfix(
                    feat=f"{l_feat.item():.4f}",
                    scale=f"{l_scale.item():.4f}",
                    off=f"{l_off.item():.4f}",
                    mask=f"{l_mask.item():.4f}",
                    dist=f"{(l_distill / max(1e-8, lam_distill)).item():.4f}",
                    prior=f"{(l_prior / max(1e-8, lam_prior)).item():.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )

            # 控制台周期性打印
            if (it + 1) % log_every == 0 or (it + 1) == steps:
                if os.getenv("HAC_GNN_FIT_LOG", "1") == "1":
                    print(f"[fit] it={it + 1}/{steps} | "
                          f"l_feat={l_feat.item():.4f} l_scale={l_scale.item():.4f} "
                          f"l_off={l_off.item():.4f} l_mask={l_mask.item():.4f} "
                          f"l_dist={l_distill.item():.4f} l_prior={l_prior.item():.4f} "
                          f"lr={opt.param_groups[0]['lr']:.2e}")

            # 轻量验证
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

                    l1_f = F.l1_loss(feat_v, feat_gt_v).item()
                    l1_s = F.l1_loss(scale_v, scale_gt_v).item()
                    l1_o = F.l1_loss(off_v, off_gt_v).item()
                    bce_m = F.binary_cross_entropy(mask_v, torch.clamp(msk_gt_v, 0, 1)).item()
                    print(f"[val] it={it + 1} | L1(feat)={l1_f:.4f} L1(scale)={l1_s:.4f} "
                          f"L1(off)={l1_o:.4f} BCE(mask)={bce_m:.4f}")
                self.train()


