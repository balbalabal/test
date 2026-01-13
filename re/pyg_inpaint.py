# utils/pyg_inpaint.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATv2Conv
from torch_cluster import knn as _knn
# from torch_geometric.nn import EdgeConv, GATv2Conv
# from torch_cluster import knn as _knn

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

    def __init__(self,
                 k=16,
                 hidden=128,
                 device='cuda',
                 mp_iters=1,
                 conv_type=None,   # 'edge' or 'gat'
                 k2=None,          # 第二层邻域数；None/<=0 表示用 k
                 heads=None):      # GATv2 多头
        super().__init__()
        self.k = int(k)
        self.hid = int(hidden)
        self.device = device
        self.mp_iters = int(mp_iters)

        # 第二层 conv 类型
        conv_env = os.getenv("HAC_GNN_CONV", "edge").lower()
        self.conv2_kind = (conv_type or conv_env or "edge").lower()

        # 第二层 k2：参数 > 环境变量 > 默认等于 k
        k2_env = int(os.getenv("HAC_GNN_K2", str(self.k)))
        if k2 is not None and int(k2) > 0:
            self.k2 = int(k2)
        else:
            self.k2 = int(k2_env) if k2_env > 0 else self.k

        # GATv2 多头
        heads_env = int(os.getenv("HAC_GNN_HEADS", "2"))
        self.heads = int(heads if heads is not None else heads_env)

        # === ADC-lite: 自适应扩散半径（热核权） ===
        # 启用开关
        self.adc = int(os.getenv("HAC_GNN_ADC", "0")) > 0
        # 初始半径参数 gamma（先标量，按层各一个；softplus(gamma)>0）
        gamma_init = float(os.getenv("HAC_GNN_GAMMA_INIT", "1.0"))
        self.gamma1 = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))
        self.gamma2 = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32)) if self.mp_iters >= 2 else None
        # 距离归一化尺度（voxel_size 的倍数）
        self.r_scale = float(os.getenv("HAC_GNN_R_SCALE", "8.0"))


        self._built = False

    def _build(self, in_dim, D, K):
        # 第一层 EdgeConv（保持不变）
        self.conv1 = EdgeConv(mlp([2 * in_dim, self.hid, self.hid]), aggr='mean')

        # 第二层：可选 EdgeConv 或 GATv2Conv
        if self.mp_iters >= 2:
            if self.conv2_kind == "gat":
                # GATv2 多头，out_per_head = hid // heads（需整除）
                if self.hid % self.heads != 0:
                    out_per_head = max(1, self.hid // self.heads)
                    # 若不能整除，会得到 heads*out_per_head != hid，但后续 MLP 仍接受 hid 维（可加线性投影到 hid）
                    self.conv2 = GATv2Conv(in_channels=self.hid,
                                           out_channels=out_per_head,
                                           heads=self.heads,
                                           concat=True)  # 输出维度 heads*out_per_head
                    self.proj2 = nn.Linear(self.heads * out_per_head, self.hid, bias=True)
                else:
                    out_per_head = self.hid // self.heads
                    self.conv2 = GATv2Conv(in_channels=self.hid,
                                           out_channels=out_per_head,
                                           heads=self.heads,
                                           concat=True)
                    self.proj2 = None
                self.conv2_type = "gat"
            else:
                self.conv2 = EdgeConv(mlp([2 * self.hid, self.hid, self.hid]), aggr='mean')
                self.conv2_type = "edge"
        else:
            self.conv2 = None
            self.conv2_type = None

        # 头部
        self.head_feat = mlp([self.hid, self.hid, D])
        self.head_scale = mlp([self.hid, self.hid, 6])
        self.head_offsets = mlp([self.hid, self.hid, 3 * K])
        self.head_mask = mlp([self.hid, self.hid, K + 1])

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

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _heat_weight(self, r: torch.Tensor, gamma_param: torch.Tensor, voxel_size: float) -> torch.Tensor:
        """
        r: [E] 边长，gamma_param: 标量参数，voxel_size: 场景体素大小
        返回热核权: exp( - softplus(gamma) * (r / (voxel_size * r_scale))^2 )
        """
        g = self._softplus(gamma_param)[0]  # 标量>0
        rn = r / max(1e-9, voxel_size * self.r_scale)
        return torch.exp(-g * (rn * rn)).clamp_min(1e-12)  # [E]


    def _gating_and_blend(self, gm, x_q, d_feat, d_scale, d_off, d_mask,
                          feat_interp=None, scale_interp=None, off_interp=None, mask_interp=None):
        """
        残差 + 置信 gating + 3σ 剪裁 + 分项阈值回退 + （可选）插值融合
        - Debug: HAC_GNN_DEBUG=1 时打印 tau/alpha/各阈值与低置信比例
        - SAFE:  HAC_GNN_SAFE=1 时直接返回“插值基线 + 全零 mask”（若基线不存在则返回先验均值 + 全零 mask）
        - 分项开关：HAC_GNN_RESID_OFF/FEAT/SCALE=0 可禁用某一项残差
        """
        import os
        eps = 1e-6
        D = gm.feat_dim
        K = gm.n_offsets

        # 先验
        fh_q = gm.calc_interp_feat(x_q)
        prior = gm.get_grid_mlp(fh_q)
        m_f, s_f, _, m_s, s_s, m_o, s_o, _, _, _ = torch.split(prior, [D, D, D, 6, 6, 3 * K, 3 * K, 1, 1, 1], dim=-1)
        s_f = torch.clamp(s_f, min=eps)
        s_s = torch.clamp(s_s, min=eps)
        s_o = torch.clamp(s_o, min=eps)

        # 3σ 剪裁
        tau = float(os.getenv("HAC_GNN_TAU", "3.0"))
        d_feat = torch.clamp(d_feat, -tau * s_f, +tau * s_f)
        d_scale = torch.clamp(d_scale, -tau * s_s, +tau * s_s)
        d_off = torch.clamp(d_off, -tau * s_o, +tau * s_o)

        # 残差分项开关（默认启用，置 0 可禁用某项残差）
        if int(os.getenv("HAC_GNN_RESID_FEAT", "1")) == 0:
            d_feat = d_feat * 0.0
        if int(os.getenv("HAC_GNN_RESID_SCALE", "1")) == 0:
            d_scale = d_scale * 0.0
        if int(os.getenv("HAC_GNN_RESID_OFF", "1")) == 0:
            d_off = d_off * 0.0

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

        # SAFE 兜底：直接退到插值基线（若无基线则先验），并将 mask 置零，避免任何修复参与渲染
        if int(os.getenv("HAC_GNN_SAFE", "0")) == 1:
            M = x_q.shape[0]
            if (feat_interp is not None) and (scale_interp is not None) and (off_interp is not None):
                feat_safe = feat_interp
                scale_safe = scale_interp
                off_safe = off_interp
            else:
                feat_safe = m_f
                scale_safe = m_s
                off_safe = m_o.view(-1, K, 3)
            mask_safe = torch.zeros((M, K + 1, 1), device=feat_hat.device, dtype=feat_hat.dtype)
            # Debug 打印一行
            if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                print("[gnn] SAFE=1: return baseline (interp or prior) + zero masks")
            return feat_safe, scale_safe, off_safe, mask_safe

        # 融合/回退（仅推理期：有基线时才做；训练期 feat_interp 为 None）
        use_blend = (feat_interp is not None)
        if use_blend:
            # 分项阈值
            T_off = float(os.getenv("HAC_GNN_CONF_TH_OFF", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            T_fea = float(os.getenv("HAC_GNN_CONF_TH_FEA", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            T_sca = float(os.getenv("HAC_GNN_CONF_TH_SCA", os.getenv("HAC_GNN_CONF_TH", "0.4")))
            lo = (co.squeeze(-1) < T_off)  # offsets 低置信 [M]
            lf = (cf.squeeze(-1) < T_fea)  # feat     低置信 [M]
            ls = (cs.squeeze(-1) < T_sca)  # scale    低置信 [M]

            # 正确广播形状
            cf2 = cf.view(-1, 1)  # [M,1] -> 用于 [M,D]
            cs2 = cs.view(-1, 1)  # [M,1] -> 用于 [M,6]
            co2 = co.view(-1, 1)  # [M,1] -> 用于 [M,C]
            co3 = co.view(-1, 1, 1)  # [M,1,1] -> 用于 [M,K,3]

            # 软融合
            feat_bl = cf2 * feat_hat + (1.0 - cf2) * feat_interp  # [M,D]
            scale_bl = cs2 * scale_hat + (1.0 - cs2) * scale_interp  # [M,6]
            off_bl = co3 * off_hat + (1.0 - co3) * off_interp  # [M,K,3]

            # 无原地写入：先对低置信做“硬回退”，再对高置信做“软融合”
            lo3 = lo.view(-1, 1, 1)  # [M,1,1]
            lf2 = lf.view(-1, 1)  # [M,1]
            ls2 = ls.view(-1, 1)  # [M,1]

            # offsets
            off_low = torch.where(lo3, off_interp, off_hat)  # 低置信退回插值
            off_final = torch.where(~lo3, off_bl, off_low)  # 高置信采用软融合

            # feat
            feat_low = torch.where(lf2, feat_interp, feat_hat)
            feat_final = torch.where(~lf2, feat_bl, feat_low)

            # scale
            scale_low = torch.where(ls2, scale_interp, scale_hat)
            scale_final = torch.where(~ls2, scale_bl, scale_low)

            # mask（仅融合/回退前 C 个通道；其余通道保持原样），也避免原地写入
            C = mask_hat.shape[1]
            if mask_interp is not None:
                C = min(C, mask_interp.shape[1])

            if C > 0:
                mask_hat_firstC = mask_hat[:, :C, :]  # [M,C,1]
                mask_interp_firstC = mask_interp[:, :C].unsqueeze(-1)  # [M,C,1]
                co2e = co2.unsqueeze(-1)  # [M,1,1]

                # 软融合的 mask（前 C 通道）
                mask_bl_firstC = co2e * mask_hat_firstC + (1.0 - co2e) * mask_interp_firstC  # [M,C,1]
                lo_mask = lo.view(-1, 1, 1)  # [M,1,1]

                # 低置信硬回退 -> 高置信软融合（前 C 通道）
                mask_low_firstC = torch.where(lo_mask, mask_interp_firstC, mask_hat_firstC)
                mask_firstC_final = torch.where(~lo_mask, mask_bl_firstC, mask_low_firstC)

                # 合并回所有通道
                if mask_hat.shape[1] > C:
                    mask_rest = mask_hat[:, C:, :]  # [M,C2,1]
                    mask_final = torch.cat([mask_firstC_final, mask_rest], dim=1)
                else:
                    mask_final = mask_firstC_final
            else:
                mask_final = mask_hat

            # 用无原地写入的结果覆盖变量
            feat_hat = feat_final
            scale_hat = scale_final
            off_hat = off_final
            mask_hat = mask_final
            mask_hat = torch.clamp(mask_hat, 0.0, 1.0)
            if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                with torch.no_grad():
                    mi, ma = float(mask_hat.min().item()), float(mask_hat.max().item())
                    if not (0.0 - 1e-5 <= mi and ma <= 1.0 + 1e-5):
                        print(f"[gnn][warn] mask_hat out of [0,1]: min={mi:.4f}, max={ma:.4f}")

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
        # 第二层（可选，支持 k2）
        used_edge2_for_interp = False


        D = gm.feat_dim
        K = gm.n_offsets

        # 统一节点特征（查询=先验，已知=真实）
        X, in_dim = self._compose_nodes(gm, x_query, x_known, feat_known, scale_known, offsets_known, mask_known)
        if not self._built:
            self._build(in_dim, D, K)
            if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                conv2 = getattr(self, "conv2_type", None) or getattr(self, "conv2_kind", "none")
                print(f"[gnn] build dims: in_dim={in_dim}, D={D}, K={K}, hid={self.hid}, "
                      f"k={self.k}, k2={getattr(self, 'k2', self.k)}, mp={self.mp_iters}, "
                      f"conv2={conv2}, heads={getattr(self, 'heads', 1)}")

        # 第一层 EdgeConv（k1）
        edge_index1 = self._knn_bipartite(x_known, x_query, self.k)
        h = self.conv1(X, edge_index1)

        # 第二层（可选，支持 k2）
        if self.conv2 is not None:
            edge_index2 = self._knn_bipartite(x_known, x_query, self.k2)
            if self.conv2_type == "gat":
                h = self.conv2(h, edge_index2)  # [M+Nk, heads*out_per_head]
                if getattr(self, "proj2", None) is not None:
                    h = self.proj2(h)  # -> [M+Nk, hid]
            else:
                h = self.conv2(h, edge_index2)
            edge_interp = edge_index2
            used_edge2_for_interp = True
        else:
            edge_interp = edge_index1

        # 只取查询部分
        h_q = h[:x_query.shape[0]]  # [M, hid]

        # 回归头（对先验做残差）
        d_feat = self.head_feat(h_q)  # [M,D]
        d_scale = self.head_scale(h_q)  # [M,6]
        d_off = self.head_offsets(h_q)  # [M,3K]
        d_mask = self.head_mask(h_q)  # [M,K+1]

        if not apply_conf:
            # 训练路径：直接做 gating（无融合）
            feat_hat, scale_hat, off_hat, mask_hat = self._gating_and_blend(
                gm, x_query, d_feat, d_scale, d_off, d_mask,
                None, None, None, None
            )
            return feat_hat, scale_hat, off_hat, mask_hat

        # 推理路径：基于所选边 edge_interp 做距离插值（作为融合/回退的基线）
        M = x_query.shape[0]
        src, dst = edge_interp[0], edge_interp[1]  # src: known+M, dst: query
        pos_all = torch.cat([x_query, x_known], dim=0)
        dq = pos_all[dst] - pos_all[src]  # [E,3]
        r = torch.norm(dq, dim=-1) + 1e-9
        if self.adc:
            # 选择对应层的 gamma：edge_interp 用哪层边，就用哪层 gamma
            gamma_param = self.gamma2 if (used_edge2_for_interp and self.gamma2 is not None) else self.gamma1
            w = self._heat_weight(r, gamma_param, gm.voxel_size)  # [E]
            if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                g1 = float(self._softplus(self.gamma1).item())
                g2 = float(self._softplus(self.gamma2).item()) if self.gamma2 is not None else None
                print(
                    f"[gnn][adc] heat-kernel: gamma1={g1:.4f}, gamma2={g2 if g2 is not None else 'None'}, r_scale={self.r_scale}")
        else:
            w = 1.0 / r
        w_sum = torch.zeros(M, device=device).index_add_(0, dst, w) + 1e-12
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
        # mask -> [Nk, K+1] float（squeeze + pad/trim）
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
        mask_k = torch.clamp(mask_k, 0.0, 1.0)

        # 将 src 映射回已知子图索引
        src_k = src - M
        feat_interp = torch.zeros((M, D), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * feat_k[src_k])
        scale_interp = torch.zeros((M, 6), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * scale_k[src_k])
        off_interp = torch.zeros((M, 3 * K), device=device).index_add_(0, dst,
                                                                       w_norm.unsqueeze(-1) * off_k[src_k]).view(M, K,
                                                                                                                 3)
        C = K + 1
        mask_interp = torch.zeros((M, C), device=device).index_add_(0, dst, w_norm.unsqueeze(-1) * mask_k[src_k])
        mask_interp = torch.clamp(mask_interp, 0.0, 1.0)
        # 保险模式：只用插值（调试/坏权重兜底）
        if os.getenv("HAC_GNN_SAFE", "0") == "1":
            mask_safe = torch.zeros((M, K + 1, 1), device=device)
            C_safe = min(mask_interp.shape[1], K + 1)
            if C_safe > 0:
                mask_safe[:, :C_safe, 0] = mask_interp[:, :C_safe]
            return feat_interp, scale_interp, off_interp, mask_safe

        # 残差+gating+融合
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
        if os.getenv("HAC_GNN_DEBUG", "0") == "1":
            print(f"[gnn] predict: N={x_all.shape[0]}, known={x_k.shape[0]}, miss={x_q.shape[0]}, "
                  f"pred_max_known={pred_max_known}, B={B}")

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
            group_ids: torch.Tensor = None, group_rate: float = 0.2,
            save_best_path: str = None, save_meta: dict = None):
        """
        场景内自监督训练（组级伪丢包）：
        - L1/BCE + Charbonnier offsets（稳一些）
        - 验证分数：val_score = L1(off) + 0.3*L1(feat) + 0.1*L1(scale) + 0.5*BCE(mask)（越小越好）
        - 支持保存验证期最优权重：save_best_path + save_meta
        """
        import os
        device = self.device
        self.to(device);
        self.train()
        if k is not None:
            self.k = k

        idx_known_all = torch.where(known_mask.to('cpu'))[0]
        if idx_known_all.numel() < (self.k * 2 + 8):
            print("[fit] too few known anchors; skip training")
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
            q_idx_sub = torch.where(q_mask_sub)[0]
            pool_idx_sub = torch.where(pool_mask_sub)[0]
            # 搬到 GPU，避免后续 CPU/GPU 索引混用
            q_idx_sub = q_idx_sub.to(device);
            pool_idx_sub = pool_idx_sub.to(device)
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

        # 懒构建触发（小批，apply_conf=False）
        if not self._built:
            with torch.no_grad():
                b0 = min(batch_q, x_q.shape[0])
                if b0 > 0:
                    _ = self.forward_once(gm, x_q[:b0], x_pool, feat_pool, scale_pool, off_pool, msk_pool,
                                          apply_conf=False)
        if os.getenv("HAC_GNN_DEBUG", "0") == "1":
            print(f"[fit] adc={self.adc} gamma_init={float(self._softplus(self.gamma1).item()):.4f} "
                  f"t_lr={os.getenv('HAC_GNN_T_LR', '5e-3')} r_scale={self.r_scale} "
                  f"backend=pyg conv={self.conv2_kind} mp={self.mp_iters} k={self.k} k2={getattr(self, 'k2', self.k)} hid={self.hid}")

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps)) if cosine_decay else None

        # === ADC-lite: gamma 优化器（仅更新半径参数） ===
        if self.adc:
            gamma_params = [self.gamma1]
            if self.mp_iters >= 2 and self.gamma2 is not None:
                gamma_params.append(self.gamma2)
            t_lr = float(os.getenv("HAC_GNN_T_LR", "5e-3"))
            opt_gamma = torch.optim.Adam(gamma_params, lr=t_lr)
        else:
            gamma_params = []
            opt_gamma = None

        # 验证子集
        has_val = val_ratio > 0 and x_q.shape[0] >= 32
        if has_val:
            n_val = max(8, int(x_q.shape[0] * val_ratio))
            v_idx = torch.randperm(x_q.shape[0], device=q_idx_sub.device)[:n_val]
            x_v = x_q[v_idx]
            gt_v_idx = q_idx_sub[v_idx]
        else:
            x_v = None;
            gt_v_idx = None

        # best 记录
        best_score = float("inf")
        best_step = -1

        # 训练循环
        try:
            from tqdm import trange
            iters = trange(steps, desc="PYG fit", dynamic_ncols=True)
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
            l_off_all = torch.sqrt(diff * diff + eps_c * eps_c)  # [B,K,3]
            w_mask = torch.clamp(msk_gt[:, :K, :], 0, 1)  # [B,K,1]
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
                    print(f"[fit] it={it + 1}/{steps} | "
                          f"l_feat={l_feat.item():.4f} l_scale={l_scale.item():.4f} "
                          f"l_off={l_off.item():.4f} l_mask={l_mask.item():.4f} "
                          f"lr={opt.param_groups[0]['lr']:.2e}")

            # 轻量验证 + 保存 best
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

                    val_score = l1o + 0.3 * l1f + 0.1 * l1s + 0.5 * bcem

                    print(f"[val] it={it + 1} | L1(feat)={l1f:.4f} L1(scale)={l1s:.4f} "
                          f"L1(off)={l1o:.4f} BCE(mask)={bcem:.4f} | score={val_score:.4f}")

                    # === ADC-lite: 用验证子集更新 gamma（仅半径参数，冻结主干）===
                    if self.adc and opt_gamma is not None:
                        # 冻结主干，只保留 gamma 梯度
                        for p in self.parameters():
                            p.requires_grad_(False)
                        gamma_params = [self.gamma1] + (
                            [self.gamma2] if (self.mp_iters >= 2 and self.gamma2 is not None) else [])
                        for p in gamma_params:
                            p.requires_grad_(True)

                        self.train()  # 进入训练模式（使 BN/Dropout 等与 val 区分；主要是需要梯度）
                        # 重要：确保不在 no_grad 里
                        with torch.enable_grad():
                            # 重新 forward 验证 batch（走插值/融合路径，让 gamma 进图）
                            feat_v2, scale_v2, off_v2, mask_v2 = self.forward_once(
                                gm, x_v, x_pool, feat_pool, scale_pool, off_pool, msk_pool, apply_conf=True
                            )
                            l1f2 = F.l1_loss(feat_v2, feat_k[gt_v_idx])
                            l1s2 = F.l1_loss(scale_v2, scale_k[gt_v_idx])
                            l1o2 = F.l1_loss(off_v2, off_k[gt_v_idx])
                            bce2 = F.binary_cross_entropy(mask_v2, torch.clamp(msk_k[gt_v_idx], 0, 1))
                            val_score2 = l1o2 + 0.3 * l1f2 + 0.1 * l1s2 + 0.5 * bce2

                            # Debug：看一眼梯度开关与参数状态
                            if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                                print(f"[gnn][adc] enable_grad={torch.is_grad_enabled()} "
                                      f"val_score2.requires_grad={val_score2.requires_grad} "
                                      f"gamma1.req={self.gamma1.requires_grad} "
                                      f"gamma2.req={(self.gamma2.requires_grad if (self.gamma2 is not None) else 'None')}")

                            opt_gamma.zero_grad(set_to_none=True)
                            val_score2.backward()
                            opt_gamma.step()

                        # 恢复主干参数的 requires_grad 与 eval 模式
                        for p in self.parameters():
                            p.requires_grad_(True)
                        self.eval()

                        if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                            g1 = float(self._softplus(self.gamma1).item())
                            g2 = float(self._softplus(self.gamma2).item()) if self.gamma2 is not None else None
                            print(
                                f"[gnn][adc] gamma updated: gamma1={g1:.4f}, gamma2={g2 if g2 is not None else 'None'}")

                    # 保存 best
                    if save_best_path and (val_score < best_score):
                        best_score = val_score
                        best_step = it + 1
                        payload = {"state_dict": self.state_dict()}
                        if save_meta:
                            payload.update(save_meta)
                        torch.save(payload, save_best_path)
                        print(f"[best] step={best_step} score={best_score:.4f} -> saved to {save_best_path}")

                    self.train()

        # 训练结束提示 best
        if save_best_path:
            if best_step > 0:
                print(f"[fit] best model at step={best_step} with score={best_score:.4f} -> {save_best_path}")
            else:
                print("[fit] no best model saved (val_ratio<=0 or no improvement)")


