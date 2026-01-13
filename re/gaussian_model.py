#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order
from utils.gnn_inpaint import GNNInpaintor
from utils.pyg_inpaint import PYGInpaintor


bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

def _dump_gnn_runtime_cfg():
    import os
    cfg = {
        "backend": os.getenv("HAC_GNN_BACKEND", "vanilla"),
        "k": int(os.getenv("HAC_GNN_K", "16")),
        "k2": int(os.getenv("HAC_GNN_K2", os.getenv("HAC_GNN_K", "16"))),
        "hidden": int(os.getenv("HAC_GNN_H", "128")),
        "mp": int(os.getenv("HAC_GNN_MP", "1")),
        "conv": os.getenv("HAC_GNN_CONV", "edge"),
        "heads": int(os.getenv("HAC_GNN_HEADS", "2")),
        "safe": int(os.getenv("HAC_GNN_SAFE", "0")),
        "tau": float(os.getenv("HAC_GNN_TAU", "3.0")),
        "alpha": float(os.getenv("HAC_GNN_ALPHA", "0.7")),
        "conf_off": float(os.getenv("HAC_GNN_CONF_TH_OFF", os.getenv("HAC_GNN_CONF_TH", "0.4"))),
        "conf_fea": float(os.getenv("HAC_GNN_CONF_TH_FEA", os.getenv("HAC_GNN_CONF_TH", "0.4"))),
        "conf_sca": float(os.getenv("HAC_GNN_CONF_TH_SCA", os.getenv("HAC_GNN_CONF_TH", "0.4"))),
        "pred_max_known": int(os.getenv("HAC_GNN_PRED_MAX_KNOWN", "0")),
        "pred_bq": int(os.getenv("HAC_GNN_PRED_BQ", "4096")),
        "att_decay": float(os.getenv("HAC_GNN_ATT_DECAY", "0.0")),
        "r_scale": float(os.getenv("HAC_GNN_R_SCALE", "8.0")),
        "r_th_near": float(os.getenv("HAC_GNN_R_TH_NEAR", "0.0")),
        "r_th_far": float(os.getenv("HAC_GNN_R_TH_FAR", "0.0")),
    }
    print(f"[decode] GNN runtime cfg: {cfg}")


class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i

class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50*3+10*0, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50*3+10*1, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50*3+10*2, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50*3+10*3, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50*3+10*4, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=50,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+feat_dim+1+1+1),
        ).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea().cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def set_encoding_params(self, params):
        """
        设置 encoding 的参数（用于解码后恢复）
        params: 展平的参数张量
        """
        if self.use_2D:
            # 计算每个编码器的参数数量
            param_counts = [
                self.encoding_xyz.encoding_xyz.params.numel(),
                self.encoding_xyz.encoding_xy.params.numel(),
                self.encoding_xyz.encoding_xz.params.numel(),
                self.encoding_xyz.encoding_yz.params.numel(),
            ]

            # 分割参数
            param_splits = torch.split(params, param_counts)

            # 恢复到对应的编码器
            with torch.no_grad():
                self.encoding_xyz.encoding_xyz.params.copy_(
                    param_splits[0].view_as(self.encoding_xyz.encoding_xyz.params))
                self.encoding_xyz.encoding_xy.params.copy_(
                    param_splits[1].view_as(self.encoding_xyz.encoding_xy.params))
                self.encoding_xyz.encoding_xz.params.copy_(
                    param_splits[2].view_as(self.encoding_xyz.encoding_xz.params))
                self.encoding_xyz.encoding_yz.params.copy_(
                    param_splits[3].view_as(self.encoding_xyz.encoding_yz.params))
        else:
            # 单个 3D 编码器
            with torch.no_grad():
                self.encoding_xyz.params.copy_(params.view_as(self.encoding_xyz.params))

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._mask,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask[:, :10, :]
        mask_sig = torch.sigmoid(self._mask[:, :10, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        mask = self.get_mask  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets+1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets+1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x


    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        probs = torch.stack([prob, prob_adj], dim=-1)
        probs = torch.softmax(probs, dim=-1)

        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        bit_feat = self.EG_mix_prob_2.forward(_feat,
                                            mean, mean_adj,
                                            scale, scale_adj,
                                            probs[..., 0], probs[..., 1],
                                            Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info


    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        import json
        import numpy as np

        # ---- pack helpers: group-level .pak ----
        def _collect_group_files_for_base(base_path: str):
            """收集 base(.b) 与其 chunk（base_*.b）"""
            files = []
            pref = base_path[:-2] if base_path.endswith('.b') else base_path
            gl = glob.glob(pref + '_*.b')
            if len(gl) > 0:
                files.extend(sorted(gl))
            elif os.path.exists(base_path):
                files.append(base_path)
            return files

        def _pack_group_pak(pak_path: str, file_list: list, delete_temp: bool = True):
            """
            简单 .pak 容器：magic(4)=PAK1 + count(u32LE) + entries[*]
            每个 entry: name_len(u32LE) + name(bytes) + size(u64LE) + content
            只存放 basename，解包时写回 pre_path_name 下。
            """
            # 去重且只保留存在的文件
            uniq = []
            seen = set()
            for f in file_list:
                if (f not in seen) and os.path.exists(f):
                    uniq.append(f)
                    seen.add(f)

            if len(uniq) == 0:
                return

            with open(pak_path, 'wb') as fw:
                fw.write(b'PAK1')
                fw.write(len(uniq).to_bytes(4, 'little', signed=False))
                for fp in uniq:
                    name = os.path.basename(fp).encode('utf-8')
                    size = os.path.getsize(fp)
                    fw.write(len(name).to_bytes(4, 'little', signed=False))
                    fw.write(name)
                    fw.write(int(size).to_bytes(8, 'little', signed=False))
                    # 流式复制
                    with open(fp, 'rb') as fr:
                        while True:
                            chunk = fr.read(1024 * 1024)
                            if not chunk:
                                break
                            fw.write(chunk)
            if delete_temp:
                for fp in uniq:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

        def splitmix64_numpy(x: np.ndarray) -> np.ndarray:
            # x: np.uint64
            x = np.uint64(x)
            x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(30))
            x = (x * np.uint64(0xbf58476d1ce4e5b9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(27))
            x = (x * np.uint64(0x94d049bb133111eb)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(31))
            return x

        def lanes_for_range(start: int, count: int, M: int, seed: int):
            # 返回每个 lane 在 [0,count) 的局部索引数组（np.ndarray）
            idx = (np.arange(start, start + count, dtype=np.uint64) ^ np.uint64(seed))
            h = splitmix64_numpy(idx)
            lanes = (h % np.uint64(M)).astype(np.int64)
            buckets = [np.where(lanes == m)[0] for m in range(M)]
            return buckets

        # 通道参数：可用环境变量覆盖，默认 M=32, seed=1337
        M = int(os.getenv("HAC_INTERLEAVE_M", "8"))
        SEED = int(os.getenv("HAC_INTERLEAVE_SEED", "1337"))

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0
        t_codec = 0

        torch.cuda.synchronize();
        t1 = time.time()
        print('Start encoding (interleaver) ...')

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N mask on anchors

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]  # [N, feat_dim]
        _grid_offsets = self._offset[mask_anchor]  # [N, n_offsets, 3]
        _scaling = self.get_scaling[mask_anchor]  # [N, 6]
        _mask_full = self.get_mask[mask_anchor]  # [N, ?, 1] 这里只使用前 n_offsets 维

        N = _anchor.shape[0]

        # 1) xyz 使用 GPCC 压缩（保持原逻辑）
        t_anchor_0 = time.time()
        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        means_strings = compress_gpcc(_anchor_int)
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
        bits_xyz = os.path.getsize(npz_path) * 8
        t_anchor += time.time() - t_anchor_0

        # 按 Morton 排序后重排所有属性
        _anchor = (_anchor_int * self.voxel_size)
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask_full = _mask_full[sorted_indices]

        # 保存边界
        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_total = 0
        bit_scaling_total = 0
        bit_offsets_total = 0
        bit_masks_total = 0

        # hash & masks 文件名（hash 单一文件；masks/feat/scaling/offsets 走 lane+step）
        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        # 每个 step 内做路由，按照 lane 写相应的子批文件
        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s + 1) * MAX_batch_size, N)
            count = N_end - N_start
            if count <= 0:
                continue

            # 计算该 step 的各 lane 局部索引（0..count-1）
            buckets = lanes_for_range(N_start, count, M, SEED)

            # 准备该 step 的切片
            anchor_step = _anchor[N_start:N_end]
            feat_step = _feat[N_start:N_end]
            scaling_step = _scaling[N_start:N_end]
            offsets_step = _grid_offsets[N_start:N_end]
            # 只编码前 n_offsets 维的 mask（与解码读取一致）
            masks_step = _mask_full[N_start:N_end, :self.n_offsets, :]

            # 遍历每个 lane
            for m in range(M):
                pos = buckets[m]
                B = int(pos.size)
                # 即使 B=0，我们也不强制创建空文件；解码时会根据 B 跳过
                if B == 0:
                    continue

                idx_local = torch.from_numpy(pos).to('cuda', dtype=torch.long)

                anchor_slice = anchor_step[idx_local]  # [B, 3]
                feat_slice = feat_step[idx_local]  # [B, feat_dim]
                scaling_slice = scaling_step[idx_local]  # [B, 6]
                offsets_slice = offsets_step[idx_local]  # [B, n_offsets, 3]
                masks_slice = masks_step[idx_local]  # [B, n_offsets, 1]

                # 路径
                masks_file = os.path.join(pre_path_name, f'masks_lane{m}_s{s}.b')
                scaling_file = os.path.join(pre_path_name, f'scaling_lane{m}_s{s}.b')
                offsets_file = os.path.join(pre_path_name, f'offsets_lane{m}_s{s}.b')

                # 量化超参（保持与原逻辑一致）
                Q_feat = 1.0
                Q_scaling = 0.001
                Q_offsets = 0.2

                # 计算上下文（每个 lane 子批独立）
                feat_context = self.calc_interp_feat(anchor_slice)
                mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = torch.split(
                    self.get_grid_mlp(feat_context),
                    split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim,
                                            6, 6, 3 * self.n_offsets, 3 * self.n_offsets,
                                            1, 1, 1],
                    dim=-1
                )

                # Q 的展开（形状严格对齐）
                Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])  # [B, feat_dim]
                Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)  # [B*6]
                Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)  # [B*3*n_offsets]
                Q_feat_t = Q_feat * (1 + torch.tanh(Q_feat_adj))  # [B, feat_dim]
                Q_scaling_t = Q_scaling * (1 + torch.tanh(Q_scaling_adj))  # [B*6]
                Q_offsets_t = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [B*3*n_offsets]

                # 1) 写 masks（每锚点 n_offsets 个）
                t0 = time.time()
                bit_masks = encoder(masks_slice, file_name=masks_file)
                t_mask += time.time() - t0
                bit_masks_total += bit_masks

                # 2) 写 feat（5 段因果，落到 feat_lane{m}_cc{cc}_s{s}.b）
                # 使用量化后的 feat 做因果上下文（保持原逻辑）
                feat_q = STE_multistep.apply(feat_slice, Q_feat_t, self._anchor_feat.mean())
                mean_scale_cat = torch.cat([mean, scale, prob], dim=-1)  # 供 deform MLP 使用
                scale = torch.clamp(scale, min=1e-9)

                for cc in range(5):
                    feat_file = os.path.join(pre_path_name, f'feat_lane{m}_cc{cc}_s{s}.b')
                    # 预测第二分支参数（因果）
                    mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat_q, mean_scale_cat, to_dec=cc)
                    scale_adj = torch.clamp(scale_adj, min=1e-9)
                    probs = torch.stack([prob[:, cc * 10:cc * 10 + 10], prob_adj], dim=-1)
                    probs = torch.softmax(probs, dim=-1)

                    feat_part = feat_q[:, cc * 10:cc * 10 + 10].contiguous().view(-1)
                    Q_feat_part = Q_feat_t[:, cc * 10:cc * 10 + 10].contiguous().view(-1)

                    t0 = time.time()
                    bit_feat = encoder_gaussian_mixed_chunk(
                        feat_part,
                        [mean[:, cc * 10:cc * 10 + 10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                        [scale[:, cc * 10:cc * 10 + 10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                        [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                        Q_feat_part,
                        file_name=feat_file,
                        chunk_size=500_000
                    )
                    t_codec += time.time() - t0
                    bit_feat_total += bit_feat

                # 3) 写 scaling（展平到 [B*6]）
                scaling_flat = scaling_slice.view(-1)
                scaling_flat_q = STE_multistep.apply(scaling_flat, Q_scaling_t, self.get_scaling.mean())
                t0 = time.time()
                bit_scaling = encoder_gaussian_chunk(
                    scaling_flat_q,
                    mean_scaling.contiguous().view(-1),
                    torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9),
                    Q_scaling_t,
                    file_name=scaling_file,
                    chunk_size=100_000
                )
                t_codec += time.time() - t0
                bit_scaling_total += bit_scaling

                # 4) 写 offsets（仅写 True 位）
                mask_flat = masks_slice.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
                offsets_flat = offsets_slice.view(-1, 3 * self.n_offsets).view(-1)
                offsets_flat_q = STE_multistep.apply(offsets_flat, Q_offsets_t, self._offset.mean())
                offsets_flat_q[~mask_flat] = 0.0
                t0 = time.time()
                bit_offsets = encoder_gaussian_chunk(
                    offsets_flat_q[mask_flat],
                    mean_offsets.contiguous().view(-1)[mask_flat],
                    torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)[mask_flat],
                    Q_offsets_t[mask_flat],
                    file_name=offsets_file,
                    chunk_size=100_000
                )
                t_codec += time.time() - t0
                bit_offsets_total += bit_offsets

                torch.cuda.empty_cache()

                # ---- group pack to .pak ----
                if int(os.getenv("HAC_PAK_ENABLE", "1")) > 0:
                    pak_path = os.path.join(pre_path_name, f'pak_lane{m}_s{s}.pak')
                    # 组内需要打包的基流（含 chunk）
                    files_to_pack = []
                    # masks
                    files_to_pack += _collect_group_files_for_base(masks_file)
                    # scaling / offsets
                    files_to_pack += _collect_group_files_for_base(scaling_file)
                    files_to_pack += _collect_group_files_for_base(offsets_file)
                    # feats 5 段
                    for cc in range(5):
                        feat_file = os.path.join(pre_path_name, f'feat_lane{m}_cc{cc}_s{s}.b')
                        files_to_pack += _collect_group_files_for_base(feat_file)
                    # 写 .pak 并按需删除临时 .b
                    _pack_group_pak(
                        pak_path,
                        files_to_pack,
                        delete_temp=(int(os.getenv("HAC_PAK_DELETE_TEMP", "1")) > 0)
                    )


        # 5) hash（保持单文件）
        t0 = time.time()
        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bit_hash = encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel() * 32
        t_hash += time.time() - t0

        # 写 meta
        meta = {
            "schema": "interleave_v1",
            "M": M,
            "seed": SEED,
            "steps": steps,
            "max_batch_size": MAX_batch_size,
            "order": "morton",
            "N": int(N)
        }
        # ========== 新增：保存 MLP/encoding 权重供解码使用 ==========
        mlp_ckpt_path = os.path.join(pre_path_name, 'mlp_ckpt.pth')
        try:
            self.save_mlp_checkpoints(mlp_ckpt_path)
            print(f"[encode] Saved MLP/encoding weights to {mlp_ckpt_path}")
        except Exception as e:
            print(f"[encode] Warning: save_mlp_checkpoints failed: {e}")
        # ========== 新增结束 ==========
        with open(os.path.join(pre_path_name, "interleave_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        torch.cuda.synchronize();
        t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        log_info = (
            f"\nEncoded sizes in MB: "
            f"anchor {round(bits_xyz / bit2MB_scale, 4)}, "
            f"feat {round(bit_feat_total / bit2MB_scale, 4)}, "
            f"scaling {round(bit_scaling_total / bit2MB_scale, 4)}, "
            f"offsets {round(bit_offsets_total / bit2MB_scale, 4)}, "
            f"hash {round(bit_hash / bit2MB_scale, 4)}, "
            f"masks {round(bit_masks_total / bit2MB_scale, 4)}, "
            f"MLPs {round(self.get_mlp_size()[0] / bit2MB_scale, 4)}, "
            f"Total {round((bits_xyz + bit_feat_total + bit_scaling_total + bit_offsets_total + bit_hash + bit_masks_total + self.get_mlp_size()[0]) / bit2MB_scale + 32 * 3 * 2 / bit2MB_scale, 4)}, "
            f"EncTime {round(t2 - t1, 4)}"
        )
        log_info_time = (
            f"\nEncoded time in s: "
            f"anchor {round(t_anchor, 4)}, "
            f"feat {round(t_feature, 4)}, "
            f"scaling {round(t_scaling, 4)}, "
            f"offsets {round(t_offset, 4)}, "
            f"hash {round(t_hash, 4)}, "
            f"masks {round(t_mask, 4)}, "
            f"Total {round((t2 - t1), 4)}"
        )
        return log_info + log_info_time

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):
        import os, json, glob, numpy as np, time, torch
        import torch.nn.functional as F

        def splitmix64_numpy(x: np.ndarray) -> np.ndarray:
            x = np.uint64(x)
            x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(30))
            x = (x * np.uint64(0xbf58476d1ce4e5b9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(27))
            x = (x * np.uint64(0x94d049bb133111eb)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            x ^= (x >> np.uint64(31))
            return x

        def lanes_for_range(start: int, count: int, M: int, seed: int):
            idx = (np.arange(start, start + count, dtype=np.uint64) ^ np.uint64(seed))
            h = splitmix64_numpy(idx)
            lanes = (h % np.uint64(M)).astype(np.int64)
            buckets = [np.where(lanes == m)[0] for m in range(M)]
            return buckets

        def chunks_present(base_path: str) -> bool:
            pref = base_path[:-2] if base_path.endswith('.b') else base_path
            gl = glob.glob(pref + '_*.b')
            return (len(gl) > 0) and any(x.endswith('_0.b') for x in gl)

        # ---- unpack helpers: group-level .pak ----
        def _unpack_group_pak(pak_path: str, dest_dir: str):
            """
            解出 .pak 中的文件到 dest_dir，下发 basename；返回写出的绝对路径 list。
            """
            written = []
            try:
                with open(pak_path, 'rb') as fr:
                    magic = fr.read(4)
                    if magic != b'PAK1':
                        return written
                    cnt = int.from_bytes(fr.read(4), 'little', signed=False)
                    for _ in range(cnt):
                        nl = int.from_bytes(fr.read(4), 'little', signed=False)
                        name = fr.read(nl).decode('utf-8', errors='ignore')
                        sz = int.from_bytes(fr.read(8), 'little', signed=False)
                        outp = os.path.join(dest_dir, name)
                        # 确保目录存在（这里是 pre_path_name 直下，通常已经存在）
                        with open(outp, 'wb') as fw:
                            remaining = sz
                            while remaining > 0:
                                blk = fr.read(min(1024 * 1024, remaining))
                                if not blk:
                                    break
                                fw.write(blk)
                                remaining -= len(blk)
                        written.append(outp)
            except Exception:
                # 解包失败视为未写任何文件
                for fp in written:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
                written = []
            return written

        def _cleanup_files(file_list: list):
            for fp in file_list:
                try:
                    os.remove(fp)
                except Exception:
                    pass

        torch.cuda.synchronize();
        t1 = time.time()
        print('Start decoding ...')
        # GNN 补洞开关（环境变量控制）
        use_gnn_inpaint = int(os.getenv("HAC_GNN_INPAINT", "1")) > 0
        gnn_only = int(os.getenv("HAC_GNN_ONLY", "0")) > 0

        meta_path = os.path.join(pre_path_name, "interleave_meta.json")
        if not os.path.exists(meta_path):
            raise RuntimeError("interleave_meta.json not found; decoding expects interleaver bitstreams.")

        # bounds
        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))

        # 1) anchors (GPCC)
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        data_dict = np.load(npz_path)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * voxel_size
        N = int(anchor_decoded.shape[0])

        # 2) hash
        t_hash = 0.0
        t0 = time.time()
        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder(N_hash, os.path.join(pre_path_name, 'hash.b'))
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)
        t_hash += time.time() - t0

        # interleave meta
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if not all(k in meta for k in ("M", "seed", "steps", "max_batch_size")):
            raise RuntimeError(f"interleave_meta.json missing fields.")
        M = int(meta["M"]);
        SEED = int(meta["seed"]);
        steps = int(meta["steps"]);
        MAX_bs = int(meta["max_batch_size"])

        def _load_gnn_weights_strict(inpaintor, path):
            import torch
            sd_loaded = torch.load(path, map_location="cpu")
            meta = None
            sd = sd_loaded
            if isinstance(sd_loaded, dict) and "state_dict" in sd_loaded:
                meta = {k: v for k, v in sd_loaded.items() if k != "state_dict"}
                sd = sd_loaded["state_dict"]
            res = inpaintor.load_state_dict(sd, strict=False)
            matched = 0
            for k, v in sd.items():
                if k in inpaintor.state_dict() and inpaintor.state_dict()[k].shape == v.shape:
                    matched += 1
            # 样例键，便于识别后端类型
            model_keys = list(inpaintor.state_dict().keys())[:6]
            ckpt_keys = list(sd.keys())[:6]
            print(f"[decode] model key sample: {model_keys}")
            print(f"[decode] ckpt  key sample: {ckpt_keys}")
            print(
                f"[decode] GNN load summary: matched={matched}, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            if meta:
                print(f"[decode] GNN ckpt meta: {meta}")
            if matched < max(8, len(inpaintor.state_dict()) // 4):
                raise RuntimeError("Loaded GNN weights mismatch too much. Train/eval backend or arch may differ.")

        # load MLP/encoding weights
        def _find_latest_training_mlp_ckpt(model_root: str):
            pc_dir = os.path.join(model_root, "point_cloud")
            if not os.path.isdir(pc_dir):
                return None
            latest_iter, latest_path = -1, None
            for name in os.listdir(pc_dir):
                if name.startswith("iteration_"):
                    try:
                        it = int(name.split("_", 1)[1])
                    except Exception:
                        continue
                    ck = os.path.join(pc_dir, name, "checkpoint.pth")
                    if os.path.isfile(ck) and it > latest_iter:
                        latest_iter, latest_path = it, ck
            return latest_path

        model_root = os.path.abspath(os.path.join(pre_path_name, os.pardir))
        bit_mlp = os.path.join(pre_path_name, "mlp_ckpt.pth")
        loaded_from = None
        if os.path.isfile(bit_mlp):
            try:
                self.load_mlp_checkpoints(bit_mlp);
                loaded_from = bit_mlp
                print(f"[decode] Loaded MLP/encoding weights from {bit_mlp}")
            except Exception as e:
                print(f"[decode] Failed to load {bit_mlp}: {e}")
        if loaded_from is None:
            latest_ck = _find_latest_training_mlp_ckpt(model_root)
            if latest_ck and os.path.isfile(latest_ck):
                self.load_mlp_checkpoints(latest_ck);
                loaded_from = latest_ck
                print(f"[decode] Loaded MLP/encoding weights from {latest_ck}")
            else:
                raise RuntimeError(f"[decode] No MLP checkpoint found.")

        # global buffers and flags
        feat_decoded = torch.zeros((N, self.feat_dim), device='cuda', dtype=torch.float32)
        scaling_decoded = torch.zeros((N, 6), device='cuda', dtype=torch.float32)
        offsets_decoded = torch.zeros((N, self.n_offsets, 3), device='cuda', dtype=torch.float32)
        masks_decoded = torch.zeros((N, self.n_offsets, 1), device='cuda', dtype=torch.float32)

        feat_cc_ok = [torch.zeros((N,), device='cuda', dtype=torch.bool) for _ in range(5)]
        scaling_ok = torch.zeros((N,), device='cuda', dtype=torch.bool)
        offsets_ok = torch.zeros((N,), device='cuda', dtype=torch.bool)

        missing_records = []

        # decode loop (step/lane/attr)
        for s in range(steps):
            N_start = s * MAX_bs
            N_end = min((s + 1) * MAX_bs, N)
            count = N_end - N_start
            if count <= 0: continue

            buckets = lanes_for_range(N_start, count, M, SEED)
            anchor_step = anchor_decoded[N_start:N_end]

            for m in range(M):
                pos = buckets[m];
                B = int(pos.size)
                if B == 0: continue
                idx_local = torch.from_numpy(pos).to('cuda', dtype=torch.long)
                idx_global = (torch.arange(N_start, N_end, device='cuda', dtype=torch.long))[idx_local]

                masks_file = os.path.join(pre_path_name, f'masks_lane{m}_s{s}.b')
                scaling_base = os.path.join(pre_path_name, f'scaling_lane{m}_s{s}.b')
                offsets_base = os.path.join(pre_path_name, f'offsets_lane{m}_s{s}.b')
                feat_bases = [os.path.join(pre_path_name, f'feat_lane{m}_cc{cc}_s{s}.b') for cc in range(5)]
                miss_list = []

                # ---- 若存在 .pak，先解包到当前 bitstreams 目录 ----
                unpacked_files = []
                unpacked = False
                if int(os.getenv("HAC_PAK_ENABLE", "1")) > 0:
                    pak_path = os.path.join(pre_path_name, f'pak_lane{m}_s{s}.pak')
                    if os.path.exists(pak_path):
                        unpacked_files = _unpack_group_pak(pak_path, pre_path_name)
                        unpacked = True
                    # 如果 .pak 不存在，后续各流读不到会自然进入 "missing" 分支（整组等效丢失）

                # masks
                if os.path.exists(masks_file):
                    try:
                        masks_m = decoder(B * self.n_offsets, masks_file)
                        if not masks_m.is_cuda: masks_m = masks_m.to('cuda')
                        masks_m = masks_m.to(torch.float32).view(-1, self.n_offsets, 1)
                        masks_decoded[idx_global] = masks_m
                    except Exception as e:
                        masks_decoded[idx_global] = 0.0;
                        miss_list.append('masks_runtime_error:' + str(e))
                else:
                    masks_decoded[idx_global] = 0.0;
                    miss_list.append(os.path.basename(masks_file))

                # context
                feat_context = self.calc_interp_feat(anchor_step[idx_local])
                mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = torch.split(
                    self.get_grid_mlp(feat_context),
                    split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim,
                                            6, 6, 3 * self.n_offsets, 3 * self.n_offsets,
                                            1, 1, 1],
                    dim=-1
                )
                Q_feat_rep = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
                Q_scaling_vec = (0.001 * (1 + torch.tanh(Q_scaling_adj.contiguous()
                                                         .repeat(1, mean_scaling.shape[-1]).view(-1))))
                Q_offsets_vec = (0.2 * (1 + torch.tanh(Q_offsets_adj.contiguous()
                                                       .repeat(1, mean_offsets.shape[-1]).view(-1))))

                # feat (5 cc)
                feat_m = torch.zeros((B, self.feat_dim), device='cuda', dtype=torch.float32)
                mean_scale_cat = torch.cat([mean, scale, prob], dim=-1)
                scale = torch.clamp(scale, min=1e-9)
                for cc in range(5):
                    fb = feat_bases[cc]
                    if not chunks_present(fb):
                        miss_list.append(os.path.basename(fb).replace('.b', '_*.b'));
                        continue
                    try:
                        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat_m, mean_scale_cat, to_dec=cc)
                        scale_adj = torch.clamp(scale_adj, min=1e-9)
                        probs = torch.stack([prob[:, cc * 10:cc * 10 + 10], prob_adj], dim=-1)
                        probs = torch.softmax(probs, dim=-1)
                        Q_feat_part = (1.0 * (1 + torch.tanh(Q_feat_rep[:, cc * 10:cc * 10 + 10]))).contiguous().view(
                            -1)
                        dec = decoder_gaussian_mixed_chunk(
                            [mean[:, cc * 10:cc * 10 + 10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                            [scale[:, cc * 10:cc * 10 + 10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                            [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                            Q_feat_part,
                            file_name=fb,
                            chunk_size=500_000
                        )
                        feat_m[:, cc * 10:cc * 10 + 10] = dec.view(B, 10)
                        feat_cc_ok[cc][idx_global] = True
                    except (FileNotFoundError, OSError) as e:
                        miss_list.append(f'feat_cc{cc}_runtime_error:{e}')
                feat_decoded[idx_global] = feat_m

                # scaling
                if chunks_present(scaling_base):
                    try:
                        scaling_dec = decoder_gaussian_chunk(
                            mean_scaling.contiguous().view(-1),
                            torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9),
                            Q_scaling_vec,
                            file_name=scaling_base,
                            chunk_size=100_000
                        )
                        scaling_decoded[idx_global] = scaling_dec.view(B, 6)
                        scaling_ok[idx_global] = True
                    except (FileNotFoundError, OSError) as e:
                        miss_list.append('scaling_runtime_error:' + str(e))
                else:
                    miss_list.append(os.path.basename(scaling_base) + '_*.b')

                # offsets（True位）
                if chunks_present(offsets_base):
                    try:
                        masks_m = masks_decoded[idx_global]
                        mask_flat = masks_m.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
                        off_vals = decoder_gaussian_chunk(
                            mean_offsets.contiguous().view(-1)[mask_flat],
                            torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)[mask_flat],
                            Q_offsets_vec[mask_flat],
                            file_name=offsets_base,
                            chunk_size=100_000
                        )
                        off_full = torch.zeros_like(mean_offsets.contiguous().view(-1))
                        off_full[mask_flat] = off_vals
                        offsets_decoded[idx_global] = off_full.view(B, 3 * self.n_offsets).view(B, self.n_offsets, 3)
                        offsets_ok[idx_global] = True
                    except (FileNotFoundError, OSError) as e:
                        miss_list.append('offsets_runtime_error:' + str(e))
                else:
                    miss_list.append(os.path.basename(offsets_base) + '_*.b')

                if miss_list:
                    missing_records.append({'step': int(s), 'lane': int(m), 'missing': miss_list})
                # ---- 若刚才解过包，已读完该组，按需清理解包出的临时 .b ----
                if unpacked and int(os.getenv("HAC_PAK_CLEAN", "1")) > 0:
                    _cleanup_files(unpacked_files)


        if missing_records:
            with open(os.path.join(pre_path_name, 'lost_report.json'), 'w') as f:
                json.dump(missing_records, f, indent=2)
        # ---------- GNN 补洞（整锚缺失优先在此恢复） ----------
        # ---------- GNN 补洞（整锚缺失优先在此恢复） ----------
        if use_gnn_inpaint:
            # 以“所有5段 feat + scaling + offsets 都成功解码”为已知锚点
            feat_ok_all = feat_cc_ok[0]
            for _cc in range(1, 5):
                feat_ok_all = feat_ok_all & feat_cc_ok[_cc]
            known_mask_anchor = feat_ok_all & scaling_ok & offsets_ok  # [N] bool
            num_missing = (~known_mask_anchor).sum().item()

            if num_missing > 0:
                print(f"[decode] GNN inpaint: missing anchors = {num_missing}")

                # 读取权重路径
                wpth = os.getenv("HAC_GNN_WEIGHTS", "").strip()
                if not (wpth and os.path.isfile(wpth)):
                    raise RuntimeError("HAC_GNN_WEIGHTS is not set or file not found; cannot run GNN inpaint safely.")

                # 加载权重（可能含 meta 或纯 state_dict）
                ck = torch.load(wpth, map_location="cpu")
                meta = None
                sd_loaded = ck
                if isinstance(ck, dict) and "state_dict" in ck:
                    meta = {k: v for k, v in ck.items() if k != "state_dict"}
                    sd_loaded = ck["state_dict"]

                # 构造两套配置：优先 CLI（或 env），失败则尝试 meta
                def _cfg_from_cli_and_env():
                    return {
                        "backend": os.getenv("HAC_GNN_BACKEND", "vanilla").lower(),
                        "k": int(os.getenv("HAC_GNN_K", "16")),
                        "k2": int(os.getenv("HAC_GNN_K2", os.getenv("HAC_GNN_K", "16"))),
                        "hidden": int(os.getenv("HAC_GNN_H", "128")),
                        "mp": int(os.getenv("HAC_GNN_MP", "1")),
                        "conv": os.getenv("HAC_GNN_CONV", "edge").lower(),
                        "heads": int(os.getenv("HAC_GNN_HEADS", "2")),
                    }

                def _cfg_from_meta(m):
                    if not m:
                        return None
                    return {
                        "backend": m.get("backend", "vanilla").lower(),
                        "k": int(m.get("k", 16)),
                        "k2": int(m.get("k2", m.get("k", 16))),
                        "hidden": int(m.get("hidden", 128)),
                        "mp": int(m.get("mp_iters", 1)),
                        "conv": m.get("conv", "edge").lower(),
                        "heads": int(m.get("heads", 2)),
                    }

                cfg_cli = _cfg_from_cli_and_env()
                cfg_meta = _cfg_from_meta(meta)

                # 构造器
                def _build_inpaintor(cfg):
                    print(f"[decode] GNN config: {cfg}")
                    if cfg["backend"] == "pyg":
                        from utils.pyg_inpaint import PYGInpaintor
                        return PYGInpaintor(
                            k=cfg["k"], hidden=cfg["hidden"], device='cuda',
                            mp_iters=cfg["mp"], conv_type=cfg["conv"], k2=cfg["k2"], heads=cfg["heads"]
                        )
                    else:
                        return GNNInpaintor(k=cfg["k"], hidden=cfg["hidden"], device='cuda')

                # 暖启动（触发懒构建）
                def _warmup_build(inpaintor):
                    print(f"[decode] warmup build: state_dict keys before={len(inpaintor.state_dict())}")
                    try:
                        idx_known = torch.where(known_mask_anchor)[0]
                        idx_miss = torch.where(~known_mask_anchor)[0]
                        if idx_known.numel() > 0 and idx_miss.numel() > 0:
                            m_small = min(8, idx_miss.numel())
                            k_small = min(256, idx_known.numel())
                            xq_small = anchor_decoded[idx_miss[:m_small]]
                            xk_small = anchor_decoded[idx_known[:k_small]]
                            feat_k_s = feat_decoded[idx_known[:k_small]]
                            sca_k_s = scaling_decoded[idx_known[:k_small]]
                            off_k_s = offsets_decoded[idx_known[:k_small]]
                            msk_k_s = masks_decoded[idx_known[:k_small]]
                            _ = inpaintor.forward_once(self, xq_small, xk_small, feat_k_s, sca_k_s, off_k_s, msk_k_s,
                                                       apply_conf=False)
                            print(f"[decode] warmup build done: state_dict keys after={len(inpaintor.state_dict())}")
                    except Exception as e:
                        print(f"[decode] warmup build failed (non-fatal): {e}")

                # 严格加载（带 DataParallel 前缀剥离 + 键样例打印）
                def _try_load(inpaintor, state_or_dict):
                    def _strip_module_prefix(d):
                        out = {}
                        for k, v in d.items():
                            if k.startswith("module."):
                                out[k[7:]] = v
                            else:
                                out[k] = v
                        return out

                    sd = state_or_dict
                    if isinstance(state_or_dict, dict) and "state_dict" in state_or_dict:
                        sd = state_or_dict["state_dict"]
                    sd = _strip_module_prefix(sd)

                    model_keys = list(inpaintor.state_dict().keys())
                    sd_keys = list(sd.keys())
                    print("[decode] model key sample:", model_keys[:6])
                    print("[decode] ckpt  key sample:", sd_keys[:6])

                    res = inpaintor.load_state_dict(sd, strict=False)
                    matched = 0
                    for k_name, v in sd.items():
                        if k_name in inpaintor.state_dict() and inpaintor.state_dict()[k_name].shape == v.shape:
                            matched += 1
                    print(
                        f"[decode] GNN load summary: matched={matched}, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
                    ok = matched >= max(8, len(inpaintor.state_dict()) // 4)
                    return ok, matched

                # 尝试 1：CLI 配置
                inpaintor = _build_inpaintor(cfg_cli)
                _warmup_build(inpaintor)
                ok, _ = _try_load(inpaintor, ck)

                # 尝试 2：若失败且存在 meta 且不同于 CLI，则用 meta 配置重建再试
                def _cfg_equal(a, b):
                    if b is None: return False
                    keys = ["backend", "k", "k2", "hidden", "mp", "conv", "heads"]
                    return all(str(a.get(k)) == str(b.get(k)) for k in keys)

                if not ok and cfg_meta and not _cfg_equal(cfg_cli, cfg_meta):
                    print("[decode] retry load with weight meta config ...")
                    inpaintor = _build_inpaintor(cfg_meta)
                    _warmup_build(inpaintor)
                    ok, _ = _try_load(inpaintor, ck)

                if not ok:
                    print("[decode] WARN: GNN weights mismatch; enable safe interpolation fallback.")
                    os.environ["HAC_GNN_SAFE"] = "1"

                inpaintor.eval()
                _load_gnn_weights_strict(inpaintor, wpth)
                print(f"[decode] Loaded GNN weights: {wpth}")
                if os.getenv("HAC_GNN_DEBUG", "0") == "1":
                    _dump_gnn_runtime_cfg()

                # 可选场景内自适应（默认关闭）
                fit_steps = int(os.getenv("HAC_GNN_FIT_STEPS", "0"))
                if fit_steps > 0:
                    try:
                        inpaintor.fit(
                            gm=self,
                            x_all=anchor_decoded,
                            known_mask=known_mask_anchor,
                            feat=feat_decoded,
                            scale=scaling_decoded,
                            offsets=offsets_decoded,
                            mask=masks_decoded,
                            steps=fit_steps,
                            lr=float(os.getenv("HAC_GNN_LR", "1e-3")),
                            k=int(os.getenv("HAC_GNN_K", str(cfg_cli["k"]))),
                            max_known=int(os.getenv("HAC_GNN_MAX_KNOWN", "20000")),
                            batch_q=int(os.getenv("HAC_GNN_BQ", "1024"))
                        )
                    except Exception as e:
                        print(f"[decode] GNN fit skipped: {e}")

                # 预测并写回缺失锚点
                # 预测
                # 预测
                feat_hat, scale_hat, off_hat, mask_hat = inpaintor.predict(
                    gm=self, x_all=anchor_decoded, known_mask=known_mask_anchor,
                    feat=feat_decoded, scale=scaling_decoded, offsets=offsets_decoded, mask=masks_decoded
                )
                miss = ~known_mask_anchor

                # SAFE 守卫：SAFE=1 时完全不写回（保持 Drop-noGNN 的解码结果）
                _safe = int(os.getenv("HAC_GNN_SAFE", "0"))
                if _safe == 1:
                    print("[decode] SAFE=1: skip write-back; keep Drop-noGNN decode untouched")
                else:
                    # 正常写回
                    feat_decoded[miss] = feat_hat
                    scaling_decoded[miss] = scale_hat
                    offsets_decoded[miss] = off_hat
                    # 仅取前 n_offsets 个通道
                    masks_decoded[miss] = mask_hat[:, :self.n_offsets, :]

                    # 可选 smoothing（保持你原逻辑；SAFE 下不会执行）
                    if os.getenv("HAC_GNN_SMOOTH", "0") == "1":
                        try:
                            smoother = GNNInpaintor(device='cuda')
                            alpha = float(os.getenv("HAC_GNN_SMOOTH_ALPHA", "0.7"))
                            k_s = int(os.getenv("HAC_GNN_SMOOTH_K", "12"))
                            feat_decoded, scaling_decoded, offsets_decoded, masks_decoded = smoother.smooth_missing(
                                anchor_decoded, known_mask_anchor,
                                feat_decoded, scaling_decoded, offsets_decoded, masks_decoded,
                                x_known=anchor_decoded[known_mask_anchor], k=k_s, alpha=alpha
                            )
                            print(f"[decode] smoothing applied: k={k_s}, alpha={alpha}")
                        except Exception as e:
                            print(f"[decode] smoothing skipped: {e}")

                    # 仅在非 SAFE 时把这些位置标记为“已恢复”
                    for _cc in range(5):
                        feat_cc_ok[_cc][miss] = True
                    scaling_ok[miss] = True
                    offsets_ok[miss] = True

                print(f"[decode] GNN inpaint filled {miss.sum().item()} anchors.")

        # ---------- GNN 补洞结束 ----------

        # ---------- GNN 补洞结束 ----------


        # === EC：属性感知权重 + 迭代细化 + 可选岭回归 ===
        feat_ctx_all = self.calc_interp_feat(anchor_decoded)
        mean_all, scale_all, prob_all, mean_scal_all, scale_scal_all, mean_offs_all, scale_offs_all, \
            Q_feat_adj_all, Q_scal_adj_all, Q_offs_adj_all = torch.split(
            self.get_grid_mlp(feat_ctx_all),
            split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim,
                                    6, 6, 3 * self.n_offsets, 3 * self.n_offsets,
                                    1, 1, 1],
            dim=-1
        )
        scale_all = torch.clamp(scale_all, min=1e-9)

        # 超参
        K = int(os.getenv("HAC_EC_K", "8"))
        L = int(os.getenv("HAC_EC_L", "128"))
        Bm = int(os.getenv("HAC_EC_MB", "1024"))
        AvC = int(os.getenv("HAC_EC_AVC", "8192"))
        alpha = float(os.getenv("HAC_EC_ALPHA", "1.0"))
        beta = float(os.getenv("HAC_EC_BETA", "2.0"))
        gamma = float(os.getenv("HAC_EC_GAMMA", "1.0"))
        delta = float(os.getenv("HAC_EC_DELTA", "1.5"))
        conf_ec = float(os.getenv("HAC_EC_CONF_EC", "0.5"))
        iters = int(os.getenv("HAC_EC_ITERS", "2"))
        ridge_on = int(os.getenv("HAC_EC_RIDGE", "0")) > 0
        ridge_lambda = float(os.getenv("HAC_EC_RIDGE_LAMBDA", "1e-3"))
        ridge_blend = float(os.getenv("HAC_EC_RIDGE_BLEND", "0.5"))
        K = max(1, K);
        L = max(K, L)

        pos_all = anchor_decoded
        fh_all = feat_ctx_all
        fh_all_n = F.normalize(fh_all, dim=1)

        def space_topL(miss_idx_slice, avail_idx):
            pos_miss = pos_all[miss_idx_slice]
            pos_av = pos_all[avail_idx]
            B = miss_idx_slice.numel()
            Ma = avail_idx.numel()
            topd = torch.full((B, L), float('inf'), device='cuda')
            topi = torch.full((B, L), -1, dtype=torch.long, device='cuda')
            for as_ in range(0, Ma, AvC):
                ae = min(as_ + AvC, Ma)
                d = torch.cdist(pos_miss, pos_av[as_:ae])
                d_all = torch.cat([topd, d], dim=1)
                i_block = torch.arange(as_, ae, device='cuda', dtype=torch.long)
                i_all = torch.cat([topi, i_block.unsqueeze(0).expand(B, -1)], dim=1)
                newd, newk = torch.topk(d_all, k=L, largest=False)
                newi = torch.gather(i_all, 1, newk)
                topd, topi = newd, newi
                del d, d_all, i_block, i_all, newd, newk, newi
            return topd, topi

        def build_signatures(fea_cur):
            """
            构造属性签名：feat各段的|残差|均值(5维)、scaling残差均值(1)、offset残差均值(1)
            返回：
              feat_res_means: [N,5]
              scal_res_mean:  [N,1]
              off_res_mean:   [N,1]
            """
            # feat 残差均值 [N,5]
            feat_res_means = torch.zeros((N, 5), device='cuda', dtype=torch.float32)
            mean_scale_cat_all = torch.cat([mean_all, scale_all, prob_all], dim=-1)
            for cc in range(5):
                mean_adj_all, scale_adj_all, prob_adj_all = self.get_deform_mlp.forward(
                    fea_cur, mean_scale_cat_all, to_dec=cc
                )
                probs_cc = torch.stack([prob_all[:, cc * 10:cc * 10 + 10], prob_adj_all], dim=-1)
                probs_cc = torch.softmax(probs_cc, dim=-1)
                mu_cc = probs_cc[..., 0] * mean_all[:, cc * 10:cc * 10 + 10] + probs_cc[..., 1] * mean_adj_all  # [N,10]
                feat_res_means[:, cc] = (feat_decoded[:, cc * 10:cc * 10 + 10] - mu_cc).abs().mean(dim=1)

            # scaling 残差均值 [N,1]
            scal_res_mean = (scaling_decoded - mean_scal_all).abs().mean(dim=1, keepdim=True)  # [N,1]

            # offsets 残差均值（仅 True 位）→ [N,1]
            off_pred = mean_offs_all.view(N, self.n_offsets, 3)  # [N,Koff,3]
            off_res = (offsets_decoded - off_pred).abs()  # [N,Koff,3]
            mt = masks_decoded.to(torch.bool).expand_as(off_res)  # [N,Koff,3]
            off_res_masked = torch.where(mt, off_res, torch.zeros_like(off_res))
            # 先把分母/分子都降到 [N]，再 unsqueeze 回 [N,1]
            denom = mt.to(torch.float32).sum(dim=(1, 2)).clamp_min(1.0)  # [N]
            num = off_res_masked.sum(dim=(1, 2))  # [N]
            off_res_mean = (num / denom).unsqueeze(1)  # [N,1]

            return feat_res_means, scal_res_mean, off_res_mean

        def knn_weights_with_signature(miss_idx_slice, avail_idx, topd, topi, fh_miss_n,
                                       sigma_scalar_all, conf_scalar_all,
                                       sig_ref, delta):
            """
            在 knn_weights 基础上增加“属性签名相似度”sig_sim^delta
            sig_ref: [N,Ds] 的签名向量（不同属性/cc用不同的sig_ref）
            """
            B, Lc = topd.shape
            nn_idx_L = avail_idx[topi]  # [B,L]
            spatial_w = (1.0 / torch.sqrt(torch.clamp(topd, min=1e-9))) ** alpha

            fh_nei = fh_all_n[nn_idx_L.view(-1)].view(B, Lc, -1)
            sim = (fh_nei * fh_miss_n.unsqueeze(1)).sum(dim=-1).clamp(-1, 1)  # [B,L]
            sim_w = ((sim + 1.0) * 0.5) ** beta

            sigma_nei = sigma_scalar_all[nn_idx_L]  # [B,L]
            sigma_w = (1.0 / torch.clamp(sigma_nei, min=1e-6)) ** gamma

            conf_nei = conf_scalar_all[nn_idx_L]  # [B,L]

            # 签名相似度（cosine）
            sig_miss = sig_ref[miss_idx_slice]  # [B,Ds]
            sig_nei = sig_ref[nn_idx_L.view(-1)].view(B, Lc, -1)  # [B,L,Ds]
            sig_miss_n = F.normalize(sig_miss + 1e-9, dim=1)
            sig_nei_n = F.normalize(sig_nei + 1e-9, dim=2)
            sig_sim = (sig_nei_n * sig_miss_n.unsqueeze(1)).sum(dim=-1).clamp(0, 1)  # [B,L]
            sig_w = sig_sim ** delta

            comb = spatial_w * sim_w * sigma_w * conf_nei * sig_w  # [B,L]
            combK, idxK = torch.topk(comb, k=min(K, Lc), largest=True)
            nn_idx_K = nn_idx_L.gather(1, idxK)
            w = combK / torch.clamp(combK.sum(dim=1, keepdim=True), min=1e-9)
            return w, nn_idx_K

        # 迭代细化
        iters = max(0, iters)
        if (not gnn_only) and iters > 0:
            for it in range(iters):
                # 构造“属性签名”基于当前特征（用现有的 feat_decoded 作为上下文）
                feat_res_means, scal_res_mean, off_res_mean = build_signatures(feat_decoded)

                # feat 因果逐段
                fea_est = feat_decoded.clone()
                mean_scale_cat_all = torch.cat([mean_all, scale_all, prob_all], dim=-1)

                for cc in range(5):
                    miss_mask_cc = ~feat_cc_ok[cc]
                    miss_idx_cc = miss_mask_cc.nonzero(as_tuple=False).view(-1)

                    mean_adj_all, scale_adj_all, prob_adj_all = self.get_deform_mlp.forward(
                        fea_est, mean_scale_cat_all, to_dec=cc
                    )
                    scale_adj_all = torch.clamp(scale_adj_all, min=1e-9)
                    probs_cc = torch.stack([prob_all[:, cc * 10:cc * 10 + 10], prob_adj_all], dim=-1)
                    probs_cc = torch.softmax(probs_cc, dim=-1)
                    mu_cc_all = probs_cc[..., 0] * mean_all[:, cc * 10:cc * 10 + 10] + probs_cc[
                        ..., 1] * mean_adj_all  # [N,10]

                    sigma_cc_scalar = 0.5 * (
                                scale_all[:, cc * 10:cc * 10 + 10].mean(dim=1) + scale_adj_all.mean(dim=1))  # [N]
                    conf_cc = torch.where(feat_cc_ok[cc], torch.ones_like(sigma_cc_scalar),
                                          torch.full_like(sigma_cc_scalar, conf_ec))
                    avail_idx_cc = feat_cc_ok[cc].nonzero(as_tuple=False).view(-1)

                    if miss_idx_cc.numel() == 0:
                        fea_est[:, cc * 10:cc * 10 + 10] = feat_decoded[:, cc * 10:cc * 10 + 10]
                        continue

                    if avail_idx_cc.numel() == 0:
                        fea_est[miss_idx_cc, cc * 10:cc * 10 + 10] = mu_cc_all[miss_idx_cc]
                    else:
                        # 构造本cc的“签名向量”：其他4段feat残差均值+scal残差均值+off残差均值 → [N,6]
                        idx_others = [k for k in range(5) if k != cc]
                        sig_ref = torch.cat([feat_res_means[:, idx_others], scal_res_mean, off_res_mean], dim=1)  # [N,6]

                        topd, topi = space_topL(miss_idx_cc, avail_idx_cc)
                        fh_miss_n = fh_all_n[miss_idx_cc]

                        w, nn_idx_K = knn_weights_with_signature(
                            miss_idx_cc, avail_idx_cc, topd, topi, fh_miss_n,
                            sigma_cc_scalar, conf_cc, sig_ref, delta
                        )

                        Bcur = miss_idx_cc.numel()
                        pred_ref = mu_cc_all
                        actual_ref = torch.zeros_like(mu_cc_all)
                        actual_ref[avail_idx_cc] = feat_decoded[avail_idx_cc, cc * 10:cc * 10 + 10]

                        neigh_act = actual_ref[nn_idx_K.view(-1)].view(Bcur, -1, 10)
                        neigh_mu = pred_ref[nn_idx_K.view(-1)].view(Bcur, -1, 10)
                        r = neigh_act - neigh_mu
                        r_hat = (w.unsqueeze(-1) * r).sum(dim=1)
                        rec_cc = pred_ref[miss_idx_cc] + r_hat

                        # 可选：岭回归微调
                        if ridge_on:
                            # 输入特征：其他段feat(40)+scaling(6)+offset强度(1)
                            with torch.no_grad():
                                feat_others = torch.cat([feat_decoded[:, :cc * 10], feat_decoded[:, (cc + 1) * 10:]],
                                                        dim=-1)  # [N,40]
                                offsets_mag = torch.norm(offsets_decoded.view(N, -1), dim=1, keepdim=True)  # [N,1]
                                X_all = torch.cat([feat_others, scaling_decoded, offsets_mag], dim=-1)  # [N,47]
                                Y_all = feat_decoded[:, cc * 10:cc * 10 + 10]  # [N,10]
                            # 仍然用 nn_idx_K 的邻居做每点小规模 ridge
                            X_nei = X_all[nn_idx_K.view(-1)].view(Bcur, -1, X_all.shape[1])  # [B,K,Dx]
                            Y_nei = Y_all[nn_idx_K.view(-1)].view(Bcur, -1, 10)  # [B,K,10]
                            X_miss = X_all[miss_idx_cc]  # [B,Dx]
                            Xt = X_nei.transpose(1, 2)
                            XtX = torch.matmul(Xt, X_nei)  # [B,Dx,Dx]
                            lamI = ridge_lambda * torch.eye(X_all.shape[1], device='cuda').unsqueeze(0).expand(Bcur, -1, -1)
                            A = XtX + lamI
                            XtY = torch.matmul(Xt, Y_nei)  # [B,Dx,10]
                            try:
                                W = torch.linalg.solve(A, XtY)
                            except RuntimeError:
                                W = torch.linalg.lstsq(A, XtY).solution
                            Y_hat = torch.matmul(X_miss.unsqueeze(1), W).squeeze(1)  # [B,10]
                            rec_cc = ridge_blend * Y_hat + (1.0 - ridge_blend) * rec_cc

                        fea_est[miss_idx_cc, cc * 10:cc * 10 + 10] = rec_cc

                feat_decoded = fea_est

                # scaling
                miss_scal_idx = (~scaling_ok).nonzero(as_tuple=False).view(-1)
                if miss_scal_idx.numel() > 0:
                    avail_scal_idx = scaling_ok.nonzero(as_tuple=False).view(-1)
                    pred_scal = mean_scal_all
                    if avail_scal_idx.numel() == 0:
                        scaling_decoded[miss_scal_idx] = pred_scal[miss_scal_idx]
                    else:
                        # 签名：5段feat残差均值+off残差均值 → [N,6]
                        sig_ref = torch.cat([feat_res_means, off_res_mean], dim=1)
                        topd, topi = space_topL(miss_scal_idx, avail_scal_idx)
                        fh_miss_n = fh_all_n[miss_scal_idx]
                        sigma_scal_scalar = scale_scal_all.mean(dim=1)
                        conf_scal = torch.where(scaling_ok, torch.ones_like(sigma_scal_scalar),
                                                torch.full_like(sigma_scal_scalar, conf_ec))
                        w, nn_idx_K = knn_weights_with_signature(
                            miss_scal_idx, avail_scal_idx, topd, topi, fh_miss_n,
                            sigma_scal_scalar, conf_scal, sig_ref, delta
                        )
                        Bcur = miss_scal_idx.numel()
                        neigh_act = scaling_decoded[nn_idx_K.view(-1)].view(Bcur, -1, 6)
                        neigh_mu = pred_scal[nn_idx_K.view(-1)].view(Bcur, -1, 6)
                        r = neigh_act - neigh_mu
                        r_hat = (w.unsqueeze(-1) * r).sum(dim=1)
                        scaling_decoded[miss_scal_idx] = pred_scal[miss_scal_idx] + r_hat

                # offsets（仅 True 位）
                miss_off_idx = (~offsets_ok).nonzero(as_tuple=False).view(-1)
                if miss_off_idx.numel() > 0:
                    avail_off_idx = offsets_ok.nonzero(as_tuple=False).view(-1)
                    pred_offs = mean_offs_all.view(N, self.n_offsets, 3)
                    mask_true = masks_decoded.to(torch.bool)
                    if avail_off_idx.numel() == 0:
                        rec = pred_offs[miss_off_idx]
                        mt = mask_true[miss_off_idx].to(rec.dtype).expand_as(rec)
                        offsets_decoded[miss_off_idx] = rec * mt
                    else:
                        # 签名：5段feat残差均值+scal残差均值 → [N,6]
                        sig_ref = torch.cat([feat_res_means, scal_res_mean], dim=1)
                        topd, topi = space_topL(miss_off_idx, avail_off_idx)
                        fh_miss_n = fh_all_n[miss_off_idx]
                        sigma_off_scalar = scale_offs_all.view(N, -1).mean(dim=1)
                        conf_off = torch.where(offsets_ok, torch.ones_like(sigma_off_scalar),
                                               torch.full_like(sigma_off_scalar, conf_ec))
                        w, nn_idx_K = knn_weights_with_signature(
                            miss_off_idx, avail_off_idx, topd, topi, fh_miss_n,
                            sigma_off_scalar, conf_off, sig_ref, delta
                        )
                        Bcur = miss_off_idx.numel()
                        neigh_act = offsets_decoded[nn_idx_K.view(-1)].view(Bcur, -1, self.n_offsets, 3)
                        neigh_mu = pred_offs[nn_idx_K.view(-1)].view(Bcur, -1, self.n_offsets, 3)
                        r = neigh_act - neigh_mu
                        r_hat = (w.view(Bcur, -1, 1, 1) * r).sum(dim=1)
                        rec = pred_offs[miss_off_idx] + r_hat
                        mt = mask_true[miss_off_idx].to(rec.dtype).expand_as(rec)
                        offsets_decoded[miss_off_idx] = rec * mt

        # 写回
        _anchor = anchor_decoded
        _anchor_feat = feat_decoded
        _offset = offsets_decoded
        _scaling = scaling_decoded
        _mask = torch.zeros((N, self.n_offsets + 1, 1), device='cuda', dtype=torch.float32)
        _mask[:, :self.n_offsets, :] = masks_decoded

        print('Replacing parameters (with EC if needed)...')
        self._anchor_feat = nn.Parameter(_anchor_feat)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)

        # rotation/opacity 尺寸对齐
        Nnew = _anchor.shape[0]
        if self._rotation is None or self._rotation.shape[0] != Nnew:
            rot = torch.zeros((Nnew, 4), device='cuda', dtype=torch.float32);
            rot[:, 0] = 1.0
            self._rotation = nn.Parameter(rot)
        if self._opacity is None or self._opacity.shape[0] != Nnew:
            self._opacity = nn.Parameter(
                inverse_sigmoid(0.1 * torch.ones((Nnew, 1), device='cuda', dtype=torch.float32)))
        self.max_radii2D = torch.zeros((Nnew,), device='cuda', dtype=torch.float32)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D + len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D:len_3D + len_2D * 2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D * 2:len_3D + len_2D * 3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        torch.cuda.synchronize();
        t2 = time.time()
        log_info = f"\nDecTime {round(t2 - t1, 4)}"
        log_info_time = f"\nDecoded time in s: hash {round(t_hash, 4)}, Total {round(t2 - t1, 4)}"
        return log_info + log_info_time

    # # ========== 辅助函数 ==========
    # def _save_pack_meta(self, pre_path_name, scheme, group_count, N):
    #     """保存打包元数据"""
    #     meta = {
    #         'scheme': scheme,
    #         'group_count': group_count,
    #         'N': N,
    #     }
    #     torch.save(meta, os.path.join(pre_path_name, 'pack_meta.pkl'))
    #
    # def _load_pack_meta(self, pre_path_name):
    #     """加载打包元数据"""
    #     return torch.load(os.path.join(pre_path_name, 'pack_meta.pkl'))
    #
    # def _build_interleave_groups(self, N, G, device='cuda'):
    #     """
    #     构建交错分组索引
    #     锚点n被分配到组 n % G
    #
    #     返回: list of tensor，每个tensor包含一个组的锚点索引
    #     """
    #     groups = [[] for _ in range(G)]
    #     for n in range(N):
    #         groups[n % G].append(n)
    #
    #     return [torch.tensor(g, dtype=torch.long, device=device) for g in groups]



