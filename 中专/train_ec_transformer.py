import os
import torch
import faiss
import random
import logging
from argparse import ArgumentParser
from tqdm import tqdm
from torch.optim import Adam
from types import SimpleNamespace  # <--- [!! 新增修复 !!]
from scene import Scene, GaussianModel
from gaussian_renderer import prefilter_voxel
from utils.loss_utils import l1_loss
from utils.encodings import STE_multistep  # (需要从 train_ec.py 复制这个函数)

# 假设 ec_transformer.py 和这个脚本在同一个目录
from ec_transformer import ErrorConcealmentTransformer


# --- [!!] 关键：从 train_ec.py 复制 Faiss KNN 函数 ---
# (你需要从 train_ec.py 复制 find_k_nearest_neighbors 函数到这里)
# def find_k_nearest_neighbors(...):
#     ...

# --- 日志记录 ---
# --- ↓↓↓ [!! 新增 !!] 添加 Faiss GPU KNN 函数 ↓↓↓ ---
def find_k_nearest_neighbors(target_indices, all_anchor_pos, k, gpu_index, faiss_initialized):
    """
    使用 Faiss-GPU 查找 K 近邻。
    """
    B = len(target_indices)
    N = len(all_anchor_pos)
    device = all_anchor_pos.device

    neighbor_indices = torch.zeros((B, k), dtype=torch.long, device=device)
    neighbor_padding_mask = torch.ones((B, k), dtype=torch.bool, device=device) # True = Padding

    if B == 0 or N <= 1 or not faiss_initialized:
        return neighbor_indices, neighbor_padding_mask

    torch.cuda.synchronize(device=device)
    target_pos_cpu_np = all_anchor_pos[target_indices].detach().cpu().numpy().astype('float32')

    actual_k_to_search = min(k + 1, N)
    try:
        D, I = gpu_index.search(target_pos_cpu_np, actual_k_to_search)
        torch.cuda.synchronize(device=device)
    except Exception as e:
         print(f"Faiss search failed: {e}. Returning padding.")
         return neighbor_indices, neighbor_padding_mask

    for i in range(B):
         found_indices = I[i]
         self_mask = (found_indices == target_indices[i].item())
         valid_indices = found_indices[~self_mask]
         num_found = len(valid_indices)
         num_to_copy = min(num_found, k)
         if num_to_copy > 0:
              neighbor_indices[i, :num_to_copy] = torch.from_numpy(valid_indices[:num_to_copy]).to(device)
              neighbor_padding_mask[i, :num_to_copy] = False

    if B > 0 and not (~neighbor_padding_mask).any():
         neighbor_padding_mask.fill_(True)

    return neighbor_indices, neighbor_padding_mask
# --- ↑↑↑ [!! 新增 !!] 新增结束 ↑↑↑ ---
def get_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(path, "train_ec.log"))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def train(args):
    """
    全新的、干净的 EC Transformer 训练函数
    """
    # --- 1. 设置 ---
    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    logger.info(f"Starting new EC Transformer training. Args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载第一阶段 (HAC++) 模型 ---
    logger.info(f"Loading Stage 1 PLY: {args.load_stage1_ply}")
    logger.info(f"Loading Stage 1 Checkpoint (MLPs): {args.load_stage1_checkpoint}")

    # (这些参数需要与你的HAC++模型匹配)
    dataset_params = {
        'feat_dim': 50,
        'n_offsets': 10,
        'voxel_size': 0.16,
        'update_depth': 4,
        'update_init_factor': 1.0,
        'update_hierachy_factor': 2.0,
        'use_feat_bank': False,
        # 'fh_dim': 32  # (!! 确保这个值与你的模型一致 !!)
        # 'model_path': args.model_path,  # <--- [!! 新增修复 !!]
        # 'source_path': args.source_path  # <--- [!! 新增修复 !!]
    }

    # 初始化 GaussianModel 并 *包含* EC Transformer
    gaussians = GaussianModel(
        **dataset_params,
        fh_dim=args.fh_dim,
        # --- ↓↓↓ [!! 新增修复 3 !!] ---
        n_features_per_level=args.n_features_per_level,
        log2_hashmap_size=args.log2_hashmap_size,
        log2_hashmap_size_2D=args.log2_hashmap_size_2D,
        # --- ↑↑↑ [!! 修复结束 3 !!] ---
        ec_model_dim=args.ec_model_dim,
        ec_nhead=args.ec_nhead,
        ec_num_encoder_layers=args.ec_num_encoder_layers,
        ec_dim_feedforward=args.ec_dim_feedforward,
        ec_max_neighbors=args.ec_max_neighbors,
        ec_dropout=args.ec_dropout
    ).to(device)

    # 关键修复：加载 .ply ("身体")
    scene = Scene(args, gaussians, ply_path=args.load_stage1_ply) # <--- [!! 修复 !!]
    gaussians.update_anchor_bound()
    # 加载 .pth ("大脑")
    gaussians.load_mlp_checkpoints(args.load_stage1_checkpoint)

    logger.info("Stage 1 model (.ply and .pth) loaded successfully.")

    # --- 3. 冻结 HAC++，只训练 Transformer ---
    total_params = 0
    trainable_params = 0
    for n, p in gaussians.named_parameters():
        total_params += p.numel()
        if 'ec_transformer' not in n:
            p.requires_grad = False
        else:
            trainable_params += p.numel()

    logger.info(f"Model loaded. Total params: {total_params}. Trainable (EC Transformer) params: {trainable_params}")

    # --- 4. 初始化 Faiss (用于 KNN) ---
    try:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(3)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        all_anchor_pos_np = gaussians.get_anchor.detach().cpu().numpy().astype('float32')
        gpu_index.add(all_anchor_pos_np)
        faiss_initialized = True
        logger.info(f"Faiss GPU index populated with {gpu_index.ntotal} anchors.")
    except Exception as e:
        logger.error(f"Failed to initialize Faiss GPU index: {e}. KNN will fail.")
        faiss_initialized = False
        gpu_index = None

    # --- 5. 设置优化器 ---
    optimizer = Adam(gaussians.ec_transformer.parameters(), lr=args.lr)  #

    # --- 6. 训练循环 ---
    # (这部分逻辑复制自 train_ec.py L338-L481, 因为它是正确的训练逻辑)

    # (这些是 train_ec.py L360-L362 中硬编码的量化基数)
    Q_feat_base = 1.0
    Q_scaling_base = 0.001
    Q_offsets_base = 0.2

    bg_color = [0, 0, 0]  # (假设背景为黑)
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe_mock = SimpleNamespace(debug=False, compute_cov3D_python=False) # <--- [!! 再次修复 !!]
    viewpoint_stack = scene.getTrainCameras().copy()

    progress_bar = tqdm(range(1, args.iterations + 1), desc="EC Training")
    for iteration in progress_bar:

        # --- A. 获取一批可见锚点 ---
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        with torch.no_grad():
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe_mock, background)

        visible_anchors_indices = torch.where(voxel_visible_mask)[0]
        B_full = len(visible_anchors_indices)

        if B_full == 0:
            continue

        # 从可见锚点中采样一个批次
        if B_full > args.batch_size:
            sampled_indices = torch.randperm(B_full, device=device)[:args.batch_size]
            target_indices = visible_anchors_indices[sampled_indices]
        else:
            target_indices = visible_anchors_indices

        B = len(target_indices)

        # --- B. 获取“真实”的量化属性 (Ground Truth) ---
        with torch.no_grad():
            # (这部分逻辑来自 train_ec.py L346-L373)
            target_pos = gaussians.get_anchor[target_indices]
            target_fh = gaussians.calc_interp_feat(target_pos)
            grid_mlp_output = gaussians.get_grid_mlp(target_fh)

            split_sections = [
                gaussians.feat_dim, gaussians.feat_dim, gaussians.feat_dim,
                6, 6,
                3 * gaussians.n_offsets, 3 * gaussians.n_offsets,
                1, 1, 1
            ]
            _, _, _, _, _, _, _, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = torch.split(
                grid_mlp_output, split_sections, dim=-1
            )

            Q_feat = Q_feat_base * (1 + torch.tanh(Q_feat_adj.expand(-1, gaussians.feat_dim)))
            Q_scaling = Q_scaling_base * (1 + torch.tanh(Q_scaling_adj.expand(-1, 6)))
            Q_offsets = Q_offsets_base * (1 + torch.tanh(Q_offsets_adj.expand(-1, 3 * gaussians.n_offsets)))

            real_feat = gaussians._anchor_feat[target_indices]
            real_scaling_log = gaussians._scaling[target_indices]
            real_offset = gaussians._offset[target_indices]

            feat_q = STE_multistep.apply(real_feat, Q_feat)
            scaling_q = STE_multistep.apply(torch.exp(real_scaling_log), Q_scaling)
            offset_q = STE_multistep.apply(real_offset, Q_offsets.view(B, gaussians.n_offsets, 3))

            real_attrs_q = torch.cat([feat_q, scaling_q, offset_q.reshape(B, -1)], dim=1).detach()

        # --- C. 模拟丢包 ---
        dynamic_p_loss = random.uniform(0.05, 0.5)  # 隨機丟失 5% 到 50%
        target_missing_mask = torch.rand(B, 3, device=device) < dynamic_p_loss

        if not target_missing_mask.any():
            continue  # (如果没有丢包，就跳过，节省计算)

        target_partial_attrs_q = real_attrs_q.clone()

        # --- D. 查找邻居 (KNN) ---
        k_neighbors = gaussians.ec_transformer.max_neighbors

        # (你需要在这里调用 find_k_nearest_neighbors 函数)
        neighbor_indices, neighbor_padding_mask = find_k_nearest_neighbors(
            target_indices, gaussians.get_anchor.detach(), k=k_neighbors,
            gpu_index=gpu_index,
            faiss_initialized=faiss_initialized
        )

        # # [!!] 占位符 (如果 KNN 还没复制过来)
        # neighbor_indices = torch.zeros(B, k_neighbors, dtype=torch.long, device=device)
        # neighbor_padding_mask = torch.ones(B, k_neighbors, dtype=torch.bool, device=device)

        # --- E. 获取邻居的量化属性 (Ground Truth) ---
        with torch.no_grad():
            # (这部分逻辑来自 train_ec.py L431-L457)
            neighbor_pos = gaussians.get_anchor[neighbor_indices]
            neighbor_fh = gaussians.calc_interp_feat(neighbor_pos.view(-1, 3)).view(B, k_neighbors, -1)
            neighbor_grid_mlp_output = gaussians.get_grid_mlp(neighbor_fh.view(-1, gaussians.fh_dim))

            _, _, _, _, _, _, _, n_Q_feat_adj, n_Q_scaling_adj, n_Q_offsets_adj = torch.split(
                neighbor_grid_mlp_output, split_sections, dim=-1
            )
            n_Q_feat = Q_feat_base * (1 + torch.tanh(n_Q_feat_adj.expand(-1, gaussians.feat_dim)))
            n_Q_scaling = Q_scaling_base * (1 + torch.tanh(n_Q_scaling_adj.expand(-1, 6)))
            n_Q_offsets = Q_offsets_base * (1 + torch.tanh(n_Q_offsets_adj.expand(-1, 3 * gaussians.n_offsets)))

            neighbor_feat_orig = gaussians._anchor_feat[neighbor_indices]
            neighbor_scaling_log_orig = gaussians._scaling[neighbor_indices]
            neighbor_offset_orig = gaussians._offset[neighbor_indices]

            neighbor_feat_q = STE_multistep.apply(neighbor_feat_orig.view(-1, gaussians.feat_dim), n_Q_feat).view(B,
                                                                                                                  k_neighbors,
                                                                                                                  -1)
            neighbor_scaling_q = STE_multistep.apply(torch.exp(neighbor_scaling_log_orig.view(-1, 6)),
                                                     n_Q_scaling).view(B, k_neighbors, -1)
            neighbor_offset_q = STE_multistep.apply(neighbor_offset_orig.view(-1, 3), n_Q_offsets.view(-1, 3)).view(B,
                                                                                                                    k_neighbors,
                                                                                                                    gaussians.n_offsets,
                                                                                                                    3)

            neighbor_attrs_q = torch.cat(
                [neighbor_feat_q, neighbor_scaling_q, neighbor_offset_q.reshape(B, k_neighbors, -1)],
                dim=2
            ).detach()

        # --- F. 调用 Transformer 并计算损失 ---
        optimizer.zero_grad()

        predicted_attrs = gaussians.ec_transformer(  #
            target_pos.detach(),
            target_fh.detach(),
            target_partial_attrs_q,
            target_missing_mask,
            neighbor_pos.detach(),
            neighbor_attrs_q,
            neighbor_padding_mask
        )

        ec_loss_mask = torch.zeros_like(real_attrs_q, dtype=torch.bool)
        if target_missing_mask[:, 0].any(): ec_loss_mask[target_missing_mask[:, 0], :gaussians.feat_dim] = True
        if target_missing_mask[:, 1].any(): ec_loss_mask[target_missing_mask[:, 1],
                                            gaussians.feat_dim: gaussians.feat_dim + 6] = True
        if target_missing_mask[:, 2].any(): ec_loss_mask[target_missing_mask[:, 2], gaussians.feat_dim + 6:] = True

        ec_loss = torch.tensor(0.0, device=device)
        if ec_loss_mask.any():
            ec_loss = l1_loss(predicted_attrs[ec_loss_mask], real_attrs_q[ec_loss_mask])

        # --- G. 反向传播 ---
        if ec_loss.item() > 0:
            total_loss = args.lambda_ec * ec_loss
            total_loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                progress_bar.set_postfix({"EC Loss": f"{ec_loss.item():.6f}"})

    # --- 7. 保存训练好的 EC Transformer ---
    final_model_path = os.path.join(args.model_path, "ec_transformer.pth")
    torch.save(gaussians.ec_transformer.state_dict(), final_model_path)  #
    logger.info(f"Training complete. EC Transformer weights saved to: {final_model_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean Stage 2 EC Transformer Training Script")

    # --- 路径参数 ---
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="Path to the scene data (e.g., data/db/playroom)")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Path to save the trained EC Transformer (e.g., outputs/ec_model)")
    parser.add_argument("--load_stage1_checkpoint", type=str, required=True,
                        help="Path to Stage 1 .pth checkpoint (MLPs)")
    parser.add_argument("--load_stage1_ply", type=str, required=True, help="Path to Stage 1 .ply file (Anchors)")

    # --- 训练参数 ---
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64, help="Number of visible anchors per training step")
    parser.add_argument("--p_loss", type=float, default=0.1, help="Simulated packet loss rate (0.0 to 1.0)")
    parser.add_argument("--lambda_ec", type=float, default=1.0, help="Weight for the EC loss")
    # --- [!! 在这里添加 !!] ---
    parser.add_argument("--fh_dim", type=int, default=48, help="Feature hierarchy dimension (必须与HAC++模型匹配)")
    # --- [!! 添加结束 !!] ---
    # --- EC Transformer 模型参数 ---
    parser.add_argument("--ec_model_dim", type=int, default=256)
    parser.add_argument("--ec_nhead", type=int, default=8)
    parser.add_argument("--ec_num_encoder_layers", type=int, default=6)
    parser.add_argument("--ec_dim_feedforward", type=int, default=1024)
    parser.add_argument("--ec_max_neighbors", type=int, default=10)
    parser.add_argument("--ec_dropout", type=float, default=0.1)

    # `Scene` 构造函数需要这些额外的参数
    parser.add_argument("--images", type=str, default="images", help="Name of the images folder (required by Scene)")
    parser.add_argument("--eval", action="store_true", default=False, help="Set to eval mode (required by Scene)")
    parser.add_argument("--lod", type=int, default=0, help="Level of detail (required by Scene)")
    # --- ↑↑↑ [!! 修复结束 !!] ---
    # --- ↓↓↓ [!! 新增修复 !!] ---
    parser.add_argument("--resolution", type=int, default=1, help="Resolution scale (required by camera_utils)")
    # --- ↑↑↑ [!! 修复结束 !!] ---
    # --- ↓↓↓ [!! 新增修复 2 !!] ---
    parser.add_argument("--data_device", type=str, default="cuda",
                        help="Device for data loading (required by camera_utils)")
    # --- ↑↑↑ [!! 修复结束 2 !!] ---
    # Hashgrid parameters (必须与 Stage 1 匹配)
    parser.add_argument("--n_features_per_level", type=int, default=4,
                        help="Features per hashgrid level (from Stage 1)")
    parser.add_argument("--log2_hashmap_size", type=int, default=13, help="Log2 hashmap size 3D (from Stage 1)")
    parser.add_argument("--log2_hashmap_size_2D", type=int, default=15, help="Log2 hashmap size 2D (from Stage 1)")
    # --- ↑↑↑ [!! 修复结束 3 !!] ---
    args = parser.parse_args()

    # (你需要从你的 `train_ec.py` 脚本中复制 `find_k_nearest_neighbors` 和 `STE_multistep` 函数
    #  到这个文件的顶部，否则它无法运行)
    if not callable(globals().get('find_k_nearest_neighbors')):
        print("=" * 80)
        print("ERROR: 'find_k_nearest_neighbors' function not found.")
        print("Please copy it from 'train_ec.py' into this script.")
        print("=" * 80)
    elif not callable(globals().get('STE_multistep')):
        print("=" * 80)
        print("ERROR: 'STE_multistep' function not found.")
        print("Please copy 'from utils.encodings import STE_multistep' and ensure it exists.")
        print("=" * 80)
    else:
        train(args)