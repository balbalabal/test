import os
import torch
import shutil
import random
import logging
import faiss  # 确保 faiss 被导入
from types import SimpleNamespace
from argparse import ArgumentParser
from tqdm import tqdm

# 假设这些文件都在你的HAC-plus-main目录中
from scene import Scene, GaussianModel
from gaussian_renderer import render, prefilter_voxel
from train import get_logger, readImages, evaluate, render_set
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from scene.dataset_readers import sceneLoadTypeCallbacks


# (我们必须从 gaussian_model.py 复制这个函数，因为它在解码时被调用)
def find_k_nearest_neighbors(target_indices, all_anchor_pos, k, gpu_index, faiss_initialized):
    B = len(target_indices)
    N = len(all_anchor_pos)
    device = all_anchor_pos.device
    neighbor_indices = torch.zeros((B, k), dtype=torch.long, device=device)
    neighbor_padding_mask = torch.ones((B, k), dtype=torch.bool, device=device)
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
    return neighbor_indices, neighbor_padding_mask


def simulate_packet_loss(source_dir, temp_dir, loss_rate, logger):
    """
    准备一个用于解码的临时码流目录，并模拟丢包。
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源码流目录未找到: {source_dir}")

    # 创建一个临时的、干净的目标目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    shutil.copytree(source_dir, temp_dir)

    if loss_rate == 0.0:
        logger.info("--- Packet Loss Rate is 0%. Evaluating lossless performance. ---")
        return 0

    files_deleted = 0
    total_files = 0
    for filename in os.listdir(temp_dir):
        # 你的要求：只删除锚点属性文件
        if filename.startswith('feat_') or filename.startswith('scaling_') or filename.startswith('offsets_'):
            total_files += 1
            if random.random() < loss_rate:
                os.remove(os.path.join(temp_dir, filename))
                files_deleted += 1

    logger.info(
        f"--- Simulated Packet Loss: Deleted {files_deleted} of {total_files} attribute files ({loss_rate * 100:.1f}%) ---")
    return files_deleted


def main():
    parser = ArgumentParser(description="Clean Decoder-Only EC Evaluation Script")

    # 1. 必需的路径
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="Path to scene data (e.g., data/db/playroom)")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to save evaluation renders/metrics")
    parser.add_argument("--bitstream_path", type=str, required=True,
                        help="Path to the SOURCE Stage 1 (HAC++) bitstream directory")
    parser.add_argument("--hac_checkpoint", type=str, required=True,
                        help="Path to the Stage 1 (HAC++) MLP weights (.pth)")
    parser.add_argument("--ec_checkpoint", type=str, required=True,
                        help="Path to the trained Stage 2 EC Transformer weights (.pth)")

    # 2. 评估参数
    parser.add_argument("--packet_loss_rate", type=float, default=0.1, help="Simulated packet loss rate (0.0 to 1.0)")

    # 3. 模型形状参数 (必须与 Stage 1 检查点匹配)
    parser.add_argument("--feat_dim", type=int, default=50)
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--fh_dim", type=int, default=96)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--update_depth", type=int, default=3)
    parser.add_argument("--update_init_factor", type=int, default=16)
    parser.add_argument("--update_hierachy_factor", type=int, default=4)
    parser.add_argument("--use_feat_bank", type=bool, default=False)

    # 4. EC Transformer 形状参数 (必须与 ec_checkpoint 匹配)
    parser.add_argument("--ec_model_dim", type=int, default=128)
    parser.add_argument("--ec_nhead", type=int, default=4)
    parser.add_argument("--ec_num_encoder_layers", type=int, default=3)
    parser.add_argument("--ec_dim_feedforward", type=int, default=512)
    parser.add_argument("--ec_max_neighbors", type=int, default=5)
    parser.add_argument("--ec_dropout", type=float, default=0.1)

    # 5. Scene/Camera 加载所需的参数
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--data_device", type=str, default="cuda")
    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--sh_degree", type=int, default=3)  # (来自 ModelParams)

    args = parser.parse_args()

    # --- 1. 设置 ---
    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    logger.info(f"Starting new decoder-only evaluation. Args: {args}")
    safe_state(False)

    # --- 2. 加载模型 (不加载 .ply) ---
    logger.info("Initializing GaussianModel...")
    is_synthetic_nerf = os.path.exists(os.path.join(args.source_path, "transforms_train.json"))

    gaussians = GaussianModel(
        fh_dim=args.fh_dim,
        feat_dim=args.feat_dim,
        n_offsets=args.n_offsets,
        voxel_size=args.voxel_size,
        update_depth=args.update_depth,
        update_init_factor=args.update_init_factor,
        update_hierachy_factor=args.update_hierachy_factor,
        use_feat_bank=args.use_feat_bank,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        is_synthetic_nerf=is_synthetic_nerf,
        sh_degree=args.sh_degree,  # 确保传递这个

        # 传递 EC 参数
        ec_model_dim=args.ec_model_dim,
        ec_nhead=args.ec_nhead,
        ec_num_encoder_layers=args.ec_num_encoder_layers,
        ec_dim_feedforward=args.ec_dim_feedforward,
        ec_max_neighbors=args.ec_max_neighbors,
        ec_dropout=args.ec_dropout
    )
    gaussians.to("cuda")
    logger.info("GaussianModel created.")

    # --- 3. 加载权重 ---
    logger.info(f"Loading Stage 1 MLPs from: {args.hac_checkpoint}")
    gaussians.load_mlp_checkpoints(args.hac_checkpoint)

    logger.info(f"Loading EC Transformer weights from: {args.ec_checkpoint}")
    ec_weights = torch.load(args.ec_checkpoint, map_location="cuda")
    gaussians.ec_transformer.load_state_dict(ec_weights)
    logger.info("All weights loaded successfully.")

    gaussians.eval()

    # --- 4. 准备一个临时的、有损的码流目录 ---
    temp_bitstream_dir = os.path.join(args.model_path, "temp_bitstream_for_eval")
    simulate_packet_loss(args.bitstream_path, temp_bitstream_dir, args.packet_loss_rate, logger)

    # --- 5. 解码与修复 [你的要求] ---
    logger.info(f"Starting decoding from: {temp_bitstream_dir}")
    with torch.no_grad():
        log_info = gaussians.conduct_decoding(pre_path_name=temp_bitstream_dir)
    logger.info(log_info)

    # --- 6. 手动设置边界 (在解码之后) ---
    logger.info("Manually updating anchor bounds from decoded anchors...")
    if gaussians._anchor is not None and gaussians._anchor.shape[0] > 0:
        gaussians.x_bound_min = torch.min(gaussians._anchor, dim=0)[0].detach()
        gaussians.x_bound_max = torch.max(gaussians._anchor, dim=0)[0].detach()
        logger.info(f"Manually set bounds: min={gaussians.x_bound_min.tolist()}, max={gaussians.x_bound_max.tolist()}")
    else:
        logger.error("Decoded anchors are empty! Cannot set bounds.")
        raise ValueError("Decoder failed to produce any anchors from the bitstream.")

    # --- 7. 加载相机 (不加载 .ply) ---
    logger.info("Loading cameras...")
    # (我们必须创建一个模拟的 Scene 对象来加载相机)
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1, args)

    # --- 8. 渲染与评估 ---
    logger.info("Starting rendering of decoded/concealed model...")
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # (创建一个模拟的 pipe 对象，用于渲染)
    pipe_mock = SimpleNamespace(debug=False, compute_cov3D_python=False)

    # (调用 render_set, 注意 loaded_iter 设为 "decoded")
    t_list, visible_count = render_set(args.model_path, "test", "decoded", test_cameras, gaussians, pipe_mock,
                                       background)

    logger.info("Starting evaluation of rendered images...")
    evaluate(args.model_path, visible_count=visible_count, logger=logger)

    # --- 9. 清理 ---
    shutil.rmtree(temp_bitstream_dir)
    logger.info(f"Temporary bitstream directory deleted: {temp_bitstream_dir}")
    logger.info("--- Evaluation Finished ---")


if __name__ == "__main__":
    main()