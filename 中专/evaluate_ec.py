import os
import torch
import shutil
import random
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams, PipelineParams  # 从 __init__.py 导入
from scene import Scene, GaussianModel
from gaussian_renderer import render, prefilter_voxel
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from train import get_logger, readImages, evaluate, render_sets , render_set # 从 train.py 导入评估函数
from tqdm import tqdm
from utils.general_utils import safe_state
import sys

# 确保能导入 ec_transformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_evaluation(args_param, dataset, opt, pipe, logger):
    logger.info("--- Starting Error Concealment Evaluation ---")

    # 1. 加载模型 (HAC++ 冻结 + EC Transformer 激活)
    logger.info("Loading GaussianModel with EC parameters...")
    is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
    gaussians = GaussianModel(
        fh_dim=args_param.fh_dim,
        feat_dim=dataset.feat_dim,
        n_offsets=dataset.n_offsets,
        voxel_size=dataset.voxel_size,
        update_depth=dataset.update_depth,
        update_init_factor=dataset.update_init_factor,
        update_hierachy_factor=dataset.update_hierachy_factor,
        use_feat_bank=dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        is_synthetic_nerf=is_synthetic_nerf,

        # 传递 EC 参数 (从命令行获取)
        ec_model_dim=getattr(args_param, 'ec_model_dim'),  # <--- [!! 修复 !!]
        ec_nhead=getattr(args_param, 'ec_nhead'),  # <--- [!! 修复 !!]
        ec_num_encoder_layers=getattr(args_param, 'ec_num_encoder_layers'),  # <--- [!! 修复 !!]
        ec_dim_feedforward=getattr(args_param, 'ec_dim_feedforward'),  # <--- [!! 修复 !!]
        ec_max_neighbors=getattr(args_param, 'ec_max_neighbors'),  # <--- [!! 修复 !!]1
        ec_dropout=getattr(args_param, 'ec_dropout')
    )
    # 只需要一个最小的场景来加载相机，模型权重将从检查点加载
    # scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    # 加载阶段 1 (HAC++) 权重
    # load_chkpnt = getattr(opt, 'load_stage1_checkpoint')
    # if not load_chkpnt or not os.path.exists(load_chkpnt):
    #     raise ValueError(f"Stage 1 checkpoint not found: {load_chkpnt}")
    # logger.info(f"Loading Stage 1 weights from: {load_chkpnt}")
    # gaussians.load_mlp_checkpoints(load_chkpnt)  # 只加载 MLPs

    # 加载阶段 2 (EC Transformer) 权重
    # (我们假设 --start_checkpoint 指向 阶段 2 的最终模型)
    # if not args_param.start_checkpoint:
    #     raise ValueError("Please provide the final Stage 2 checkpoint path via --start_checkpoint")
    #
    # logger.info(f"Loading final Stage 2 (full model) weights from: {args_param.start_checkpoint}")
    # (model_params, first_iter) = torch.load(args_param.start_checkpoint)
    # gaussians.restore(model_params, opt)  # 加载完整模型 (锚点, 优化器状态等)
    #
    # logger.info(f"Model loaded successfully from iteration {first_iter}.")
    # 1. 检查所有必需的文件路径
    if not args_param.load_stage1_ply:
        raise ValueError("Please provide the Stage 1 .ply path via --load_stage1_ply")
    if not args_param.start_checkpoint:
        raise ValueError("Please provide the Stage 1 .pth (HAC++ MLPs) path via --start_checkpoint")
    if not args_param.ec_checkpoint:
        raise ValueError("Please provide the trained EC weights path via --ec_checkpoint")

    # 2. 创建 Scene 并加载 .ply ("身体")
    logger.info(f"Loading Stage 1 PLY from: {args_param.load_stage1_ply}")
    scene = Scene(args_param, gaussians, ply_path=args_param.load_stage1_ply, shuffle=False)

    # 3. 设置边界 (修复 AssertionError)
    logger.info("Updating anchor bounds from .ply file...")
    gaussians.update_anchor_bound()  #

    # 4. 加载 Stage 1 MLPs ("大脑")
    logger.info(f"Loading Stage 1 MLPs from: {args_param.start_checkpoint}")
    gaussians.load_mlp_checkpoints(args_param.start_checkpoint)  #

    # 5. 加载 Stage 2 EC Transformer (注入纠错模块)
    logger.info(f"Loading EC Transformer weights from: {args_param.ec_checkpoint}")
    ec_weights = torch.load(args_param.ec_checkpoint, map_location="cuda")
    gaussians.ec_transformer.load_state_dict(ec_weights)  #

    logger.info("Model (HAC++ MLPs + EC Transformer) loaded successfully.")
    gaussians.eval()  # 设为评估模式

    # 2. 运行编码 (生成比特流)
    bit_stream_path = os.path.join(dataset.model_path, 'bitstreams')
    # 检查码流文件是否存在
    if not os.path.exists(bit_stream_path):
        raise FileNotFoundError(f"码流目录未找到! 请手动将HAC++码流复制到: {bit_stream_path}")

    logger.info(f"--- SKIPPING ENCODING ---")
    logger.info(f"Using existing Stage 1 bitstream from: {bit_stream_path}")

    # 3. 模拟丢包 (!! 关键 !!)
    # (从命令行获取丢包率，默认 10%)
    packet_loss_rate = getattr(opt, 'packet_loss_rate', 0.0)
    files_deleted = 0
    total_files = 0

    if packet_loss_rate > 0:
        for filename in os.listdir(bit_stream_path):
            # 我们只删除属性文件 (feat, scaling, offsets)
            if filename.startswith('feat_') or filename.startswith('scaling_') or filename.startswith('offsets_'):
                total_files += 1
                if random.random() < packet_loss_rate:
                    os.remove(os.path.join(bit_stream_path, filename))
                    files_deleted += 1
        logger.info(
            f"--- Simulated Packet Loss: Deleted {files_deleted} of {total_files} attribute files ({packet_loss_rate * 100:.1f}%) ---")
    else:
        logger.info("--- Packet Loss Rate is 0%. Evaluating lossless performance. ---")

    # 4. 运行解码 (使用修改后的 conduct_decoding)
    logger.info("Starting decoding (with Error Concealment)...")
    with torch.no_grad():
        log_info = gaussians.conduct_decoding(pre_path_name=bit_stream_path)
    logger.info(log_info)
    logger.info("Manually updating anchor bounds from decoded anchors...")
    if gaussians._anchor is not None and gaussians._anchor.shape[0] > 0:
        # 手动计算边界
        gaussians.x_bound_min = torch.min(gaussians._anchor, dim=0)[0]
        gaussians.x_bound_max = torch.max(gaussians._anchor, dim=0)[0]
        logger.info(f"Manually set bounds: min={gaussians.x_bound_min.tolist()}, max={gaussians.x_bound_max.tolist()}")
    else:
        logger.error("Decoded anchors are empty! Cannot set bounds. Rendering will fail.")
        # 如果你在这里看到错误，说明你的码流文件是空的或已损坏
        raise ValueError("Decoded anchor data is empty.")
    # gaussians.update_anchor_bound()
    # 5. 运行渲染和评估
    logger.info("Starting rendering of decoded model...")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 调用 train.py 中的函数
    visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe,
                                background)

    logger.info("Starting evaluation of rendered images...")
    evaluate(dataset.model_path, visible_count=visible_count, logger=logger)

    logger.info("--- Error Concealment Evaluation Finished ---")


if __name__ == "__main__":
    # (从 train.py 复制 __main__ 块 L1020-L1048)
    parser = ArgumentParser(description="Evaluation script for EC Transformer")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # --- ↓↓↓ 添加评估特定参数 ↓↓↓ ---
    parser.add_argument("--packet_loss_rate", type=float, default=0.1, help="Simulated packet loss rate (0.0 to 1.0)")
    # --- ↑↑↑ 添加结束 ↑↑↑ ---

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])  # (评估脚本不需要)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])  # (评估脚本不需要)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)  # [!! 关键 !!]
    # --- ↓↓↓ [!! 新增修复 !!] ---
    # parser.add_argument("--load_stage1_ply", type=str, default=None, help="Path to Stage 1 .ply file (Anchors)")
    # parser.add_argument("--ec_checkpoint", type=str, default=None,
    #                     help="Path to trained EC Transformer weights (ec_transformer.pth)")
    # --- ↑↑↑ [!! 修复结束 !!] ---
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--lmbda", type=float, default=0.001)

    # --- ↓↓↓ [!! 新增 !!] 添加 EC 和 场景(Scene) 所需的所有参数 ↓↓↓ ---

    # 1. 新增的路径参数
    parser.add_argument("--load_stage1_ply", type=str, default=None, help="Path to Stage 1 .ply file (Anchors)")
    parser.add_argument("--ec_checkpoint", type=str, default=None,
                        help="Path to trained EC Transformer weights (ec_transformer.pth)")

    # 2. 模型形状参数 (必须与 Stage 1 匹配)
    parser.add_argument("--fh_dim", type=int, default=48, help="Feature hierarchy dimension (HAC++: 48)")
    # parser.add_argument("--n_features_per_level", type=int, default=4,
    #                     help="Features per hashgrid level (from Stage 1)")
    # parser.add_argument("--log2_hashmap_size", type=int, default=13, help="Log2 hashmap size 3D (from Stage 1)")
    # parser.add_argument("--log2_hashmap_size_2D", type=int, default=15, help="Log2 hashmap size 2D (from Stage 1)")

    # 3. Scene 加载所需的参数
    # parser.add_argument("--images", type=str, default="images", help="Name of the images folder (required by Scene)")
    # parser.add_argument("--eval", action="store_true", default=True, help="Set to eval mode (required by Scene)")
    # parser.add_argument("--lod", type=int, default=0, help="Level of detail (required by Scene)")
    # parser.add_argument("--resolution", type=int, default=1, help="Resolution scale (required by camera_utils)")
    # parser.add_argument("--data_device", type=str, default="cuda",
    #                     help="Device for data loading (required by camera_utils)")

    # 4. EC Transformer 形状参数 (必须与训练时一致)
    # parser.add_argument("--ec_model_dim", type=int, default=128)
    # parser.add_argument("--ec_nhead", type=int, default=4)
    # parser.add_argument("--ec_num_encoder_layers", type=int, default=3)
    # parser.add_argument("--ec_dim_feedforward", type=int, default=512)
    # parser.add_argument("--ec_max_neighbors", type=int, default=5)
    # parser.add_argument("--ec_dropout", type=float, default=0.1)
    # --- ↑↑↑ [!! 新增结束 !!] ---
    args = parser.parse_args(sys.argv[1:])

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)
    logger.info(f'args: {args}')
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    safe_state(args.quiet)

    # 运行评估
    run_evaluation(args, lp.extract(args), op.extract(args), pp.extract(args), logger)