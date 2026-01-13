# 文件: test_ec.py
# (完整替换)

import os
import sys
import shutil
import torch
import random
import glob
import re
from argparse import ArgumentParser

# 导入 train.py 中的必要函数和参数类
try:
    import train
    from train import render_sets, evaluate, get_logger
    from arguments import ModelParams, PipelineParams, OptimizationParams
except ImportError as e:
    print(f"Error: Failed to import from train.py. Make sure this script is in the same directory.")
    print(e)
    sys.exit(1)

# --- 步骤 1: 配置 (请确保与您的 run_shell_db.py 匹配) ---

# ==========================================================
# === 在这里修改丢包率 ===
PACKET_LOSS_RATE = 0.30  # <-- 30% 的锚点属性将被随机置空和插值
# ==========================================================

LMBDA = 0.004
SCENE = 'drjohnson'
DATASET_PATH = f'data/db/{SCENE}'
MODEL_PATH = f'outputs/blending/{SCENE}/{LMBDA}'

BITSTREAM_PATH = os.path.join(MODEL_PATH, 'bitstreams')

# --- 步骤 2: 检查文件 (不再删除) ---
print(f"---[EC TEST]---: Checking for required files in: {BITSTREAM_PATH}")
if not os.path.exists(BITSTREAM_PATH):
    print(f"Error: Bitstream path not found: {BITSTREAM_PATH}")
    print("Please run the training/compression (run_shell_db.py) at least once.")
    sys.exit(1)
print(f"---[EC TEST]---: Bitstream files found. Proceeding with in-memory loss simulation.")


# --- 步骤 3: 运行评估 (调用解码、模拟丢失、修复和渲染) ---
print(f"---[EC TEST]---: Running evaluation with {PACKET_LOSS_RATE*100}% in-memory loss...")

sys.argv = [
    'test_ec.py',
    f'--model_path={MODEL_PATH}',
    f'--source_path={DATASET_PATH}',
    '--gpu=0'  # 假设使用 GPU 0
]

parser = ArgumentParser(description="Error Concealment Test Script")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
# [cite_start]# [cite: 606-610]
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument('--warmup', action='store_true', default=False)
parser.add_argument('--use_wandb', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
parser.add_argument("--gpu", type=str, default = '-1')
parser.add_argument("--log2", type=int, default = 13)
parser.add_argument("--log2_2D", type=int, default = 15)
parser.add_argument("--n_features", type=int, default = 4)
parser.add_argument("--lmbda", type=float, default = 0.001)

# === 新增：添加 EC 测试参数 ===
parser.add_argument("--ec_loss_rate", type=float, default=0.0)
# === 修改结束 ===

args = parser.parse_args()

# === 新增：将我们配置的丢包率设置到参数中 ===
args.ec_loss_rate = PACKET_LOSS_RATE
# === 修改结束 ===

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger = get_logger(args.model_path)

logger.info(f"---[EC TEST]---: Starting evaluation with {args.ec_loss_rate*100}% simulated IN-MEMORY loss.")
logger.info(f"---[EC TEST]---: Model Path: {args.model_path}")
logger.info(f"---[EC TEST]---: Source Path: {args.source_path}")

try:
    x_bound_min = torch.load(os.path.join(BITSTREAM_PATH, 'x_bound_min.pkl'))
    x_bound_max = torch.load(os.path.join(BITSTREAM_PATH, 'x_bound_max.pkl'))
except FileNotFoundError as e:
    logger.error(f"---[EC TEST]---: CRITICAL ERROR: Failed to load pkl bounds from {BITSTREAM_PATH}")
    logger.error("These are critical data and cannot be lost. Aborting.")
    sys.exit(1)

dataset_args = lp.extract(args)
pipeline_args = pp.extract(args)

train.run_codec = True # 确保解码被调用

# 调用修改过的 render_sets，并传入新的 ec_loss_rate 参数
visible_count = render_sets(args, dataset_args, 30_000, pipeline_args,
                            skip_train=True, skip_test=False,
                            logger=logger,
                            x_bound_min=x_bound_min,
                            x_bound_max=x_bound_max,
                            ec_loss_rate=args.ec_loss_rate) # <-- 将丢包率传进去

if visible_count:
    logger.info(f"\n---[EC TEST]---: Evaluating repaired model performance (after {args.ec_loss_rate*100}% loss)...")
    evaluate(args.model_path, visible_count=visible_count, logger=logger)
    logger.info("\n---[EC TEST]---: Evaluation complete.")
else:
    logger.error("\n---[EC TEST]---: Rendering failed, cannot evaluate.")