import os

# --- 1. 配置基本参数 ---
scene = 'playroom'
dataset_name = 'db'
data_root = 'data'
output_root = 'outputs'
stage1_lmbda = 0.004

# --- 2. [!! 关键 !!] 指定阶段 1 的检查点路径 ---
# (!! 请确保这指向你阶段 1 训练生成的 chkpnt30000.pth 文件 !!)
stage1_folder = f"{stage1_lmbda}_stage1_hac_only"
stage1_checkpoint = os.path.join(output_root, dataset_name, scene, stage1_folder, "chkpnt30000.pth")

# --- 3. [!! 关键 !!] 定义 Transformer 参数 (使用简化版) ---
ec_model_dim = 128
ec_nhead = 4
ec_num_encoder_layers = 3 # (Encoder-Only 架构)
ec_dim_feedforward = 512
ec_max_neighbors = 5
ec_dropout = 0.1

# --- 4. 阶段 2 训练参数 ---
ec_iterations = 50_000 # (为 Transformer 训练 50k 次)
ec_transformer_lr_init = 0.001
ec_transformer_lr_final = 0.00001
ec_transformer_lr_max_steps = ec_iterations

cuda_device = 0

# --- 5. 构造参数 ---
data_path = os.path.join(data_root, dataset_name, scene)
output_path = os.path.join(output_root, dataset_name, scene, f"{stage1_lmbda}_stage2_ec_only") # 新的输出目录

# --- 6. 构建完整的训练命令 ---
# [!! 关键 !!] 调用 train_ec.py
command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train_ec.py " # <-- 调用新脚本
command += f"-s {data_path} "
command += f"-m {output_path} "
command += f"--iterations {ec_iterations} "
command += f"--eval "

# --- [!! 关键 !!] 传递 阶段 2 和 EC 参数 ---
command += f"--load_stage1_checkpoint {stage1_checkpoint} "
command += f"--ec_model_dim {ec_model_dim} "
command += f"--ec_nhead {ec_nhead} "
command += f"--ec_num_encoder_layers {ec_num_encoder_layers} "
command += f"--ec_dim_feedforward {ec_dim_feedforward} "
command += f"--ec_max_neighbors {ec_max_neighbors} "
command += f"--ec_dropout {ec_dropout} "

# --- [!! 关键 !!] 传递 EC 学习率 ---
command += f"--ec_transformer_lr_init {ec_transformer_lr_init} "
command += f"--ec_transformer_lr_final {ec_transformer_lr_final} "
command += f"--ec_transformer_lr_max_steps {ec_transformer_lr_max_steps} "

command += f"--save_iterations {ec_iterations} "
command += f"--test_iterations {ec_iterations} "

print("="*80)
print(f"Starting STAGE 2 (EC Transformer Only) training for scene: {scene}.")
print(f"Loading Stage 1 model from: {stage1_checkpoint}")
print(f"Output directory: {output_path}")
print("Command to be executed:")
print(command)
print("="*80)

if not os.path.exists(stage1_checkpoint):
    print(f"ERROR: Stage 1 checkpoint not found at: {stage1_checkpoint}")
else:
    os.system(command)

print("="*80)
print(f"STAGE 2 Training finished for scene: {scene}")
print("="*80)