import os

# --- 配置参数 ---
scene = 'playroom' # 要训练的场景
dataset_name = 'db' # 数据集名称 (用于路径)
data_root = 'data' # 数据集的根目录
output_root = 'outputs' # 输出的根目录

lmbda = 0.004 # 压缩率参数
iterations = 30_000 # 总训练迭代次数

# 场景特定参数 (来自 run_shell_db.py)
lod = 0 #
voxel_size = 0.005 #
update_init_factor = 16 #
mask_lr_final_factor = 0.08 # db 场景的特定因子

# GPU 设置
cuda_device = 0

# --- 构造参数 ---
data_path = os.path.join(data_root, dataset_name, scene)
# 为阶段1创建一个清晰的输出目录
output_path = os.path.join(output_root, dataset_name, scene, f"{lmbda}_stage1_hac_only")

# 计算 mask_lr_final (与原始脚本一致)
mask_lr_final = mask_lr_final_factor * lmbda / 0.001 #

# --- 构建完整的训练命令 ---
# 确保你使用的是修改后的 train.py (即 train (2).py)
command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py "
command += f"-s {data_path} " # 数据源路径
command += f"-m {output_path} " # 模型输出路径
command += f"--iterations {iterations} " # 训练迭代次数
command += f"--eval " #
command += f"--lod {lod} " #
command += f"--voxel_size {voxel_size} " #
command += f"--update_init_factor {update_init_factor} " #
command += f"--lmbda {lmbda} " #
command += f"--mask_lr_final {mask_lr_final} " #

# --- [!! 关键 !!] ---
# --- 不传递任何 ec_* 参数 (ec_model_dim, ec_transformer_lr_init 等) ---
# --- 这将使 train.py 和 gaussian_model.py ---
# --- 自动进入 阶段 1 模式 (ec_model_dim=None, self.ec_transformer=None) ---
# --- ↓↓↓ (已删除) ↓↓↓ ---
# command += f"--ec_transformer_lr_init {ec_transformer_lr_init} "
# command += f"--ec_transformer_lr_final {ec_transformer_lr_final} "
# --- ↑↑↑ (已删除) ↑↑↑ ---

command += f"--save_iterations {iterations} " # 在最后保存
command += f"--test_iterations {iterations} " # 在最后测试

print("="*80)
print(f"Starting STAGE 1 (HAC++ Only) training for scene: {scene}.")
print(f"Output directory: {output_path}")
print("Command to be executed:")
print(command)
print("="*80)
print("This will run Stage 1 (HAC++ only). EC Transformer will NOT be loaded.")
print("This should resolve the distCUDA2 OOM error.")
print("="*80)

# --- 执行命令 ---
os.system(command)

print("="*80)
print(f"STAGE 1 Training finished for scene: {scene}")
print(f"Checkpoint saved at: {output_path}/chkpnt{iterations}.pth")
print("="*80)