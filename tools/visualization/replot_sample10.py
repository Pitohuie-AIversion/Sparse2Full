import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer

run_dir = project_root / "runs_drd_paper" / "AR-DR2D-E2E-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260116"
config_path = run_dir / "config_merged.yaml"
ckpt_path = run_dir / "best.ckpt"

trainer = RealDataARTrainer(str(config_path))
trainer.output_dir = str(run_dir)
trainer.load_checkpoint(str(ckpt_path))

model = trainer.get_model()
model.eval()

test_loader = trainer.test_loader
sample_batch = None
for i, batch in enumerate(test_loader):
    if i == 10:
        sample_batch = batch
        break

if sample_batch is None:
    print("Sample 10 not found!")
    sys.exit(1)

device = trainer.device
input_seq = sample_batch['input_sequence'].to(device)
target_seq = sample_batch['target_sequence'].to(device)

print(f"input_seq: {input_seq.shape}")
print(f"target_seq: {target_seq.shape}")

with torch.no_grad():
    # Pass target_seq directly to match the expected signature in sequential_spatiotemporal.py
    pred_seq = model(input_seq, target_seq)
    # Extract final_pred if it's a dict
    if isinstance(pred_seq, dict) and 'final_pred' in pred_seq:
        pred_seq = pred_seq['final_pred']
    elif hasattr(pred_seq, 'final_pred'):
        pred_seq = pred_seq.final_pred

obs_seq = sample_batch.get('observed_lr_sequence', input_seq)
if obs_seq is None:
    obs_seq = input_seq

# 合并输入时间步和预测时间步
# T_in 是输入的历史帧数
T_in = input_seq.shape[1]
T_out = pred_seq.shape[1]
total_t = T_in + T_out

# 通过模型自带的降质算子，显式地对原图进行降采样，以获得真正的 Obs (低分辨率/稀疏观测)
observation_op = trainer.observation_op if hasattr(trainer, 'observation_op') else getattr(trainer, 'training_degradation_op', None)

if observation_op is not None:
    # input_seq 是高分辨率原图 [B, T_in, C, H, W]
    B, T, C, H, W = input_seq.shape
    input_seq_flat = input_seq.view(B * T, C, H, W)
    with torch.no_grad():
        obs_seq_flat = observation_op(input_seq_flat)
    _, _, H_lr, W_lr = obs_seq_flat.shape
    obs_seq_history = obs_seq_flat.view(B, T, C, H_lr, W_lr)
else:
    # 兜底：如果找不到降质算子，使用之前获取的 obs_seq
    obs_seq_history = obs_seq

# 修改 t_indices 为连续的时间步，从 0 到 10（取决于预测序列的长度）
seq_len = pred_seq.shape[1]
t_indices = list(range(seq_len))

# 修改布局：左边把后5s对应的两行平移放到下面空白处变成4*5+4*1的排布
# 意味着图表总共需要 4 行，列数为 5 + T_out。
# 历史序列有 10 帧 (t=-10 到 -1)，现在分成两行显示：
# 第一行（第0行）：Obs t=-10 到 t=-6，紧接着是未来的 Obs t=0 到 t=9
# 第二行（第1行）：GT t=-10 到 t=-6，紧接着是未来的 GT t=0 到 t=9
# 第三行（第2行）：Obs t=-5 到 t=-1，紧接着是未来的 Pred t=0 到 t=9
# 第四行（第3行）：GT t=-5 到 t=-1，紧接着是未来的 Error t=0 到 t=9
# 这样总列数 = 5 (历史) + T_out (未来预测 10 帧) = 15 列
cols_history = 5
total_cols = cols_history + T_out

# 创建图表
fig, axes = plt.subplots(4, total_cols, figsize=(3 * total_cols, 12), squeeze=False)

# 统一使用 vmin=-2.5, vmax=2.5 作为 colormap 的范围
vmin, vmax = -2.5, 2.5

# 1. 绘制历史输入序列 (t = -T_in 到 -1)
# 前 5 帧 (-10 到 -6) 放在第 0 行和第 1 行
# 后 5 帧 (-5 到 -1) 放在第 2 行和第 3 行
for idx in range(T_in):
    t_relative = idx - T_in  # -10, -9, ..., -1
    
    # 确定放置的行和列
    if idx < 5:
        row_obs, row_gt = 0, 1
        col = idx
    else:
        row_obs, row_gt = 2, 3
        col = idx - 5
        
    # 历史观测
    if obs_seq_history.shape[-1] != input_seq.shape[-1]:
        obs_tensor = obs_seq_history[0:1, idx:idx+1]
        B_temp, T_temp, C_temp, H_lr, W_lr = obs_tensor.shape
        obs_tensor = obs_tensor.view(B_temp * T_temp, C_temp, H_lr, W_lr)
        obs_tensor_up = torch.nn.functional.interpolate(obs_tensor, size=(input_seq.shape[-2], input_seq.shape[-1]), mode='nearest')
        obs = obs_tensor_up[0, 0].cpu().numpy()
    else:
        obs = obs_seq_history[0, idx, 0].cpu().numpy() if obs_seq_history.shape[1] > idx else obs_seq_history[0, -1, 0].cpu().numpy()
    
    # 历史真实值
    gt_history = input_seq[0, idx, 0].cpu().numpy() if input_seq.shape[1] > idx else input_seq[0, -1, 0].cpu().numpy()
    
    # 绘制 Obs
    im0 = axes[row_obs, col].imshow(obs, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[row_obs, col].set_title(f'Obs t={t_relative}')
    axes[row_obs, col].axis('off')
    
    # 绘制 GT
    im1 = axes[row_gt, col].imshow(gt_history, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[row_gt, col].set_title(f'GT t={t_relative}')
    axes[row_gt, col].axis('off')

# 2. 绘制未来预测序列 (t = 0 到 T_out-1)
# 放在第 5 列到最后，占用所有的 4 行
for t in range(T_out):
    col = cols_history + t
    
    if t >= target_seq.shape[1]:
        t_data = target_seq.shape[1] - 1
    else:
        t_data = t
    
    # 获取数据
    if observation_op is not None:
        target_frame = target_seq[:, t_data:t_data+1]
        B_temp, T_frame, C_temp, H_temp, W_temp = target_frame.shape
        target_frame_flat = target_frame.view(B_temp * T_frame, C_temp, H_temp, W_temp)
        with torch.no_grad():
            obs_frame_flat = observation_op(target_frame_flat)
        if obs_frame_flat.shape[-1] != target_frame.shape[-1]:
             obs_frame_up = torch.nn.functional.interpolate(obs_frame_flat, size=(H_temp, W_temp), mode='nearest')
             future_obs = obs_frame_up[0, 0].cpu().numpy()
        else:
             future_obs = obs_frame_flat[0, 0].cpu().numpy()
    else:
        future_obs = target_seq[0, t_data, 0].cpu().numpy()

    gt = target_seq[0, t_data, 0].cpu().numpy()
    pred = pred_seq[0, t_data, 0].cpu().numpy()
    err = torch.abs(target_seq[0, t_data, 0] - pred_seq[0, t_data, 0]).cpu().numpy()
    
    # 第一行: Obs
    im0 = axes[0, col].imshow(future_obs, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'Obs t={t}')
    axes[0, col].axis('off')
    
    # 第二行: GT
    im1 = axes[1, col].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'GT t={t}')
    axes[1, col].axis('off')
    
    # 第三行: Pred
    im2 = axes[2, col].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2, col].set_title(f'Pred t={t}')
    axes[2, col].axis('off')
    
    # 第四行: Error
    im3 = axes[3, col].imshow(err, cmap='inferno')
    axes[3, col].set_title(f'Error t={t}')
    axes[3, col].axis('off')

# 调整子图间距，为大标题留出空间
plt.subplots_adjust(top=0.90)

# 使用 fig.text 添加大标题，使其与图片分栏对齐
fig.canvas.draw()
pos_history_start = axes[0, 0].get_position()
pos_history_end = axes[0, cols_history-1].get_position()
pos_future_start = axes[0, cols_history].get_position()
pos_future_end = axes[0, -1].get_position()

# 历史部分的大标题
history_center_x = (pos_history_start.x0 + pos_history_end.x1) / 2
fig.text(history_center_x, 0.95, 'Input Sequence (History)', ha='center', va='center', fontsize=24, fontweight='bold', color='black')

# 预测部分的大标题
future_center_x = (pos_future_start.x0 + pos_future_end.x1) / 2
fig.text(future_center_x, 0.95, 'Output Sequence (Predictions)', ha='center', va='center', fontsize=24, fontweight='bold', color='black')

# 为全图添加 colorbar
# 我们希望 colorbar 使用你指定的样式 (针对四联图的)
# 考虑到现在有 4 行，其中前 3 行是物理量，第 4 行是误差
# 但在历史部分，第 3、4 行也是物理量 (Obs, GT)
# 所以对于物理量的 colorbar，我们需要应用到所有属于物理量的图
cbar_ax = fig.add_axes([1.01, 0.35, 0.01, 0.5])
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label('Value', fontsize=12)

cbar_err_ax = fig.add_axes([1.01, 0.1, 0.01, 0.15])
cbar_err = fig.colorbar(im3, cax=cbar_err_ax)
cbar_err.set_label('Abs Error', fontsize=12)

# 添加垂直分隔线 (跨越所有行，在 cols_history-1 和 cols_history 之间)
import matplotlib.lines as lines
x_line = (pos_history_end.x1 + pos_future_start.x0) / 2
line = lines.Line2D([x_line, x_line], [0.05, 0.95], transform=fig.transFigure, color='red', linestyle='--', linewidth=2)
fig.add_artist(line)

out_path = run_dir / "test_visualizations" / "visualizations" / "predictions" / "sample_0010_obs_gt_pred_error_t10.svg"
plt.savefig(str(out_path), bbox_inches='tight')
print(f"Saved to {out_path}")
