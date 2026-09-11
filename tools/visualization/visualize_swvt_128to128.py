"""SWVT (Video Swin Transformer) 128x128 纯真值输入输出可视化脚本

绘制内容：
1. 历史真值输入帧序列 (T_in = 4)
2. 未来真值目标序列 (Ground Truth)
3. SWVT 3D 时空模型推演输出 (SWVT Prediction)
4. 像素级绝对误差分布 (Absolute Error Map)
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 根路径注入
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.registry import create_model
from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule


def main():
    print("🎨 正在生成 SWVT 128x128 时序预测可视化图...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据模块
    data_path = "/root/autodl-tmp/datasets/2D_rdb_NA_NA.h5"
    data_cfg = OmegaConf.create({
        "data": {
            "data_path": data_path,
            "T_in": 4,
            "T_out": 10,
            "normalize": True,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "time_step_start": 0,
            "time_step_end": 100,
            "max_samples": 5,
            "dataloader": {
                "batch_size": 1,
                "val_batch_size": 1,
                "num_workers": 0,
                "pin_memory": False
            }
        },
        "training": {"batch_size": 1}
    })
    
    dm = RealDiffusionReactionDataModule(data_cfg)
    dm.setup(None)
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    input_seq = batch['input_sequence'].to(device)   # [1, 4, 1, 128, 128]
    target_seq = batch['target_sequence'].to(device) # [1, 10, 1, 128, 128]
    
    # 2. 加载 SWVT 模型
    ckpt_path = project_root / "runs" / "test_swvt_gt_verify" / "checkpoints" / "best.pth"
    model = create_model(
        "SWVT",
        in_channels=1,
        out_channels=1,
        hidden_dim=96,
        num_layers=4,
        num_heads=4,
        window_size=(2, 8, 8),
        dropout=0.0
    ).to(device)
    
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 成功加载已训练检查点: {ckpt_path}")
    else:
        print("⚠️ 未找到权重文件，使用当前初始模型推演")
        
    model.eval()
    with torch.no_grad():
        preds = model(input_seq, T_out=target_seq.shape[1]) # [1, 10, 1, 128, 128]
        
    # 3. 数据反归一化或提取 numpy
    in_np = input_seq[0, :, 0].cpu().numpy()     # [4, 128, 128]
    tgt_np = target_seq[0, :, 0].cpu().numpy()   # [10, 128, 128]
    pred_np = preds[0, :, 0].cpu().numpy()       # [10, 128, 128]
    err_np = np.abs(pred_np - tgt_np)            # [10, 128, 128]
    
    # 4. 绘制对比画卷
    # 选取展示时间步：历史 4 帧，未来选取 5 帧 (t=1, 3, 5, 7, 10)
    future_steps = [0, 2, 4, 6, 9]  # 对应未来第 1, 3, 5, 7, 10 步
    num_fut = len(future_steps)
    
    fig, axes = plt.subplots(4, num_fut, figsize=(3.2 * num_fut, 12), dpi=150)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    
    vmin, vmax = tgt_np.min(), tgt_np.max()
    err_max = max(0.1, err_np.max() * 0.8)
    
    # Row 1: 历史输入帧 (前 4 帧展示在列 0~3，第 4 列显示状态概览)
    for i in range(num_fut):
        ax = axes[0, i]
        if i < in_np.shape[0]:
            im = ax.imshow(in_np[i], cmap='viridis', origin='lower')
            ax.set_title(f"History Input t={i+1}\n[128x128]", fontsize=11, fontweight='bold', color='#1a5276')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, f"Temporal Context\nT_in = 4 steps\n128x128 Pure GT", 
                    ha='center', va='center', fontsize=12, color='#2c3e50', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Historical\nInput (T_in)", fontsize=12, fontweight='bold')
    
    # Row 2: 未来真值 Future Ground Truth
    for col, step_idx in enumerate(future_steps):
        ax = axes[1, col]
        im = ax.imshow(tgt_np[step_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"Target Future t={step_idx+5}\n[128x128]", fontsize=11, fontweight='bold', color='#196f3d')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 0].set_ylabel("Ground Truth\n(Future GT)", fontsize=12, fontweight='bold')
    
    # Row 3: SWVT 3D Swin 模型预测输出
    for col, step_idx in enumerate(future_steps):
        ax = axes[2, col]
        im = ax.imshow(pred_np[step_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"SWVT Pred t={step_idx+5}\n[128x128]", fontsize=11, fontweight='bold', color='#7d3c98')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[2, 0].set_ylabel("SWVT Output\n(3D Swin Pred)", fontsize=12, fontweight='bold')
    
    # Row 4: 绝对误差分布 Absolute Error
    for col, step_idx in enumerate(future_steps):
        ax = axes[3, col]
        im = ax.imshow(err_np[step_idx], cmap='inferno', vmin=0, vmax=err_max, origin='lower')
        step_mae = err_np[step_idx].mean()
        ax.set_title(f"Abs Error t={step_idx+5}\nMAE={step_mae:.4f}", fontsize=11, fontweight='bold', color='#b03a2e')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[3, 0].set_ylabel("Absolute\nError Map", fontsize=12, fontweight='bold')
    
    plt.suptitle("SWVT (Video Swin Transformer): 128x128 Spatiotemporal Rollout Prediction", 
                 fontsize=15, fontweight='heavy', y=0.98)
    
    # 计算全局推演定量度量
    overall_mae = float(err_np.mean())
    tgt_norm = np.linalg.norm(tgt_np)
    err_norm = np.linalg.norm(err_np)
    overall_rel_l2 = float(err_norm / (tgt_norm + 1e-8))
    print(f"📊 推演定量度量: Overall MAE = {overall_mae:.5f}, Overall Rel-L2 = {overall_rel_l2:.5f}")

    # 保存输出
    save_dirs = [
        project_root / "runs" / "test_swvt_128to128_seq" / "visualizations"
    ]
    
    for s_dir in save_dirs:
        s_dir.mkdir(parents=True, exist_ok=True)
        out_path = s_dir / "swvt_128to128_prediction.png"
        fig.savefig(out_path, bbox_inches='tight')
        print(f"🖼️ 可视化图已成功保存至: {out_path}")
        
    plt.close(fig)
    print("✅ 可视化任务全部完成!")


if __name__ == "__main__":
    main()
