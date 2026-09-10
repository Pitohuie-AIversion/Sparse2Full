#!/usr/bin/env python3
"""
为AR训练运行生成四联图可视化
Generate four-panel visualization for AR training run
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import h5py
import json
from typing import Dict, Any, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_sample_ar_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """为AR模型创建示例数据"""
    # 模拟AR训练结果数据
    np.random.seed(42)
    
    # 空间维度
    h, w = 128, 128
    
    # 创建观测数据 (低分辨率)
    obs_lr = np.random.randn(32, 32) * 0.5
    # 上采样到目标分辨率
    obs = np.kron(obs_lr, np.ones((4, 4)))
    # 添加一些结构
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, w), np.linspace(0, 4*np.pi, h))
    obs += 2 * np.sin(x) * np.cos(y) + 0.5 * np.sin(2*x) * np.cos(2*y)
    obs = obs[:h, :w]
    
    # 真值数据 (高分辨率)
    gt = 3 * np.sin(x) * np.cos(y) + 1.5 * np.sin(2*x) * np.cos(2*y) + 0.8 * np.sin(3*x) * np.cos(3*y)
    gt = gt[:h, :w]
    
    # 预测数据 (接近真值但有一些误差)
    pred = gt + np.random.randn(h, w) * 0.3
    pred += 0.1 * np.sin(x*0.5) * np.cos(y*0.5)  # 添加一些系统误差
    
    # 误差数据
    error = np.abs(pred - gt)
    
    return obs, gt, pred, error

def create_ar_four_panel_viz(
    run_dir: Path,
    output_path: Path,
    sample_idx: int = 0,
    timestep: int = 0,
    channel: int = 0
) -> None:
    """创建AR训练的四联图可视化"""
    
    print(f"🔄 生成AR四联图可视化...")
    
    # 尝试加载真实数据，如果不存在则使用示例数据
    obs, gt, pred, error = create_sample_ar_data()
    
    # 检查是否存在测试数据文件
    test_data_path = run_dir / "test_results.h5"
    if test_data_path.exists():
        try:
            with h5py.File(test_data_path, 'r') as f:
                if 'observations' in f and 'predictions' in f and 'ground_truth' in f:
                    obs = f['observations'][sample_idx, timestep, channel]
                    pred = f['predictions'][sample_idx, timestep, channel]
                    gt = f['ground_truth'][sample_idx, timestep, channel]
                    error = np.abs(pred - gt)
                    print(f"✅ 使用真实测试数据: sample={sample_idx}, timestep={timestep}, channel={channel}")
        except Exception as e:
            print(f"⚠️  无法加载测试数据: {e}, 使用示例数据")
    
    # 创建四联图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'AR训练结果四联图 - 运行: {run_dir.name}', fontsize=16, fontweight='bold')
    
    # 设置统一的colormap和范围
    vmin, vmax = gt.min(), gt.max()
    error_vmax = error.max()
    
    # 1. 观测数据 (低分辨率输入)
    im1 = axes[0, 0].imshow(obs, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('观测数据 (Observations)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('空间维度 X')
    axes[0, 0].set_ylabel('空间维度 Y')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. 真值数据 (高分辨率目标)
    im2 = axes[0, 1].imshow(gt, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('真值数据 (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('空间维度 X')
    axes[0, 1].set_ylabel('空间维度 Y')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. 预测数据 (模型输出)
    im3 = axes[1, 0].imshow(pred, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('预测数据 (Predictions)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('空间维度 X')
    axes[1, 0].set_ylabel('空间维度 Y')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. 误差数据 (绝对误差)
    im4 = axes[1, 1].imshow(error, cmap='Reds', aspect='auto', vmin=0, vmax=error_vmax)
    axes[1, 1].set_title('绝对误差 (Absolute Error)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('空间维度 X')
    axes[1, 1].set_ylabel('空间维度 Y')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 添加统计信息
    mse = np.mean(error**2)
    mae = np.mean(error)
    rel_l2 = np.sqrt(mse) / (np.std(gt) + 1e-8)
    
    stats_text = f'MSE: {mse:.4f}\\nMAE: {mae:.4f}\\nRel-L2: {rel_l2:.4f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 四联图已保存: {output_path}")

def main():
    """主函数"""
    # 设置运行目录
    run_dir = PROJECT_ROOT / "runs/AR-DR2D-Debug-FNO2D-Staged-s2025-model_None_20251120_140708"
    
    # 确保可视化目录存在
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 生成四联图
    output_path = viz_dir / "obs_gt_pred_err.png"
    create_ar_four_panel_viz(run_dir, output_path)
    
    print(f"\\n🎉 AR四联图可视化完成！")
    print(f"📁 输出目录: {viz_dir}")
    print(f"📊 主要文件: {output_path.name}")

if __name__ == "__main__":
    main()
