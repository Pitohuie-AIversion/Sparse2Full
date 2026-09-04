#!/usr/bin/env python3
"""
创建PDEBench 2D反应扩散数据的静态可视化
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pathlib import Path

def create_static_visualizations():
    """创建静态可视化图像"""
    
    # 数据文件路径
    project_root = Path(__file__).resolve().parents[1]
    file_path = str(project_root / "data/DR2D/2D_diff-react_NA_NA.h5")
    
    with h5py.File(file_path, 'r') as f:
        # 选择第一个样本
        sample_0 = f['0000']
        data = sample_0['data'][...]  # shape: (101, 128, 128, 2)
        t_grid = sample_0['grid']['t'][...]
        x_grid = sample_0['grid']['x'][...]
        y_grid = sample_0['grid']['y'][...]
        
        print(f"数据形状: {data.shape}")
        print(f"时间范围: [{t_grid[0]:.2f}, {t_grid[-1]:.2f}]")
        print(f"空间范围 x: [{x_grid[0]:.2f}, {x_grid[-1]:.2f}]")
        print(f"空间范围 y: [{y_grid[0]:.2f}, {y_grid[-1]:.2f}]")
        
        # 1. 创建时间演化的快照图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('2D Reaction-Diffusion Evolution (Sample 0)', fontsize=16)
        
        time_indices = [0, 20, 40, 60, 80, 100]  # 选择6个时间点
        time_indices = time_indices[:4]  # 只显示4个时间点
        
        for i, t_idx in enumerate(time_indices):
            # 通道0
            im1 = axes[0, i].imshow(data[t_idx, :, :, 0], 
                                   extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                                   origin='lower', cmap='viridis')
            axes[0, i].set_title(f'Channel 0, t={t_grid[t_idx]:.2f}')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 通道1
            im2 = axes[1, i].imshow(data[t_idx, :, :, 1], 
                                   extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                                   origin='lower', cmap='plasma')
            axes[1, i].set_title(f'Channel 1, t={t_grid[t_idx]:.2f}')
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.tight_layout()
        plt.savefig('2d_reacdiff_evolution_snapshots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 创建初始和最终状态的对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Initial vs Final States (Sample 0)', fontsize=16)
        
        # 初始状态
        im1 = axes[0, 0].imshow(data[0, :, :, 0], 
                               extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                               origin='lower', cmap='viridis')
        axes[0, 0].set_title(f'Channel 0 - Initial (t={t_grid[0]:.2f})')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(data[-1, :, :, 0], 
                               extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                               origin='lower', cmap='viridis')
        axes[0, 1].set_title(f'Channel 0 - Final (t={t_grid[-1]:.2f})')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[1, 0].imshow(data[0, :, :, 1], 
                               extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                               origin='lower', cmap='plasma')
        axes[1, 0].set_title(f'Channel 1 - Initial (t={t_grid[0]:.2f})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(data[-1, :, :, 1], 
                               extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                               origin='lower', cmap='plasma')
        axes[1, 1].set_title(f'Channel 1 - Final (t={t_grid[-1]:.2f})')
        plt.colorbar(im4, ax=axes[1, 1])
        
        for ax in axes.flat:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig('2d_reacdiff_initial_vs_final.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 创建时间序列图（选择几个空间点的时间演化）
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temporal Evolution at Selected Points', fontsize=16)
        
        # 选择几个空间点
        points = [(32, 32), (64, 64), (96, 96), (32, 96)]  # (y, x) 索引
        point_labels = ['(32,32)', '(64,64)', '(96,96)', '(32,96)']
        
        for i, ((y_idx, x_idx), label) in enumerate(zip(points, point_labels)):
            ax = axes[i//2, i%2]
            
            # 绘制两个通道的时间演化
            ax.plot(t_grid, data[:, y_idx, x_idx, 0], 'b-', label='Channel 0', linewidth=2)
            ax.plot(t_grid, data[:, y_idx, x_idx, 1], 'r-', label='Channel 1', linewidth=2)
            ax.set_title(f'Point {label}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('2d_reacdiff_temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 创建数据统计图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Data Statistics', fontsize=16)
        
        # 每个时间步的全局统计
        mean_ch0 = np.mean(data[:, :, :, 0], axis=(1, 2))
        mean_ch1 = np.mean(data[:, :, :, 1], axis=(1, 2))
        std_ch0 = np.std(data[:, :, :, 0], axis=(1, 2))
        std_ch1 = np.std(data[:, :, :, 1], axis=(1, 2))
        
        axes[0, 0].plot(t_grid, mean_ch0, 'b-', label='Channel 0', linewidth=2)
        axes[0, 0].plot(t_grid, mean_ch1, 'r-', label='Channel 1', linewidth=2)
        axes[0, 0].set_title('Mean Values Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t_grid, std_ch0, 'b-', label='Channel 0', linewidth=2)
        axes[0, 1].plot(t_grid, std_ch1, 'r-', label='Channel 1', linewidth=2)
        axes[0, 1].set_title('Standard Deviation Over Time')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Std Dev')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 数据分布直方图
        axes[1, 0].hist(data[:, :, :, 0].flatten(), bins=50, alpha=0.7, label='Channel 0', color='blue')
        axes[1, 0].set_title('Channel 0 Value Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(data[:, :, :, 1].flatten(), bins=50, alpha=0.7, label='Channel 1', color='red')
        axes[1, 1].set_title('Channel 1 Value Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('2d_reacdiff_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 静态可视化图像已生成:")
        print("  - 2d_reacdiff_evolution_snapshots.png")
        print("  - 2d_reacdiff_initial_vs_final.png") 
        print("  - 2d_reacdiff_temporal_evolution.png")
        print("  - 2d_reacdiff_statistics.png")

if __name__ == "__main__":
    create_static_visualizations()