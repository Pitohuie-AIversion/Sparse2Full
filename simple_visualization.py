#!/usr/bin/env python3
"""
简单的2D反应扩散数据可视化脚本
生成可查看的PNG图像文件
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path

def create_simple_visualizations():
    """创建简单的可视化图像"""
    
    # 数据文件路径
    data_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/....data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    print(f"正在读取数据文件: {data_path}")
    
    try:
        # 读取HDF5数据
        with h5py.File(data_path, 'r') as f:
            print("数据文件结构:")
            for key in f.keys():
                print(f"  {key}: {f[key].shape}")
            
            # 获取第一个样本的数据 (样本0)
            data_key = list(f.keys())[0]  # 获取第一个数据键
            data = f[data_key][0]  # 取第一个样本 [time, height, width, channels]
            
            print(f"数据形状: {data.shape}")
            print(f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
            
            # 设置matplotlib参数
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['savefig.dpi'] = 150
            plt.rcParams['font.size'] = 10
            
            # 1. 初始状态图 (t=0)
            print("生成初始状态图...")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('初始状态 (t=0)', fontsize=14, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[0, :, :, i], cmap='viridis', origin='lower')
                axes[i].set_title(f'通道 {i+1}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_initial.png', bbox_inches='tight')
            plt.close()
            
            # 2. 中间状态图 (t=50)
            print("生成中间状态图...")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('中间状态 (t=50)', fontsize=14, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[50, :, :, i], cmap='viridis', origin='lower')
                axes[i].set_title(f'通道 {i+1}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_middle.png', bbox_inches='tight')
            plt.close()
            
            # 3. 最终状态图 (t=100)
            print("生成最终状态图...")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('最终状态 (t=100)', fontsize=14, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[100, :, :, i], cmap='viridis', origin='lower')
                axes[i].set_title(f'通道 {i+1}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_final.png', bbox_inches='tight')
            plt.close()
            
            # 4. 两个通道的对比图
            print("生成通道对比图...")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('两个通道的时间演化对比', fontsize=16, fontweight='bold')
            
            time_points = [0, 50, 100]
            time_labels = ['初始 (t=0)', '中间 (t=50)', '最终 (t=100)']
            
            for t_idx, (t, label) in enumerate(zip(time_points, time_labels)):
                for ch in range(2):
                    im = axes[ch, t_idx].imshow(data[t, :, :, ch], cmap='viridis', origin='lower')
                    axes[ch, t_idx].set_title(f'通道 {ch+1} - {label}')
                    axes[ch, t_idx].set_xlabel('X')
                    axes[ch, t_idx].set_ylabel('Y')
                    plt.colorbar(im, ax=axes[ch, t_idx])
            
            plt.tight_layout()
            plt.savefig('sample_0_channels_comparison.png', bbox_inches='tight')
            plt.close()
            
            print("\n✅ 所有图像已成功生成!")
            print("生成的文件:")
            print("  - sample_0_initial.png")
            print("  - sample_0_middle.png")
            print("  - sample_0_final.png")
            print("  - sample_0_channels_comparison.png")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    create_simple_visualizations()