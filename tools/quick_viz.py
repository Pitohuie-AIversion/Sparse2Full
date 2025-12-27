#!/usr/bin/env python3
"""
快速可视化脚本 - 生成简单的PNG图像
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def quick_visualize():
    """快速生成可视化图像"""
    
    data_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/....data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    print(f"读取数据: {data_path}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print("文件结构:")
            def print_structure(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"  {name}: {obj.shape}")
                else:
                    print(f"  {name}: (group)")
            
            f.visititems(print_structure)
            
            # 尝试找到数据
            if 'tensor' in f:
                data = f['tensor'][0]  # 取第一个样本
            elif 'data' in f:
                data = f['data'][0]
            else:
                # 取第一个可用的数据集
                keys = list(f.keys())
                data_key = keys[0]
                data = f[data_key][0]
            
            print(f"数据形状: {data.shape}")
            
            # 设置高质量图像参数
            plt.rcParams['figure.dpi'] = 200
            plt.rcParams['savefig.dpi'] = 200
            
            # 1. 初始状态
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle('初始状态 (t=0)', fontsize=12, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[0, :, :, i], cmap='viridis')
                axes[i].set_title(f'通道 {i+1}')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_initial.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_initial.png")
            
            # 2. 中间状态
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle('中间状态 (t=50)', fontsize=12, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[50, :, :, i], cmap='viridis')
                axes[i].set_title(f'通道 {i+1}')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_middle.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_middle.png")
            
            # 3. 最终状态
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle('最终状态 (t=100)', fontsize=12, fontweight='bold')
            
            for i in range(2):
                im = axes[i].imshow(data[100, :, :, i], cmap='viridis')
                axes[i].set_title(f'通道 {i+1}')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig('sample_0_final.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_final.png")
            
            # 4. 对比图
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle('通道对比 - 时间演化', fontsize=14, fontweight='bold')
            
            times = [0, 50, 100]
            labels = ['t=0', 't=50', 't=100']
            
            for t_idx, (t, label) in enumerate(zip(times, labels)):
                for ch in range(2):
                    im = axes[ch, t_idx].imshow(data[t, :, :, ch], cmap='viridis')
                    axes[ch, t_idx].set_title(f'通道{ch+1} {label}')
                    plt.colorbar(im, ax=axes[ch, t_idx])
            
            plt.tight_layout()
            plt.savefig('sample_0_channels_comparison.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_channels_comparison.png")
            
            print("\n🎉 所有图像生成完成!")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_visualize()