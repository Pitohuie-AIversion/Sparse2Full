#!/usr/bin/env python3
"""
简单可视化脚本
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = str(project_root / "data/DR2D/2D_diff-react_NA_NA.h5")
    
    print(f"正在读取数据: {data_path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print("HDF5文件结构:")
            for key in f.keys():
                print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'group'}")
            
            # 获取数据
            data = f['tensor'][0]  # 第一个样本
            print(f"数据形状: {data.shape}")
            
            # 生成图像
            plt.figure(figsize=(8, 6))
            plt.imshow(data[0, :, :, 0], cmap='viridis')
            plt.title('初始状态 - 通道1')
            plt.colorbar()
            plt.savefig('sample_0_initial.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_initial.png")
            
            plt.figure(figsize=(8, 6))
            plt.imshow(data[50, :, :, 0], cmap='viridis')
            plt.title('中间状态 - 通道1')
            plt.colorbar()
            plt.savefig('sample_0_middle.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_middle.png")
            
            plt.figure(figsize=(8, 6))
            plt.imshow(data[100, :, :, 0], cmap='viridis')
            plt.title('最终状态 - 通道1')
            plt.colorbar()
            plt.savefig('sample_0_final.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_final.png")
            
            # 对比图
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im1 = axes[0].imshow(data[0, :, :, 0], cmap='viridis')
            axes[0].set_title('通道1 - 初始')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(data[0, :, :, 1], cmap='viridis')
            axes[1].set_title('通道2 - 初始')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig('sample_0_channels_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 生成: sample_0_channels_comparison.png")
            
            print("\n🎉 所有图像生成完成!")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()