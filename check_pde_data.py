#!/usr/bin/env python3
"""
检查PDEBench数据集的结构和内容
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_hdf5_structure(file_path):
    """检查HDF5文件结构"""
    print(f"检查文件: {file_path}")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        print("HDF5文件结构:")
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: {obj.shape} {obj.dtype}")
            else:
                print(f"{indent}{name}/")
        
        f.visititems(print_structure)
        
        # 检查根级别的键
        print("\n根级别的键:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'Group'}")
        
        # 如果有tensor键，检查其详细信息
        if 'tensor' in f:
            tensor_data = f['tensor']
            print(f"\ntensor数据详细信息:")
            print(f"  形状: {tensor_data.shape}")
            print(f"  数据类型: {tensor_data.dtype}")
            print(f"  最小值: {np.min(tensor_data[:])}")
            print(f"  最大值: {np.max(tensor_data[:])}")
            print(f"  均值: {np.mean(tensor_data[:])}")
            print(f"  标准差: {np.std(tensor_data[:])}")
            
            # 检查前几个样本
            print(f"\n前3个样本的形状:")
            for i in range(min(3, tensor_data.shape[0])):
                sample = tensor_data[i]
                print(f"  样本{i}: {sample.shape}, 范围: [{np.min(sample):.6f}, {np.max(sample):.6f}]")
        
        # 检查其他可能的键
        for key in ['nu', 'x', 'y', 'coords']:
            if key in f:
                data = f[key]
                print(f"\n{key}数据:")
                print(f"  形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                if data.size < 100:  # 如果数据不大，打印一些值
                    print(f"  值: {data[:]}")

def visualize_samples(file_path, num_samples=3):
    """可视化几个样本"""
    print(f"\n可视化样本...")
    
    with h5py.File(file_path, 'r') as f:
        if 'tensor' not in f:
            print("未找到tensor数据")
            return
        
        tensor_data = f['tensor']
        
        # 创建图形
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            if i >= tensor_data.shape[0]:
                break
                
            # 获取样本数据
            sample = tensor_data[i]
            
            # 如果是4D数据 (1, 128, 128) 或 (128, 128, 1)，提取2D
            if len(sample.shape) == 3:
                if sample.shape[0] == 1:
                    sample_2d = sample[0]  # (1, 128, 128) -> (128, 128)
                elif sample.shape[-1] == 1:
                    sample_2d = sample[:, :, 0]  # (128, 128, 1) -> (128, 128)
                else:
                    sample_2d = sample[0]  # 取第一个通道
            else:
                sample_2d = sample
            
            # 绘制热图
            im = axes[i].imshow(sample_2d, cmap='viridis', aspect='equal')
            axes[i].set_title(f'DarcyFlow样本 {i}\n形状: {sample.shape}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('pde_data_samples.png', dpi=150, bbox_inches='tight')
        print(f"样本可视化已保存到: pde_data_samples.png")
        plt.close()

def main():
    """主函数"""
    data_path = "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    
    if not Path(data_path).exists():
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    # 检查文件结构
    check_hdf5_structure(data_path)
    
    # 可视化样本
    visualize_samples(data_path, num_samples=3)
    
    print("\n数据检查完成!")

if __name__ == "__main__":
    main()