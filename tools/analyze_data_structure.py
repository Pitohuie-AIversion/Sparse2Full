#!/usr/bin/env python3
"""
分析test_data.h5的数据结构
"""

import h5py
import numpy as np

def analyze_h5_structure(file_path):
    """分析H5文件的结构"""
    print(f"正在分析文件: {file_path}")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        # 打印根级别的键
        print("根级别键:")
        root_keys = list(f.keys())
        print(f"  总键数: {len(root_keys)}")
        for i, key in enumerate(root_keys[:10]):  # 只显示前10个
            print(f"  {i+1}. {key}")
        if len(root_keys) > 10:
            print(f"  ... 还有 {len(root_keys) - 10} 个键")
        print()
        
        # 分析前几个样本的结构
        print("样本结构分析:")
        for i, sample_key in enumerate(root_keys[:3]):
            print(f"\n样本 {sample_key}:")
            sample_group = f[sample_key]
            
            # 检查样本组内的键
            sample_keys = list(sample_group.keys())
            print(f"  包含的子键: {sample_keys}")
            
            for sub_key in sample_keys:
                data = sample_group[sub_key]
                if isinstance(data, h5py.Dataset):
                    print(f"    {sub_key}: shape={data.shape}, dtype={data.dtype}")
                    if len(data.shape) >= 2:
                        print(f"      数据范围: [{np.min(data[:]):.4f}, {np.max(data[:]):.4f}]")
                        print(f"      均值: {np.mean(data[:]):.4f}")
                        print(f"      前5个值: {data[:].flat[:5]}")
                elif isinstance(data, h5py.Group):
                    print(f"    {sub_key}: (Group)")
                    # 递归分析子组
                    for sub_sub_key in data.keys():
                        sub_data = data[sub_sub_key]
                        if isinstance(sub_data, h5py.Dataset):
                            print(f"      {sub_sub_key}: shape={sub_data.shape}, dtype={sub_data.dtype}")
            
            # 如果是数字键，检查其数据内容
            if sample_key.isdigit():
                print(f"  这是数字样本键，符合PDEBench格式")
        
        print("\n" + "=" * 60)
        print("总结:")
        print(f"- 文件格式: {'PDEBench官方格式' if any(key.isdigit() for key in root_keys) else '自定义格式'}")
        print(f"- 样本数量: {len(root_keys)}")
        
        # 检查是否有特定的模式
        if '0000' in root_keys:
            sample_0 = f['0000']
            if 'data' in sample_0:
                data_shape = sample_0['data'].shape
                print(f"- 数据形状: {data_shape}")
                print(f"- 时间步数: {data_shape[0] if len(data_shape) >= 4 else 'N/A'}")
                print(f"- 空间分辨率: {data_shape[1:3] if len(data_shape) >= 3 else 'N/A'}")
                print(f"- 通道数: {data_shape[-1] if len(data_shape) >= 4 else 'N/A'}")

if __name__ == "__main__":
    file_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/test_data.h5"
    analyze_h5_structure(file_path)