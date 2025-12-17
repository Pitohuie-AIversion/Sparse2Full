#!/usr/bin/env python3
"""
详细检查test_data.h5的结构
"""
import h5py

def analyze_h5_structure(file_path):
    """详细分析HDF5文件结构"""
    print(f"分析文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"根级别键: {list(f.keys())}")
        
        # 检查第一个样本
        first_key = list(f.keys())[0]
        print(f"\n第一个样本 ({first_key}):")
        first_group = f[first_key]
        print(f"  子键: {list(first_group.keys())}")
        
        # 检查data数据集
        if 'data' in first_group:
            data = first_group['data']
            print(f"  data shape: {data.shape}")
            print(f"  data dtype: {data.dtype}")
            print(f"  data dims: {len(data.shape)}")
            
            # 检查数据内容
            if len(data.shape) == 4:  # [T, H, W, C]
                print(f"  格式: [时间步, 高度, 宽度, 通道]")
                print(f"  时间步: {data.shape[0]}")
                print(f"  空间尺寸: {data.shape[1]}x{data.shape[2]}")
                print(f"  通道数: {data.shape[3]}")

if __name__ == "__main__":
    analyze_h5_structure("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/test_data.h5")