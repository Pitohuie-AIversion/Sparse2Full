#!/usr/bin/env python3
"""
检查HDF5文件的内容和结构
"""
import h5py
import numpy as np

def check_hdf5_file(file_path):
    """检查HDF5文件的内容"""
    print(f"检查文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\n=== 文件根级别的键 ===")
            for key in f.keys():
                print(f"- {key}")
            
            print("\n=== 详细信息 ===")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                    if obj.size < 10:  # 只打印小数据集的值
                        print(f"  Values: {obj[...]}")
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                    print()
            
            f.visititems(print_structure)
            
            # 特别检查tensor数据
            if 'tensor' in f:
                tensor_data = f['tensor']
                print(f"\n=== tensor数据详情 ===")
                print(f"Shape: {tensor_data.shape}")
                print(f"Dtype: {tensor_data.dtype}")
                print(f"Min value: {np.min(tensor_data)}")
                print(f"Max value: {np.max(tensor_data)}")
                print(f"Mean value: {np.mean(tensor_data)}")
                
                # 检查前几个样本的形状
                if len(tensor_data.shape) >= 3:
                    print(f"First sample shape: {tensor_data[0].shape}")
                    if len(tensor_data) > 1:
                        print(f"Second sample shape: {tensor_data[1].shape}")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    file_path = "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    check_hdf5_file(file_path)