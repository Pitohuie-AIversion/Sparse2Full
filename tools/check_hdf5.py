#!/usr/bin/env python3
"""检查HDF5文件结构"""

import h5py
import sys

def check_hdf5_structure(file_path):
    """检查HDF5文件的结构"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"=== HDF5文件: {file_path} ===")
            print("根目录键:")
            for key in f.keys():
                print(f"  - {key}: {type(f[key])}")
                if isinstance(f[key], h5py.Group):
                    print(f"    子键: {list(f[key].keys())}")
                    for subkey in f[key].keys():
                        item = f[key][subkey]
                        if hasattr(item, 'shape'):
                            print(f"      - {subkey}: shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"      - {subkey}: {type(item)}")
                elif hasattr(f[key], 'shape'):
                    print(f"  形状: {f[key].shape}, 数据类型: {f[key].dtype}")
            
            print("\n=== 递归遍历 ===")
            def print_structure(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{name}: {type(obj)}")
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_hdf5_structure(sys.argv[1])
    else:
        # 默认检查几个文件
        files = [
            "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/data/DR2D/2D_diff-react_NA_NA_small.h5",
            "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/test_data.h5",
            "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/data/DR2D/pdebench_tiny.h5"
        ]
        
        for file_path in files:
            check_hdf5_structure(file_path)
            print("\n" + "="*50 + "\n")