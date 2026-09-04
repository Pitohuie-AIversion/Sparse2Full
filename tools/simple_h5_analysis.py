#!/usr/bin/env python3
import h5py
import numpy as np
import os
from pathlib import Path

def analyze_h5():
    project_root = Path(__file__).resolve().parents[1]
    file_path = str(project_root / "data/DR2D/2D_diff-react_NA_NA.h5")
    
    print("=== PDEBench HDF5 数据格式分析 ===")
    
    if not os.path.exists(file_path):
        print("文件不存在!")
        return
    
    file_size = os.path.getsize(file_path) / (1024**2)
    print(f"文件大小: {file_size:.2f} MB")
    
    with h5py.File(file_path, 'r') as f:
        print(f"顶级组数量: {len(f.keys())}")
        
        # 获取前几个组的名称
        keys = list(f.keys())[:5]
        print(f"前5个组: {keys}")
        
        # 分析第一个组
        if keys:
            first_group = f[keys[0]]
            print(f"\n第一个组 '{keys[0]}' 的内容:")
            for key in first_group.keys():
                item = first_group[key]
                if hasattr(item, 'shape'):
                    print(f"  数据集 {key}: shape={item.shape}, dtype={item.dtype}")
                    
                    # 如果是小数据集，显示一些统计信息
                    if item.size < 1000:
                        data = item[...]
                        if np.issubdtype(item.dtype, np.number):
                            print(f"    统计: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
                else:
                    print(f"  子组 {key}: {len(item.keys())} 个项目")
        
        # 查找所有数据集
        datasets = []
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj.shape, obj.dtype))
        
        f.visititems(collect_datasets)
        
        print(f"\n总共找到 {len(datasets)} 个数据集")
        if datasets:
            print("数据集列表:")
            for name, shape, dtype in datasets[:10]:  # 只显示前10个
                print(f"  {name}: {shape} ({dtype})")
            if len(datasets) > 10:
                print(f"  ... 还有 {len(datasets) - 10} 个数据集")

if __name__ == "__main__":
    analyze_h5()