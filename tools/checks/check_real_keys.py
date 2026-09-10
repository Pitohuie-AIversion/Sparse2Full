#!/usr/bin/env python3
"""
检查实际数据文件中的键名
"""
import h5py
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def check_data_keys(data_path):
    """检查HDF5文件中的键名"""
    print(f"检查数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return None
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 文件打开成功")
            print(f"📁 根级键名: {list(f.keys())}")
            
            # 检查每个键的详细信息
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Group):
                    print(f"  - {key}: [Group] 子键: {list(data.keys())}")
                    # 检查组内的数据集
                    for subkey in data.keys():
                        subdata = data[subkey]
                        if isinstance(subdata, h5py.Dataset):
                            print(f"    - {subkey}: shape={subdata.shape}, dtype={subdata.dtype}")
                elif isinstance(data, h5py.Dataset):
                    print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
            
            return list(f.keys())
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

def main():
    """检查所有数据集的键名"""
    data_paths = [
        PROJECT_ROOT / "data/2D_diff-react_NA_NA.h5",
        PROJECT_ROOT / "data/DR2D/pdebench_tiny.h5",
        PROJECT_ROOT / "test_data.h5"
    ]
    
    results = {}
    
    for data_path in data_paths:
        print("\n" + "="*60)
        keys = check_data_keys(data_path)
        if keys:
            results[data_path] = keys
    
    print("\n" + "="*60)
    print("📊 汇总结果:")
    for path, keys in results.items():
        dataset_name = os.path.basename(path).replace('.h5', '')
        print(f"\n{dataset_name}:")
        print(f"  键名数量: {len(keys)}")
        print(f"  键名列表: {keys[:10]}{'...' if len(keys) > 10 else ''}")

if __name__ == "__main__":
    main()
