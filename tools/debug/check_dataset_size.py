#!/usr/bin/env python3
"""
检查数据集的实际大小和样本数量
"""
import h5py
import os

def check_dataset_size(data_path):
    """检查HDF5文件中每个键的样本数量"""
    print(f"检查数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return None
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 文件打开成功")
            print(f"📁 根级键名: {list(f.keys())}")
            
            min_samples = float('inf')
            max_samples = 0
            
            # 检查每个键的详细信息
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Group):
                    print(f"  - {key}: [Group] 子键: {list(data.keys())}")
                    # 检查组内的数据集
                    for subkey in data.keys():
                        subdata = data[subkey]
                        if isinstance(subdata, h5py.Dataset):
                            samples = subdata.shape[0] if len(subdata.shape) > 0 else 0
                            print(f"    - {subkey}: shape={subdata.shape}, dtype={subdata.dtype}, samples={samples}")
                            min_samples = min(min_samples, samples)
                            max_samples = max(max_samples, samples)
                elif isinstance(data, h5py.Dataset):
                    samples = data.shape[0] if len(data.shape) > 0 else 0
                    print(f"  - {key}: shape={data.shape}, dtype={data.dtype}, samples={samples}")
                    min_samples = min(min_samples, samples)
                    max_samples = max(max_samples, samples)
            
            if min_samples != float('inf'):
                print(f"📊 样本数量范围: {min_samples} - {max_samples}")
                return min_samples, max_samples
            else:
                return None, None
                
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None, None

def main():
    """检查所有数据集的样本数量"""
    data_paths = [
        "data/2D/shallow-water/2D_rdb_NA_NA.h5",
        "data/DR2D/2D_diff-react_NA_NA.h5", 
        "data/2D/NS_incom/ns_incom_inhom_2d_512-0.h5"
    ]
    
    results = {}
    
    for data_path in data_paths:
        print("\n" + "="*60)
        min_samples, max_samples = check_dataset_size(data_path)
        if min_samples is not None:
            results[data_path] = (min_samples, max_samples)
    
    print("\n" + "="*60)
    print("📊 汇总结果:")
    for path, (min_samples, max_samples) in results.items():
        dataset_name = os.path.basename(path).replace('.h5', '')
        print(f"\n{dataset_name}:")
        print(f"  最小样本数: {min_samples}")
        print(f"  最大样本数: {max_samples}")
        if min_samples != max_samples:
            print(f"  ⚠️  警告: 不同键的样本数量不一致!")

if __name__ == "__main__":
    main()