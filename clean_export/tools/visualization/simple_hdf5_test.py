#!/usr/bin/env python3
"""
简单的HDF5文件测试脚本
"""

import os
import h5py
import numpy as np

def test_hdf5_file():
    """测试HDF5文件"""
    data_path = "data/DR2D/2D_diff-react_NA_NA.h5"
    
    print(f"🔍 测试HDF5文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在")
        return False
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 成功打开HDF5文件")
            
            # 获取所有键
            keys = list(f.keys())
            print(f"🔑 顶级键数量: {len(keys)}")
            print(f"🔑 前10个键: {keys[:10]}")
            
            # 检查数字键
            time_keys = [k for k in keys if k.isdigit()]
            time_keys.sort()
            print(f"📅 时序键数量: {len(time_keys)}")
            print(f"📅 前10个时序键: {time_keys[:10]}")
            
            # 检查第一个时序键的数据
            if time_keys:
                first_key = time_keys[0]
                data = f[first_key]
                print(f"📊 第一个时序键 '{first_key}' 的数据:")
                
                if isinstance(data, h5py.Group):
                    print(f"  - 类型: Group (包含子数据集)")
                    sub_keys = list(data.keys())
                    print(f"  - 子键: {sub_keys}")
                    
                    # 检查第一个子数据集
                    if sub_keys:
                        sub_data = data[sub_keys[0]]
                        if isinstance(sub_data, h5py.Dataset):
                            print(f"  - 子数据集 '{sub_keys[0]}' 形状: {sub_data.shape}")
                            print(f"  - 子数据集 '{sub_keys[0]}' 数据类型: {sub_data.dtype}")
                            
                            # 读取一小部分数据
                            if len(sub_data.shape) >= 2:
                                sample = sub_data[:min(2, sub_data.shape[0]), :min(2, sub_data.shape[1])]
                                print(f"  - 样本数据: {sample}")
                                print(f"  - 数值范围: [{np.min(sample):.6f}, {np.max(sample):.6f}]")
                elif isinstance(data, h5py.Dataset):
                    print(f"  - 类型: Dataset")
                    print(f"  - 形状: {data.shape}")
                    print(f"  - 数据类型: {data.dtype}")
                    
                    # 读取一小部分数据
                    if len(data.shape) >= 2:
                        sample = data[:min(2, data.shape[0]), :min(2, data.shape[1])]
                        print(f"  - 样本数据: {sample}")
                        print(f"  - 数值范围: [{np.min(sample):.6f}, {np.max(sample):.6f}]")
            
            return True
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hdf5_file()
    print(f"\n🎯 测试结果: {'成功' if success else '失败'}")