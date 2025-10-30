#!/usr/bin/env python3
"""
测试数据验证修复是否生效
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.pdebench import PDEBenchBase
import h5py

def test_ns_incom_dataset():
    """测试ns_incom数据集的4D格式支持"""
    print("=== 测试 ns_incom 数据集 ===")
    
    data_path = "E:/2D/NavierStokes/ns_incom_inhom_2d_512-0.h5"
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return False
    
    # 检查数据形状
    with h5py.File(data_path, 'r') as f:
        print(f"HDF5文件键: {list(f.keys())}")
        for key in ['u', 'v', 'p']:
            if key in f:
                print(f"键 '{key}' 形状: {f[key].shape}")
    
    try:
        # 创建数据集实例
        dataset = PDEBenchBase(
            data_path=data_path,
            keys=['u', 'v', 'p'],
            case_ids=['0', '1', '2', '3'],
            use_official_format=True,
            normalize=False
        )
        
        print(f"数据集创建成功，长度: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"样本数据形状: {sample['data'].shape}")
        print("✅ ns_incom 数据集测试通过")
        return True
        
    except Exception as e:
        print(f"❌ ns_incom 数据集测试失败: {e}")
        return False

def test_diff_react_dataset():
    """测试diff-react数据集"""
    print("\n=== 测试 diff-react 数据集 ===")
    
    data_path = "E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return False
    
    # 检查数据形状
    with h5py.File(data_path, 'r') as f:
        print(f"HDF5文件根级别键数量: {len(f.keys())}")
        first_key = list(f.keys())[0]
        print(f"第一个键 '{first_key}' 内容: {list(f[first_key].keys())}")
        if 'data' in f[first_key]:
            print(f"数据形状: {f[first_key]['data'].shape}")
    
    try:
        # 创建数据集实例
        dataset = PDEBenchBase(
            data_path=data_path,
            keys=['data'],
            case_ids=['0000', '0001', '0002', '0003'],
            use_official_format=True,
            normalize=False
        )
        
        print(f"数据集创建成功，长度: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"样本数据形状: {sample['data'].shape}")
        print("✅ diff-react 数据集测试通过")
        return True
        
    except Exception as e:
        print(f"❌ diff-react 数据集测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试数据验证修复...")
    
    success_count = 0
    total_tests = 2
    
    if test_ns_incom_dataset():
        success_count += 1
    
    if test_diff_react_dataset():
        success_count += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！数据验证修复成功")
    else:
        print("⚠️ 部分测试失败，需要进一步修复")