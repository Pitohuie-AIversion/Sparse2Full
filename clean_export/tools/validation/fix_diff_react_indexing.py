#!/usr/bin/env python3
"""
修复diff-react数据集的索引问题
"""

import h5py
import torch

def analyze_diff_react_structure():
    """分析diff-react数据集的结构"""
    file_path = "data/DR2D/2D_diff-react_NA_NA.h5"
    
    print("=== 分析diff-react数据集结构 ===")
    
    with h5py.File(file_path, 'r') as f:
        print(f"根级别键数量: {len(f.keys())}")
        print(f"前5个根级别键: {list(f.keys())[:5]}")
        
        # 检查第一个组
        first_key = list(f.keys())[0]
        print(f"\n检查组 '{first_key}':")
        group = f[first_key]
        print(f"  类型: {type(group)}")
        print(f"  键: {list(group.keys())}")
        
        if 'data' in group:
            data_shape = group['data'].shape
            print(f"  data形状: {data_shape}")
            print(f"  data类型: {group['data'].dtype}")
        
        if 'grid' in group:
            print(f"  grid键: {list(group['grid'].keys())}")
    
    print("\n=== 模拟正确的数据访问方式 ===")
    
    # 模拟正确的访问方式
    with h5py.File(file_path, 'r') as f:
        # 方式1：直接使用case_id（字符串形式的组名）
        case_id = "0000"
        if case_id in f:
            group = f[case_id]
            if 'data' in group:
                data = torch.tensor(group['data'][:], dtype=torch.float32)
                print(f"方式1 - 直接访问组 '{case_id}': data形状 = {data.shape}")
        
        # 方式2：使用数字索引获取组名，然后访问
        case_idx = 16
        root_keys = list(f.keys())
        if case_idx < len(root_keys):
            group_key = root_keys[case_idx]
            group = f[group_key]
            if 'data' in group:
                data = torch.tensor(group['data'][:], dtype=torch.float32)
                print(f"方式2 - 索引访问组 '{group_key}' (idx={case_idx}): data形状 = {data.shape}")
        else:
            print(f"方式2 - 索引 {case_idx} 超出范围，总共有 {len(root_keys)} 个组")

if __name__ == "__main__":
    analyze_diff_react_structure()