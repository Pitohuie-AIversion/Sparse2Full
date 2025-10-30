#!/usr/bin/env python3
"""调试diff-react数据集的详细结构"""

import h5py
import numpy as np

def debug_diff_react_structure():
    """详细分析diff-react数据集结构"""
    file_path = "E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    print(f"=== 分析 {file_path} ===")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件根级别键: {list(f.keys())}")
            
            # 检查第一个键的详细结构
            first_key = list(f.keys())[0]
            print(f"\n分析第一个键: {first_key}")
            
            item = f[first_key]
            if isinstance(item, h5py.Group):
                print(f"  {first_key} 是一个组")
                group_keys = list(item.keys())
                print(f"  组内键数量: {len(group_keys)}")
                print(f"  前10个组内键: {group_keys[:10]}")
                
                # 检查第一个组内键的结构
                if group_keys:
                    first_group_key = group_keys[0]
                    print(f"\n  分析第一个组内键: {first_group_key}")
                    
                    group_item = item[first_group_key]
                    if isinstance(group_item, h5py.Group):
                        print(f"    {first_group_key} 也是一个组")
                        sub_keys = list(group_item.keys())
                        print(f"    子键: {sub_keys}")
                        
                        # 检查data和grid的形状
                        if 'data' in sub_keys:
                            data_shape = group_item['data'].shape
                            print(f"    data形状: {data_shape}")
                            print(f"    data类型: {group_item['data'].dtype}")
                        
                        if 'grid' in sub_keys:
                            grid_shape = group_item['grid'].shape
                            print(f"    grid形状: {grid_shape}")
                            print(f"    grid类型: {group_item['grid'].dtype}")
                    else:
                        print(f"    {first_group_key} 是数据集，形状: {group_item.shape}")
                
                # 检查几个不同的组内键
                print(f"\n  检查前5个组内键的结构:")
                for i, group_key in enumerate(group_keys[:5]):
                    group_item = item[group_key]
                    if isinstance(group_item, h5py.Group):
                        sub_keys = list(group_item.keys())
                        if 'data' in sub_keys:
                            data_shape = group_item['data'].shape
                            print(f"    {group_key}/data: {data_shape}")
                        else:
                            print(f"    {group_key}: 无data键，子键: {sub_keys}")
                    else:
                        print(f"    {group_key}: 数据集，形状: {group_item.shape}")
            else:
                print(f"  {first_key} 是数据集，形状: {item.shape}")
                
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    debug_diff_react_structure()