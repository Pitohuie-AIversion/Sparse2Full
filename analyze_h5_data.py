#!/usr/bin/env python3
"""
HDF5 数据格式分析脚本
用于分析PDEBench数据集的结构和格式
"""

import h5py
import numpy as np
import os
from pathlib import Path

def analyze_h5_file(file_path):
    """
    分析HDF5文件的结构和内容
    
    Args:
        file_path (str): HDF5文件路径
    """
    print(f"正在分析HDF5文件: {file_path}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 - {file_path}")
        return
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024**2):.2f} MB")
    print()
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("📊 HDF5文件结构分析:")
            print("-" * 50)
            
            # 递归遍历所有组和数据集
            def print_structure(name, obj, level=0):
                indent = "  " * level
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 组 (Group): {name}")
                    # 打印组的属性
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}  🏷️  属性: {attr_name} = {attr_value}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📄 数据集 (Dataset): {name}")
                    print(f"{indent}  📐 形状: {obj.shape}")
                    print(f"{indent}  🔢 数据类型: {obj.dtype}")
                    print(f"{indent}  💾 大小: {obj.size} 元素")
                    
                    # 打印数据集的属性
                    if obj.attrs:
                        print(f"{indent}  🏷️  属性:")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}    - {attr_name}: {attr_value}")
                    
                    # 显示数据的统计信息（如果数据不太大）
                    if obj.size > 0 and obj.size < 1e6:  # 只对小于1M元素的数据集显示统计
                        try:
                            data_sample = obj[...]
                            if np.issubdtype(obj.dtype, np.number):
                                print(f"{indent}  📈 统计信息:")
                                print(f"{indent}    - 最小值: {np.min(data_sample):.6f}")
                                print(f"{indent}    - 最大值: {np.max(data_sample):.6f}")
                                print(f"{indent}    - 平均值: {np.mean(data_sample):.6f}")
                                print(f"{indent}    - 标准差: {np.std(data_sample):.6f}")
                            
                            # 显示数据样本（前几个元素）
                            if obj.ndim <= 2 and obj.size <= 100:
                                print(f"{indent}  🔍 数据样本:")
                                print(f"{indent}    {data_sample}")
                            elif obj.ndim > 2:
                                print(f"{indent}  🔍 数据样本 (前几个切片的形状):")
                                if obj.ndim == 3:
                                    print(f"{indent}    第一个切片形状: {data_sample[0].shape}")
                                elif obj.ndim == 4:
                                    print(f"{indent}    第一个样本形状: {data_sample[0].shape}")
                                    if len(data_sample) > 1:
                                        print(f"{indent}    第二个样本形状: {data_sample[1].shape}")
                        except Exception as e:
                            print(f"{indent}  ⚠️  无法读取数据样本: {e}")
                    
                    print()
            
            # 遍历根级别的所有项目
            f.visititems(print_structure)
            
            # 打印根级别的属性
            print("\n🏷️  根级别属性:")
            if f.attrs:
                for attr_name, attr_value in f.attrs.items():
                    print(f"  - {attr_name}: {attr_value}")
            else:
                print("  (无根级别属性)")
            
            # 总结信息
            print("\n" + "=" * 80)
            print("📋 数据集总结:")
            
            datasets = []
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.dtype, obj.size))
            
            f.visititems(collect_datasets)
            
            if datasets:
                print(f"  📊 总共发现 {len(datasets)} 个数据集:")
                for name, shape, dtype, size in datasets:
                    print(f"    - {name}: {shape} ({dtype}) - {size} 元素")
            else:
                print("  ❌ 未发现任何数据集")
                
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 目标文件路径
    file_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/....data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    print("🔍 PDEBench HDF5 数据格式分析器")
    print("=" * 80)
    
    analyze_h5_file(file_path)
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()