#!/usr/bin/env python3
"""
PDEBench数据集分析脚本
分析E:\2D目录下的HDF5文件结构和内容
为主项目训练提供数据配置信息
"""

import os
import h5py
import numpy as np
from pathlib import Path
import json

def analyze_hdf5_file(file_path):
    """分析单个HDF5文件的结构和内容"""
    try:
        with h5py.File(file_path, 'r') as f:
            file_info = {
                'path': str(file_path),
                'size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'keys': list(f.keys()),
                'datasets': {}
            }
            
            # 分析每个数据集
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key]
                    file_info['datasets'][key] = {
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'size_mb': data.nbytes / (1024 * 1024)
                    }
                    
                    # 对于小数据集，计算统计信息
                    if data.nbytes < 100 * 1024 * 1024:  # 小于100MB
                        try:
                            values = data[:]
                            if np.issubdtype(values.dtype, np.number):
                                file_info['datasets'][key].update({
                                    'min': float(np.min(values)),
                                    'max': float(np.max(values)),
                                    'mean': float(np.mean(values)),
                                    'std': float(np.std(values))
                                })
                        except Exception as e:
                            print(f"无法计算 {key} 的统计信息: {e}")
            
            return file_info
            
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {e}")
        return None

def scan_directory(root_dir):
    """扫描目录下的所有HDF5文件"""
    root_path = Path(root_dir)
    hdf5_files = []
    
    # 递归查找所有.hdf5文件
    for file_path in root_path.rglob("*.hdf5"):
        hdf5_files.append(file_path)
    
    return hdf5_files

def generate_training_metadata(analysis_results):
    """基于分析结果生成训练元数据"""
    metadata = {
        'dataset_summary': {
            'total_files': len(analysis_results['files']),
            'total_size_gb': analysis_results['total_size_gb'],
            'subdirectories': analysis_results['subdirectories']
        },
        'training_configs': {},
        'data_format': {
            'coordinate_system': {
                'x_range': [0.003906, 0.996094],
                'y_range': [0.003906, 0.996094],
                'resolution': 128,
                'data_type': 'float32'
            },
            'tensor_structure': {
                'typical_shape': [1000, 1, 128, 128],
                'channels': 1,
                'time_steps': 1000
            }
        }
    }
    
    # 为每个子目录生成训练配置
    for subdir, info in analysis_results['subdirectories'].items():
        size_gb = info['size_gb']
        
        # 根据数据大小调整批处理大小
        if size_gb < 10:
            batch_size = 16
            num_workers = 4
        elif size_gb < 50:
            batch_size = 8
            num_workers = 4
        else:
            batch_size = 4
            num_workers = 2
            
        metadata['training_configs'][subdir] = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'recommended_memory_gb': max(32, int(size_gb * 0.5)),
            'estimated_training_time_hours': int(size_gb * 0.1)
        }
    
    return metadata

def main():
    """主函数"""
    data_root = "E:/2D"
    
    print("=" * 80)
    print("PDEBench 数据集分析")
    print("=" * 80)
    print(f"分析目录: {data_root}")
    
    # 扫描HDF5文件
    hdf5_files = scan_directory(data_root)
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 分析结果存储
    analysis_results = {
        'files': [],
        'total_size_gb': 0,
        'subdirectories': {}
    }
    
    # 分析每个文件
    for i, file_path in enumerate(hdf5_files):
        print(f"\n正在分析文件 {i+1}/{len(hdf5_files)}: {file_path.name}")
        
        file_info = analyze_hdf5_file(file_path)
        if file_info:
            analysis_results['files'].append(file_info)
            analysis_results['total_size_gb'] += file_info['size_mb'] / 1024
            
            # 按子目录分组
            subdir = file_path.parent.name
            if subdir not in analysis_results['subdirectories']:
                analysis_results['subdirectories'][subdir] = {
                    'files': 0,
                    'size_gb': 0
                }
            
            analysis_results['subdirectories'][subdir]['files'] += 1
            analysis_results['subdirectories'][subdir]['size_gb'] += file_info['size_mb'] / 1024
            
            # 详细显示第一个文件的信息
            if i == 0:
                print(f"\n文件详细信息:")
                print(f"  大小: {file_info['size_mb']:.2f} MB")
                print(f"  键名: {file_info['keys']}")
                
                for key, dataset_info in file_info['datasets'].items():
                    print(f"\n数据集 '{key}':")
                    print(f"  形状: {dataset_info['shape']}")
                    print(f"  数据类型: {dataset_info['dtype']}")
                    print(f"  内存大小: {dataset_info['size_mb']:.2f} MB")
                    
                    if 'min' in dataset_info:
                        print(f"  数值范围: [{dataset_info['min']:.6f}, {dataset_info['max']:.6f}]")
                        print(f"  均值: {dataset_info['mean']:.6f}")
                        print(f"  标准差: {dataset_info['std']:.6f}")
        
        # 只详细分析前5个文件，其余只统计
        if i >= 4:
            remaining = len(hdf5_files) - i - 1
            if remaining > 0:
                print(f"\n注意: 还有 {remaining} 个文件未详细分析")
            break
    
    # 显示摘要
    print("\n" + "=" * 60)
    print("数据摘要")
    print("=" * 60)
    print(f"总文件数: {len(analysis_results['files'])}")
    print(f"总大小: {analysis_results['total_size_gb']:.2f} GB")
    
    print(f"\n按子目录分组:")
    for subdir, info in analysis_results['subdirectories'].items():
        print(f"  {subdir}: {info['files']} 文件, {info['size_gb']:.2f} GB")
    
    # 生成训练元数据
    training_metadata = generate_training_metadata(analysis_results)
    
    # 保存分析结果
    output_file = "pdebench_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_results': analysis_results,
            'training_metadata': training_metadata
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析结果已保存到: {output_file}")
    
    # 显示训练建议
    print("\n" + "=" * 60)
    print("训练配置建议")
    print("=" * 60)
    
    for subdir, config in training_metadata['training_configs'].items():
        print(f"\n{subdir}:")
        print(f"  批大小: {config['batch_size']}")
        print(f"  工作进程: {config['num_workers']}")
        print(f"  建议内存: {config['recommended_memory_gb']} GB")
        print(f"  预估训练时间: {config['estimated_training_time_hours']} 小时")
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()