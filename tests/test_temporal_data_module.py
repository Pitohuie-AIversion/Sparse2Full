#!/usr/bin/env python3
"""
测试时序数据模块的数据处理逻辑
验证TemporalPDEBenchDataModule是否正确处理数据集
"""

import os
import sys
import torch
import h5py
import numpy as np
import yaml
import pytest
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append('.')

def test_hdf5_structure(data_path_resolver):
    """测试HDF5文件结构"""
    print("🔍 测试HDF5文件结构...")

    preferred_paths = [
        '2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
        'diffusion-reaction/2D_diff-react_NA_NA.h5',
        'DR2D/2D_diff-react_NA_NA.h5',
    ]
    data_path = data_path_resolver.resolve(preferred_paths)
    if not data_path:
        pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 成功打开HDF5文件: {data_path}")
            
            # 打印文件结构
            print("\n📁 HDF5文件结构:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  📄 {name}: {obj.shape} {obj.dtype}")
                    # 如果是小数据集，显示一些统计信息
                    if obj.size < 1000:
                        data = obj[...]
                        print(f"      范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
                else:
                    print(f"  📁 {name}/")
            
            f.visititems(print_structure)
            
            # 检查关键数据
            keys = list(f.keys())
            print(f"\n🔑 顶级键: {keys}")
            
            # 分析数据维度
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    data = f[key]
                    print(f"\n📊 {key} 详细分析:")
                    print(f"  - 形状: {data.shape}")
                    print(f"  - 数据类型: {data.dtype}")
                    print(f"  - 总大小: {data.size:,} 元素")
                    
                    # 采样一小部分数据进行分析
                    if len(data.shape) >= 3:
                        sample_indices = tuple(slice(0, min(3, s)) for s in data.shape)
                        sample = data[sample_indices]
                        print(f"  - 采样数据形状: {sample.shape}")
                        print(f"  - 数值范围: [{np.min(sample):.6f}, {np.max(sample):.6f}]")
                        print(f"  - 均值: {np.mean(sample):.6f}")
                        print(f"  - 标准差: {np.std(sample):.6f}")
                        
                        # 检查是否有异常值
                        if np.any(np.isnan(sample)):
                            print(f"  ⚠️ 包含NaN值")
                        if np.any(np.isinf(sample)):
                            print(f"  ⚠️ 包含无穷大值")
            
            return True
            
    except Exception as e:
        print(f"❌ HDF5文件读取失败: {e}")
        return False

def test_temporal_data_module(data_path_resolver):
    """测试时序数据模块"""
    print("\n🧪 测试时序数据模块...")
    
    try:
        from datasets.temporal_pdebench import TemporalPDEBenchDataModule
        
        # 解析数据路径
        preferred_paths = [
            '2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            'diffusion-reaction/2D_diff-react_NA_NA.h5',
            'DR2D/2D_diff-react_NA_NA.h5',
        ]
        data_path = data_path_resolver.resolve(preferred_paths)
        if not data_path:
            pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")

        # 创建配置
        config = DictConfig({
            'data_path': data_path,
            'dataset_name': '2D_diff-react_NA_NA',
            'batch_size': 2,
            'image_size': 128,
            'task': 'Crop',
            'crop_ratio': 0.2,
            'num_workers': 0,
            'pin_memory': False,
            'use_official_format': False,
            'keys': ['0000', '0001', '0002', '0003', '0004'],  # 使用前5个时间步
            'temporal': DictConfig({
                'T_in': 4,
                'T_out': 20,
                'dt': 0.1,
                'ar': DictConfig({
                    'teacher_forcing_ratio': 0.8,
                    'scheduled_sampling': True,
                    'sampling_decay': 0.99
                })
            })
        })
        
        print(f"📋 配置信息:")
        print(f"  - 数据路径: {config.data_path}")
        print(f"  - 批次大小: {config.batch_size}")
        print(f"  - 图像大小: {config.image_size}")
        print(f"  - 任务类型: {config.task}")
        print(f"  - 裁剪比例: {config.crop_ratio}")
        print(f"  - 时间步配置: T_in={config.temporal.T_in}, T_out={config.temporal.T_out}")
        print(f"  - 数据键: {config.keys}")
        
        # 创建数据模块
        print("\n🔧 创建数据模块...")
        data_module = TemporalPDEBenchDataModule(config)
        print("✅ 数据模块创建成功")
        
        # 获取数据加载器
        print("\n📦 获取数据加载器...")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"✅ 训练数据加载器: {len(train_loader)} 批次")
        print(f"✅ 验证数据加载器: {len(val_loader)} 批次")
        
        # 测试数据加载
        print("\n🎯 测试数据加载...")
        try:
            train_batch = next(iter(train_loader))
            print("✅ 成功加载训练批次")
            
            # 分析批次数据
            if isinstance(train_batch, dict):
                print("📊 批次数据结构 (字典格式):")
                for key, value in train_batch.items():
                    if torch.is_tensor(value):
                        print(f"  - {key}: {value.shape} ({value.dtype})")
                        print(f"    范围: [{torch.min(value):.6f}, {torch.max(value):.6f}]")
                    else:
                        print(f"  - {key}: {type(value)} - {value}")
            elif isinstance(train_batch, (list, tuple)):
                print(f"📊 批次数据结构 ({type(train_batch).__name__}格式):")
                for i, item in enumerate(train_batch):
                    if torch.is_tensor(item):
                        print(f"  - 项目{i}: {item.shape} ({item.dtype})")
                        print(f"    范围: [{torch.min(item):.6f}, {torch.max(item):.6f}]")
                    else:
                        print(f"  - 项目{i}: {type(item)} - {item}")
            else:
                print(f"📊 批次数据: {type(train_batch)} - {train_batch.shape if hasattr(train_batch, 'shape') else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ 导入时序数据模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 时序数据模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency(data_path_resolver):
    """测试数据一致性"""
    print("\n🔄 测试数据一致性...")
    
    try:
        from datasets.temporal_pdebench import TemporalPDEBenchBase
        preferred_paths = [
            '2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            'diffusion-reaction/2D_diff-react_NA_NA.h5',
            'DR2D/2D_diff-react_NA_NA.h5',
        ]
        data_path = data_path_resolver.resolve(preferred_paths)
        if not data_path:
            pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")
        
        # 创建简单的数据集实例
        dataset = TemporalPDEBenchBase(
            data_path=data_path,
            keys=['0000', '0001', '0002', '0003', '0004'],
            T_in=4,
            T_out=20,
            dt=0.1,
            split='train',
            normalize=True,
            image_size=128
        )
        
        print(f"✅ 数据集创建成功")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - 时间步数: {dataset.n_timesteps}")
        print(f"  - 时序样本数: {len(dataset.temporal_indices)}")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ 成功获取样本")
            
            if isinstance(sample, dict):
                print("📊 样本数据结构:")
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        print(f"  - {key}: {value.shape} ({value.dtype})")
                    else:
                        print(f"  - {key}: {type(value)}")
            else:
                print(f"📊 样本数据: {type(sample)}")
                if hasattr(sample, 'shape'):
                    print(f"  - 形状: {sample.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_temporal_sampling(data_path_resolver):
    """分析时序采样逻辑"""
    print("\n⏰ 分析时序采样逻辑...")
    
    try:
        # 直接分析HDF5文件的时序结构
        preferred_paths = [
            '2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            'diffusion-reaction/2D_diff-react_NA_NA.h5',
            'DR2D/2D_diff-react_NA_NA.h5',
        ]
        data_path = data_path_resolver.resolve(preferred_paths)
        if not data_path:
            pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")
        
        with h5py.File(data_path, 'r') as f:
            # 查找时序数据
            time_keys = [k for k in f.keys() if k.isdigit()]
            time_keys.sort()
            
            print(f"📅 发现时序键: {len(time_keys)} 个")
            print(f"  - 前10个: {time_keys[:10]}")
            print(f"  - 后10个: {time_keys[-10:]}")
            
            if len(time_keys) >= 24:  # T_in=4 + T_out=20
                print(f"✅ 时序数据足够 (需要24个，实际{len(time_keys)}个)")
                
                # 分析几个时间步的数据
                for i, key in enumerate(time_keys[:5]):
                    data = f[key]
                    print(f"  - 时间步{i} ({key}): {data.shape} {data.dtype}")
                    
                    # 检查数据质量
                    sample = data[:min(3, data.shape[0])]
                    print(f"    范围: [{np.min(sample):.6f}, {np.max(sample):.6f}]")
                    
                    if np.any(np.isnan(sample)):
                        print(f"    ⚠️ 包含NaN值")
                    if np.any(np.isinf(sample)):
                        print(f"    ⚠️ 包含无穷大值")
            else:
                print(f"❌ 时序数据不足 (需要24个，实际{len(time_keys)}个)")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 时序采样分析失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始时序数据模块测试...")
    print("=" * 60)
    
    results = {}
    
    # 1. 测试HDF5文件结构
    results['hdf5_structure'] = test_hdf5_structure()
    
    # 2. 分析时序采样逻辑
    results['temporal_sampling'] = analyze_temporal_sampling()
    
    # 3. 测试数据一致性
    results['data_consistency'] = test_data_consistency()
    
    # 4. 测试时序数据模块
    results['temporal_data_module'] = test_temporal_data_module()
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！数据模块工作正常。")
    else:
        print("⚠️ 部分测试失败，需要检查数据配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)