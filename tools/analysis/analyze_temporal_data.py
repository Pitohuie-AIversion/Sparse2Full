#!/usr/bin/env python3
"""
分析时序数据集的详细结构和配置问题
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append('.')

def analyze_hdf5_structure():
    """分析HDF5文件结构"""
    print("🔍 分析HDF5文件结构...")
    
    data_path = "data/DR2D/2D_diff-react_NA_NA.h5"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 成功打开HDF5文件: {data_path}")
            
            # 获取所有时序键
            time_keys = [k for k in f.keys() if k.isdigit()]
            time_keys.sort()
            
            print(f"\n📅 时序数据分析:")
            print(f"  - 总时间步数: {len(time_keys)}")
            print(f"  - 时间步范围: {time_keys[0]} ~ {time_keys[-1]}")
            
            # 分析数据结构
            first_key = time_keys[0]
            first_group = f[first_key]
            
            if isinstance(first_group, h5py.Group):
                sub_keys = list(first_group.keys())
                print(f"  - 每个时间步包含: {sub_keys}")
                
                # 分析data数据集
                if 'data' in sub_keys:
                    data_dataset = first_group['data']
                    print(f"\n📊 数据集详细信息:")
                    print(f"  - 数据形状: {data_dataset.shape}")
                    print(f"  - 数据类型: {data_dataset.dtype}")
                    print(f"  - 维度解释: (样本数, 高度, 宽度, 通道数)")
                    
                    # 分析数据范围
                    sample_data = data_dataset[:5]  # 取前5个样本
                    print(f"  - 前5个样本的数值范围: [{np.min(sample_data):.6f}, {np.max(sample_data):.6f}]")
                    print(f"  - 前5个样本的均值: {np.mean(sample_data):.6f}")
                    print(f"  - 前5个样本的标准差: {np.std(sample_data):.6f}")
                    
                    # 检查数据质量
                    if np.any(np.isnan(sample_data)):
                        print(f"  ⚠️ 数据包含NaN值")
                    if np.any(np.isinf(sample_data)):
                        print(f"  ⚠️ 数据包含无穷大值")
                    
                    # 分析通道数据
                    if len(data_dataset.shape) == 4 and data_dataset.shape[3] == 2:
                        print(f"\n🔬 通道分析:")
                        for ch in range(2):
                            ch_data = sample_data[:, :, :, ch]
                            print(f"  - 通道{ch}: 范围[{np.min(ch_data):.6f}, {np.max(ch_data):.6f}], 均值{np.mean(ch_data):.6f}")
                
                # 分析grid数据集
                if 'grid' in sub_keys:
                    grid_dataset = first_group['grid']
                    print(f"\n🗺️ 网格信息:")
                    print(f"  - 网格形状: {grid_dataset.shape}")
                    print(f"  - 网格数据类型: {grid_dataset.dtype}")
                    
                    if grid_dataset.size < 1000:  # 如果网格数据不大，显示一些内容
                        grid_data = grid_dataset[...]
                        print(f"  - 网格数值范围: [{np.min(grid_data):.6f}, {np.max(grid_data):.6f}]")
            
            # 检查时序配置兼容性
            print(f"\n⏰ 时序配置兼容性检查:")
            T_in, T_out = 4, 20
            required_timesteps = T_in + T_out  # 24
            
            print(f"  - 配置要求: T_in={T_in}, T_out={T_out}")
            print(f"  - 需要时间步数: {required_timesteps}")
            print(f"  - 实际时间步数: {len(time_keys)}")
            
            if len(time_keys) >= required_timesteps:
                print(f"  ✅ 时序数据充足")
                
                # 检查配置中的keys是否合理
                config_keys = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', 
                              '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
                
                print(f"  - 配置中的keys数量: {len(config_keys)}")
                print(f"  - 配置keys范围: {config_keys[0]} ~ {config_keys[-1]}")
                
                # 检查keys是否都存在
                missing_keys = [k for k in config_keys if k not in time_keys]
                if missing_keys:
                    print(f"  ⚠️ 缺失的keys: {missing_keys}")
                else:
                    print(f"  ✅ 所有配置keys都存在")
                    
                # 检查keys数量是否与T_in+T_out匹配
                if len(config_keys) >= required_timesteps:
                    print(f"  ✅ 配置keys数量满足时序要求")
                else:
                    print(f"  ⚠️ 配置keys数量不足，需要{required_timesteps}个，实际{len(config_keys)}个")
            else:
                print(f"  ❌ 时序数据不足")
            
            return True
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_dataset():
    """测试时序数据集加载"""
    print("\n🧪 测试时序数据集加载...")
    
    try:
        from datasets.temporal_pdebench import TemporalPDEBenchBase
        
        # 创建数据集实例
        dataset = TemporalPDEBenchBase(
            data_path='data/DR2D/2D_diff-react_NA_NA.h5',
            keys=['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', 
                  '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020'],
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
            print(f"\n🎯 测试样本获取...")
            sample = dataset[0]
            print(f"✅ 成功获取第一个样本")
            
            if isinstance(sample, dict):
                print("📊 样本数据结构:")
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        print(f"  - {key}: {value.shape} ({value.dtype})")
                        print(f"    数值范围: [{torch.min(value):.6f}, {torch.max(value):.6f}]")
                        print(f"    均值: {torch.mean(value):.6f}")
                        print(f"    标准差: {torch.std(value):.6f}")
                    else:
                        print(f"  - {key}: {type(value)} - {value}")
            
            # 测试多个样本
            print(f"\n🔄 测试多个样本...")
            for i in range(min(3, len(dataset))):
                try:
                    sample = dataset[i]
                    print(f"  ✅ 样本{i}: 成功")
                except Exception as e:
                    print(f"  ❌ 样本{i}: 失败 - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 时序数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_module():
    """测试数据模块"""
    print("\n🔧 测试数据模块...")
    
    try:
        from datasets.temporal_pdebench import TemporalPDEBenchDataModule
        
        # 创建配置
        config = DictConfig({
            'data_path': 'data/DR2D/2D_diff-react_NA_NA.h5',
            'dataset_name': '2D_diff-react_NA_NA',
            'batch_size': 2,
            'image_size': 128,
            'task': 'Crop',
            'crop_ratio': 0.2,
            'num_workers': 0,
            'pin_memory': False,
            'use_official_format': False,
            'keys': ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', 
                     '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020'],
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
        
        print(f"📋 数据模块配置:")
        print(f"  - 数据路径: {config.data_path}")
        print(f"  - 任务类型: {config.task}")
        print(f"  - 图像大小: {config.image_size}")
        print(f"  - 批次大小: {config.batch_size}")
        print(f"  - 时序配置: T_in={config.temporal.T_in}, T_out={config.temporal.T_out}")
        
        # 创建数据模块
        data_module = TemporalPDEBenchDataModule(config)
        print("✅ 数据模块创建成功")
        
        # 获取数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"✅ 训练数据加载器: {len(train_loader)} 批次")
        print(f"✅ 验证数据加载器: {len(val_loader)} 批次")
        
        # 测试数据加载
        print(f"\n🎯 测试批次数据加载...")
        try:
            train_batch = next(iter(train_loader))
            print("✅ 成功加载训练批次")
            
            # 分析批次数据
            if isinstance(train_batch, dict):
                print("📊 批次数据结构:")
                for key, value in train_batch.items():
                    if torch.is_tensor(value):
                        print(f"  - {key}: {value.shape} ({value.dtype})")
                        print(f"    数值范围: [{torch.min(value):.6f}, {torch.max(value):.6f}]")
                        print(f"    均值: {torch.mean(value):.6f}")
                        print(f"    标准差: {torch.std(value):.6f}")
                    else:
                        print(f"  - {key}: {type(value)} - {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ 批次数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 数据模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_visualization_issues():
    """分析可视化问题"""
    print("\n🎨 分析可视化问题...")
    
    # 检查可视化相关的配置
    print("📋 可视化配置检查:")
    print("  - 数据格式: HDF5 Group结构 (时间步/data/样本)")
    print("  - 数据维度: (101, 128, 128, 2) - (样本数, 高度, 宽度, 通道数)")
    print("  - 通道数: 2 (可能是u和v分量，或者浓度和反应物)")
    print("  - 图像大小: 128x128")
    
    print("\n🔍 可能的可视化问题:")
    print("  1. 数据归一化问题:")
    print("     - 原始数据范围较大，可能需要适当的归一化")
    print("     - 不同通道的数值范围可能差异很大")
    
    print("  2. 时序采样问题:")
    print("     - T_in=4, T_out=20 的配置可能导致时序跳跃过大")
    print("     - dt=0.1 的时间步长可能不匹配数据的实际时间间隔")
    
    print("  3. 裁剪问题:")
    print("     - crop_ratio=0.2 可能裁剪掉重要的边界信息")
    print("     - 裁剪位置可能不合理")
    
    print("  4. 数据预处理问题:")
    print("     - 可能需要检查数据的物理意义和单位")
    print("     - 不同时间步之间的连续性")
    
    return True

def generate_recommendations():
    """生成修正建议"""
    print("\n💡 修正建议:")
    
    print("1. 数据配置优化:")
    print("   - 检查keys配置是否覆盖了足够的时间范围")
    print("   - 考虑减少T_out或增加时间步间隔")
    print("   - 验证dt参数是否与数据的实际时间步长匹配")
    
    print("2. 可视化改进:")
    print("   - 分别可视化不同通道的数据")
    print("   - 检查数据的物理意义和合理的数值范围")
    print("   - 添加时序连续性检查")
    
    print("3. 数据预处理:")
    print("   - 考虑使用更合适的归一化方法")
    print("   - 检查裁剪策略是否保留了重要信息")
    print("   - 添加数据质量检查")
    
    print("4. 调试建议:")
    print("   - 可视化原始数据的时序变化")
    print("   - 检查模型输入输出的数据格式")
    print("   - 验证数据加载器的采样逻辑")
    
    return True

def main():
    """主函数"""
    print("🚀 开始时序数据分析...")
    print("=" * 80)
    
    results = {}
    
    # 1. 分析HDF5文件结构
    results['hdf5_analysis'] = analyze_hdf5_structure()
    
    # 2. 测试时序数据集
    results['dataset_test'] = test_temporal_dataset()
    
    # 3. 测试数据模块
    results['data_module_test'] = test_data_module()
    
    # 4. 分析可视化问题
    results['visualization_analysis'] = analyze_visualization_issues()
    
    # 5. 生成修正建议
    results['recommendations'] = generate_recommendations()
    
    # 总结结果
    print("\n" + "=" * 80)
    print("📊 分析结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 成功" if result else "❌ 失败"
        print(f"  - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 项分析完成")
    
    if passed >= 3:  # 至少3项成功
        print("🎉 数据分析基本完成，已识别潜在问题并提供修正建议。")
    else:
        print("⚠️ 数据分析遇到问题，需要进一步检查配置。")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)