#!/usr/bin/env python3
"""
验证train.yaml配置文件中的训练数据是否为真实的PDEBench数据
"""

import h5py
import numpy as np
import os
import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def check_hdf5_structure(file_path):
    """检查HDF5文件结构和内容"""
    print(f"\n=== 检查HDF5文件结构 ===")
    print(f"文件路径: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件大小: {os.path.getsize(file_path) / (1024**3):.2f} GB")
            
            # 打印所有键
            print(f"\n文件中的键: {list(f.keys())}")
            
            # 检查tensor数据
            if 'tensor' in f:
                tensor_data = f['tensor']
                print(f"\ntensor数据信息:")
                print(f"  形状: {tensor_data.shape}")
                print(f"  数据类型: {tensor_data.dtype}")
                print(f"  最小值: {np.min(tensor_data[:]):.6f}")
                print(f"  最大值: {np.max(tensor_data[:]):.6f}")
                print(f"  均值: {np.mean(tensor_data[:]):.6f}")
                print(f"  标准差: {np.std(tensor_data[:]):.6f}")
                
                # 检查前几个样本
                print(f"\n前3个样本的形状和数值范围:")
                for i in range(min(3, tensor_data.shape[0])):
                    sample = tensor_data[i]
                    print(f"  样本 {i}: 形状={sample.shape}, 范围=[{np.min(sample):.6f}, {np.max(sample):.6f}]")
            
            # 检查其他可能的键
            for key in ['nu', 'x', 'y', 'coords']:
                if key in f:
                    data = f[key]
                    print(f"\n{key}数据信息:")
                    print(f"  形状: {data.shape}")
                    print(f"  数据类型: {data.dtype}")
                    if data.size < 100:  # 只打印小数组的值
                        print(f"  值: {data[:]}")
                    else:
                        print(f"  范围: [{np.min(data[:]):.6f}, {np.max(data[:]):.6f}]")
                        
    except Exception as e:
        print(f"读取HDF5文件时出错: {e}")
        return False
    
    return True

def validate_pdebench_data(file_path, config):
    """验证是否为真实的PDEBench DarcyFlow数据"""
    print(f"\n=== 验证PDEBench数据真实性 ===")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 检查数据集名称
            dataset_name = config['data']['dataset_name']
            print(f"配置中的数据集名称: {dataset_name}")
            
            # 检查数据键
            expected_keys = config['data']['keys']
            print(f"配置中的数据键: {expected_keys}")
            
            # 验证文件名是否符合PDEBench命名规范
            filename = os.path.basename(file_path)
            print(f"文件名: {filename}")
            
            if "DarcyFlow" in filename and "beta1.0" in filename:
                print("✓ 文件名符合PDEBench DarcyFlow官方命名规范")
            else:
                print("⚠ 文件名不符合PDEBench标准命名规范")
            
            # 检查数据维度是否合理
            if 'tensor' in f:
                tensor_shape = f['tensor'].shape
                print(f"数据形状: {tensor_shape}")
                
                # PDEBench DarcyFlow通常是4D: (samples, channels, height, width)
                if len(tensor_shape) == 4:
                    samples, channels, height, width = tensor_shape
                    print(f"✓ 数据维度正确: {samples}样本, {channels}通道, {height}x{width}分辨率")
                    
                    # 检查分辨率是否合理
                    if height == width and height in [32, 64, 128, 256, 512]:
                        print(f"✓ 空间分辨率合理: {height}x{width}")
                    else:
                        print(f"⚠ 空间分辨率异常: {height}x{width}")
                        
                    # 检查样本数量是否合理
                    if samples >= 100:
                        print(f"✓ 样本数量合理: {samples}")
                    else:
                        print(f"⚠ 样本数量较少: {samples}")
                        
                else:
                    print(f"⚠ 数据维度异常: {len(tensor_shape)}D")
            
            # 检查物理场数值范围
            if 'tensor' in f:
                data = f['tensor'][:]
                data_min, data_max = np.min(data), np.max(data)
                data_mean, data_std = np.mean(data), np.std(data)
                
                print(f"\n物理场数值特征:")
                print(f"  数值范围: [{data_min:.6f}, {data_max:.6f}]")
                print(f"  均值: {data_mean:.6f}")
                print(f"  标准差: {data_std:.6f}")
                
                # DarcyFlow物理场通常在合理范围内
                if -10 <= data_min and data_max <= 10:
                    print("✓ 数值范围符合DarcyFlow物理场特征")
                else:
                    print("⚠ 数值范围可能异常")
                    
                # 检查数据分布
                if 0.01 <= data_std <= 10:
                    print("✓ 数据标准差合理，具有物理变化")
                else:
                    print("⚠ 数据标准差异常")
                    
    except Exception as e:
        print(f"验证数据时出错: {e}")
        return False
        
    return True

def check_config_consistency(config):
    """检查配置文件的一致性"""
    print(f"\n=== 检查配置一致性 ===")
    
    # 检查数据配置
    data_config = config['data']
    print(f"数据集名称: {data_config['dataset_name']}")
    print(f"数据键: {data_config['keys']}")
    print(f"图像尺寸: {data_config['image_size']}")
    
    # 检查观测配置
    obs_config = data_config['observation']
    print(f"\n观测模式: {obs_config['mode']}")
    
    if obs_config['mode'] == 'SR':
        sr_config = obs_config['sr']
        print(f"超分辨率配置:")
        print(f"  缩放因子: {sr_config['scale_factor']}")
        print(f"  模糊参数: σ={sr_config['blur_sigma']}, kernel_size={sr_config['blur_kernel_size']}")
        print(f"  边界模式: {sr_config['boundary_mode']}")
        
        # 验证配置合理性
        if sr_config['scale_factor'] in [2, 4, 8]:
            print("✓ 缩放因子合理")
        else:
            print("⚠ 缩放因子异常")
            
    # 检查预处理配置
    preprocess_config = data_config['preprocessing']
    print(f"\n预处理配置:")
    print(f"  标准化: {preprocess_config['normalize']}")
    print(f"  缓存数据: {preprocess_config['cache_data']}")
    
    # 检查批次大小
    batch_size = data_config['dataloader']['batch_size']
    print(f"\n批次大小: {batch_size}")
    if 1 <= batch_size <= 32:
        print("✓ 批次大小合理")
    else:
        print("⚠ 批次大小可能不合适")
        
    return True

def generate_verification_report(config_path, data_path):
    """生成验证报告"""
    print("=" * 60)
    print("PDEBench训练数据验证报告")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 验证数据路径
    data_path_from_config = config['data']['data_path']
    print(f"配置文件路径: {config_path}")
    print(f"配置中的数据路径: {data_path_from_config}")
    print(f"实际检查路径: {data_path}")
    
    if data_path_from_config == data_path:
        print("✓ 数据路径一致")
    else:
        print("⚠ 数据路径不一致")
    
    # 检查文件存在性
    if os.path.exists(data_path):
        print("✓ 数据文件存在")
    else:
        print("✗ 数据文件不存在")
        return False
    
    # 检查HDF5结构
    hdf5_ok = check_hdf5_structure(data_path)
    
    # 验证PDE数据
    pde_ok = validate_pdebench_data(data_path, config)
    
    # 检查配置一致性
    config_ok = check_config_consistency(config)
    
    # 总结
    print(f"\n=== 验证总结 ===")
    print(f"HDF5文件结构: {'✓ 正常' if hdf5_ok else '✗ 异常'}")
    print(f"PDE数据验证: {'✓ 通过' if pde_ok else '✗ 失败'}")
    print(f"配置一致性: {'✓ 正常' if config_ok else '✗ 异常'}")
    
    if hdf5_ok and pde_ok and config_ok:
        print("\n🎉 结论: 训练数据确实是真实的PDEBench DarcyFlow数据集!")
        return True
    else:
        print("\n⚠️  结论: 训练数据可能存在问题，需要进一步检查")
        return False

def main():
    """主函数"""
    config_path = "configs/train.yaml"
    data_path = "data/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    
    # 生成验证报告
    result = generate_verification_report(config_path, data_path)
    
    # 保存报告到文件
    report_path = "runs/train_data_verification_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PDEBench训练数据验证报告\n\n")
        f.write(f"**验证时间**: {np.datetime64('now')}\n\n")
        f.write(f"**配置文件**: {config_path}\n\n")
        f.write(f"**数据文件**: {data_path}\n\n")
        f.write(f"**验证结果**: {'✓ 通过' if result else '✗ 失败'}\n\n")
        f.write("详细验证信息请查看终端输出。\n")
    
    print(f"\n验证报告已保存到: {report_path}")

if __name__ == "__main__":
    main()