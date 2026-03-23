#!/usr/bin/env python3
"""
测试智能数据集选择工具与现有训练流程的兼容性
"""

import os
import sys
import yaml
import h5py
import torch
from pathlib import Path


if __name__ != "__main__":
    import pytest

    pytest.skip(
        "脚本型数据集兼容性检查不作为 pytest 单测执行（依赖本地数据与 auto_generated 配置）。",
        allow_module_level=True,
    )

def test_dataset_file(data_path):
    """测试数据集文件的基本信息"""
    print(f"\n=== 测试数据集文件: {data_path} ===")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return False
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"✅ 文件可以正常打开")
            print(f"📊 数据集键名: {list(f.keys())}")
            
            # 检查每个键的形状
            for key in f.keys():
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"   - {key}: shape={shape}, dtype={dtype}")
                
                # 检查数据范围
                if len(shape) > 0:
                    data_sample = f[key][0] if shape[0] > 0 else f[key][()]
                    if hasattr(data_sample, 'min') and hasattr(data_sample, 'max'):
                        print(f"     数据范围: [{data_sample.min():.4f}, {data_sample.max():.4f}]")
            
            return True
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False

def test_config_file(config_path):
    """测试配置文件的格式和内容"""
    print(f"\n=== 测试配置文件: {config_path} ===")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件格式正确")
        
        # 检查必要的配置项
        required_sections = ['data', 'model', 'training', 'loss']
        for section in required_sections:
            if section in config:
                print(f"✅ 包含 {section} 配置")
            else:
                print(f"❌ 缺少 {section} 配置")
                return False
        
        # 检查数据配置
        data_config = config.get('data', {})
        data_path = data_config.get('data_path', '')
        if data_path and os.path.exists(data_path):
            print(f"✅ 数据路径有效: {data_path}")
        else:
            print(f"❌ 数据路径无效: {data_path}")
        
        # 检查模型配置
        model_config = config.get('model', {})
        model_name = model_config.get('name', '')
        if model_name:
            print(f"✅ 模型名称: {model_name}")
        else:
            print(f"❌ 缺少模型名称")
        
        return True
        
    except Exception as e:
        print(f"❌ 解析配置文件时出错: {e}")
        return False

def test_splits_files():
    """测试数据切分文件"""
    print(f"\n=== 测试数据切分文件 ===")
    
    splits_dir = Path("splits")
    required_files = ['train.txt', 'val.txt', 'test.txt']
    
    all_valid = True
    for filename in required_files:
        filepath = splits_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                indices = [line.strip() for line in f if line.strip()]
            print(f"✅ {filename}: {len(indices)} 个样本索引")
        else:
            print(f"❌ 缺少文件: {filepath}")
            all_valid = False
    
    return all_valid

def test_training_compatibility():
    """测试训练兼容性"""
    print(f"\n=== 测试训练兼容性 ===")
    
    # 检查生成的配置文件
    auto_configs_dir = Path("configs/auto_generated")
    if not auto_configs_dir.exists():
        print(f"❌ 自动生成的配置目录不存在: {auto_configs_dir}")
        return False
    
    config_files = list(auto_configs_dir.glob("*.yaml"))
    if not config_files:
        print(f"❌ 没有找到自动生成的配置文件")
        return False
    
    print(f"✅ 找到 {len(config_files)} 个自动生成的配置文件")
    
    # 测试第一个配置文件
    test_config = config_files[0]
    print(f"📝 测试配置文件: {test_config.name}")
    
    success = test_config_file(test_config)
    if success:
        # 检查对应的数据文件
        with open(test_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        data_path = config.get('data', {}).get('data_path', '')
        if data_path:
            test_dataset_file(data_path)
    
    return success

def main():
    """主测试函数"""
    print("🚀 开始测试智能数据集选择工具的兼容性")
    
    # 测试基础文件结构
    test_results = []
    
    # 1. 测试数据切分文件
    test_results.append(("数据切分文件", test_splits_files()))
    
    # 2. 测试训练兼容性
    test_results.append(("训练兼容性", test_training_compatibility()))
    
    # 3. 测试批量训练脚本
    batch_script = Path("batch_train_selected_datasets.py")
    if batch_script.exists():
        print(f"✅ 批量训练脚本存在: {batch_script}")
        test_results.append(("批量训练脚本", True))
    else:
        print(f"❌ 批量训练脚本不存在: {batch_script}")
        test_results.append(("批量训练脚本", False))
    
    # 4. 测试数据集对比报告
    report_file = Path("dataset_comparison_report.md")
    if report_file.exists():
        print(f"✅ 数据集对比报告存在: {report_file}")
        test_results.append(("数据集对比报告", True))
    else:
        print(f"❌ 数据集对比报告不存在: {report_file}")
        test_results.append(("数据集对比报告", False))
    
    # 汇总测试结果
    print(f"\n{'='*50}")
    print(f"📊 测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！智能数据集选择工具与现有训练流程完全兼容。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试和修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
