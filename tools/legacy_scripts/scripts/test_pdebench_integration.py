#!/usr/bin/env python3
"""
PDEBench数据集集成测试脚本

验证官方PDEBench数据集的正确集成，包括：
1. 数据集加载
2. 数据格式验证
3. 观测生成验证
4. 数据一致性验证
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datasets.pdebench import PDEBenchSR, PDEBenchCrop

def test_sr_dataset():
    """测试SR数据集"""
    print("=" * 50)
    print("测试PDEBench SR数据集")
    print("=" * 50)
    
    try:
        # 测试官方格式数据集
        dataset = PDEBenchSR(
            data_path='data/pdebench/official/2D_diff-react_NA_NA.h5',
            keys=['u'], 
            scale=4,
            split='train', 
            use_official_format=True
        )
        
        print(f"✓ SR数据集加载成功")
        print(f"  - 数据集大小: {len(dataset)}")
        
        # 测试获取样本
        sample = dataset[0]
        print(f"  - 样本键: {list(sample.keys())}")
        print(f"  - 目标形状: {sample['target'].shape}")
        print(f"  - 观测形状: {sample['observation'].shape}")
        print(f"  - 基线形状: {sample['baseline'].shape}")
        print(f"  - H算子参数: {sample['h_params']}")
        
        # 验证数据类型和值域
        target = sample['target']
        observation = sample['observation']
        print(f"  - 目标数据类型: {target.dtype}")
        print(f"  - 目标值域: [{target.min():.4f}, {target.max():.4f}]")
        print(f"  - 观测值域: [{observation.min():.4f}, {observation.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ SR数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_crop_dataset():
    """测试Crop数据集"""
    print("\n" + "=" * 50)
    print("测试PDEBench Crop数据集")
    print("=" * 50)
    
    try:
        # 测试官方格式数据集
        dataset = PDEBenchCrop(
            data_path='data/pdebench/official/2D_diff-react_NA_NA.h5',
            keys=['u'], 
            crop_size=(128, 128),
            split='train', 
            use_official_format=True
        )
        
        print(f"✓ Crop数据集加载成功")
        print(f"  - 数据集大小: {len(dataset)}")
        
        # 测试获取样本
        sample = dataset[0]
        print(f"  - 样本键: {list(sample.keys())}")
        print(f"  - 目标形状: {sample['target'].shape}")
        print(f"  - 观测形状: {sample['observation'].shape}")
        print(f"  - H算子参数: {sample['h_params']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Crop数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """测试数据一致性"""
    print("\n" + "=" * 50)
    print("测试数据一致性")
    print("=" * 50)
    
    try:
        from ops.degradation import apply_degradation_operator
        
        # 创建测试数据
        test_data = torch.randn(1, 1, 256, 256)
        
        # SR一致性测试
        sr_params = {
            'task': 'SR',
            'scale': 4,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        sr_result = apply_degradation_operator(test_data, sr_params)
        print(f"✓ SR观测生成成功，输出形状: {sr_result.shape}")
        
        # Crop一致性测试
        crop_params = {
            'task': 'Crop',
            'crop_size': (128, 128),
            'patch_align': 8,
            'boundary': 'mirror'
        }
        
        crop_result = apply_degradation_operator(test_data, crop_params)
        print(f"✓ Crop观测生成成功，输出形状: {crop_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("PDEBench数据集集成测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # 测试SR数据集
    if test_sr_dataset():
        success_count += 1
    
    # 测试Crop数据集
    if test_crop_dataset():
        success_count += 1
    
    # 测试数据一致性
    if test_data_consistency():
        success_count += 1
    
    # 输出总结
    print("\n" + "=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("✓ 所有测试通过！PDEBench数据集集成成功")
        return True
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)