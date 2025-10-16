#!/usr/bin/env python3
"""
测试数据一致性检查脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        # 测试基础模块导入
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        # 测试项目模块导入
        from datasets.pdebench import PDEBenchDataModule
        print("✅ PDEBenchDataModule imported")
        
        from ops.degradation import apply_degradation_operator
        print("✅ apply_degradation_operator imported")
        
        from utils.reproducibility import set_seed
        print("✅ set_seed imported")
        
        # 测试一致性检查器导入
        from tools.check_dc_equivalence import DataConsistencyChecker
        print("✅ DataConsistencyChecker imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        from omegaconf import DictConfig
        from tools.check_dc_equivalence import DataConsistencyChecker
        
        # 创建基本配置
        config = DictConfig({
            'consistency_check': {
                'tolerance': 1e-8,
                'num_samples': 10,
                'random_seed': 42
            },
            'task': {
                'super_resolution': {
                    'scale_factors': [4],
                    'blur_sigma': 1.0,
                    'blur_kernel_size': 5,
                    'boundary_mode': 'mirror'
                }
            }
        })
        
        device = torch.device('cpu')
        checker = DataConsistencyChecker(config, device)
        print("✅ DataConsistencyChecker created successfully")
        
        # 测试默认参数获取
        default_params = checker._get_default_task_params()
        print(f"✅ Default task params: {default_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    try:
        from omegaconf import DictConfig
        from datasets.pdebench import PDEBenchDataModule
        
        # 检查数据文件是否存在
        data_path = Path("data/pdebench")
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_path}")
            return False
        
        # 列出数据文件
        data_files = list(data_path.glob("*.h5"))
        if not data_files:
            print(f"❌ No HDF5 data files found in {data_path}")
            return False
        
        print(f"✅ Found {len(data_files)} data files")
        
        # 创建数据模块配置
        data_config = DictConfig({
            'data_path': str(data_path),
            'task': 'sr',
            'batch_size': 1,
            'num_workers': 0,
            'normalize': True,
            'image_size': 256
        })
        
        # 创建数据模块
        data_module = PDEBenchDataModule(data_config)
        data_module.setup()
        print("✅ PDEBenchDataModule setup successful")
        
        # 获取数据加载器
        train_loader = data_module.train_dataloader()
        print(f"✅ Train dataloader created with {len(train_loader)} batches")
        
        # 测试获取一个batch
        batch = next(iter(train_loader))
        print(f"✅ Sample batch loaded:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("PDEBench数据一致性检查测试")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    tests = [
        ("模块导入测试", test_imports),
        ("基本功能测试", test_basic_functionality),
        ("数据加载测试", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！数据一致性检查系统准备就绪。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)