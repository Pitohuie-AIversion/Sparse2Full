"""最小化训练测试脚本

用于测试训练脚本的基本功能，不依赖真实数据
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append('.')

def create_mock_data():
    """创建模拟数据"""
    # 创建数据目录
    data_dir = Path('data/pdebench')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模拟数据文件
    mock_data = {
        'u': np.random.randn(10, 64, 64).astype(np.float32),  # 10个样本
        'x': np.linspace(0, 1, 64).astype(np.float32),
        'y': np.linspace(0, 1, 64).astype(np.float32),
        't': np.linspace(0, 1, 10).astype(np.float32)
    }
    
    # 保存为npz文件
    np.savez(data_dir / 'test_data.npz', **mock_data)
    
    # 创建数据分割文件
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # 创建简单的分割
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('test_data.npz\n')
    
    with open(splits_dir / 'val.txt', 'w') as f:
        f.write('test_data.npz\n')
        
    with open(splits_dir / 'test.txt', 'w') as f:
        f.write('test_data.npz\n')
    
    print(f"✓ 模拟数据已创建在 {data_dir}")

def test_training_imports():
    """测试训练脚本的导入"""
    print("测试训练脚本导入...")
    
    try:
        # 测试关键模块导入
        from datasets.pdebench import PDEBenchDataModule
        from models.swin_unet import SwinUNet
        from ops.total_loss import TotalLoss
        from omegaconf import OmegaConf
        
        print("✓ 所有关键模块导入成功")
        return True
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """测试配置创建"""
    print("测试配置创建...")
    
    try:
        from omegaconf import OmegaConf
        
        # 创建最小配置
        config = OmegaConf.create({
            'device': {
                'use_cuda': False
            },
            'data': {
                'data_dir': 'data/pdebench',
                'task': 'sr',
                'scale_factor': 4,
                'image_size': 64,
                'keys': ['u'],
                'dataloader': {
                    'batch_size': 1,
                    'num_workers': 0,
                    'pin_memory': False
                }
            },
            'model': {
                'name': 'SwinUNet',
                'params': {
                    'in_channels': 3,
                    'out_channels': 3,
                    'img_size': 64,
                    'kwargs': {
                        'patch_size': 4,
                        'embed_dim': 48,
                        'depths': [1, 1, 1, 1],
                        'num_heads': [2, 4, 8, 16],
                        'window_size': 8,
                        'mlp_ratio': 4.0,
                        'drop_rate': 0.0,
                        'attn_drop_rate': 0.0,
                        'drop_path_rate': 0.1
                    }
                }
            },
            'training': {
                'epochs': 1,
                'log_interval': 1,
                'save_interval': 1,
                'plot_interval': 1,
                'optimizer': {
                    'name': 'AdamW',
                    'lr': 1e-3,
                    'weight_decay': 1e-4
                },
                'scheduler': {
                    'name': 'CosineAnnealingLR',
                    'T_max': 1
                },
                'loss_weights': {
                    'reconstruction': 1.0,
                    'spectral': 0.5,
                    'data_consistency': 1.0
                }
            },
            'loss': {
                'reconstruction_weight': 1.0,
                'spectral_weight': 0.5,
                'data_consistency_weight': 1.0
            }
        })
        
        print("✓ 配置创建成功")
        return config
        
    except Exception as e:
        print(f"✗ 配置创建失败: {e}")
        return None

def test_data_module():
    """测试数据模块"""
    print("测试数据模块...")
    
    try:
        from datasets.pdebench import PDEBenchDataModule
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'data_dir': 'data/pdebench',
            'task': 'sr',
            'scale_factor': 4,
            'image_size': 64,
            'keys': ['u'],
            'dataloader': {
                'batch_size': 1,
                'num_workers': 0,
                'pin_memory': False
            }
        })
        
        data_module = PDEBenchDataModule(config)
        
        # 检查数据文件是否存在
        if not Path('data/pdebench/test_data.npz').exists():
            print("⚠ 数据文件不存在，创建模拟数据")
            create_mock_data()
        
        print("✓ 数据模块创建成功")
        return True
        
    except Exception as e:
        print(f"✗ 数据模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    
    try:
        from models.swin_unet import SwinUNet
        
        model_config = {
            'in_channels': 3,
            'out_channels': 3,
            'img_size': 64,
            'patch_size': 4,
            'embed_dim': 48,
            'depths': [1, 1, 1, 1],
            'num_heads': [2, 4, 8, 16],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1
        }
        
        model = SwinUNet(**model_config)
        
        print("✓ 模型创建成功")
        return model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_loss_function():
    """测试损失函数"""
    print("测试损失函数...")
    
    try:
        from ops.total_loss import TotalLoss
        from omegaconf import OmegaConf
        
        loss_config = OmegaConf.create({
            'reconstruction_weight': 1.0,
            'spectral_weight': 0.5,
            'data_consistency_weight': 1.0
        })
        
        loss_fn = TotalLoss(loss_config)
        
        print("✓ 损失函数创建成功")
        return loss_fn
        
    except Exception as e:
        print(f"✗ 损失函数创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("=" * 60)
    print("PDEBench 稀疏观测重建系统 - 训练脚本最小化测试")
    print("=" * 60)
    
    # 创建模拟数据
    create_mock_data()
    
    # 测试各个组件
    tests = [
        ("训练脚本导入", test_training_imports),
        ("配置创建", test_config_creation),
        ("数据模块", test_data_module),
        ("模型创建", test_model_creation),
        ("损失函数", test_loss_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"开始 {test_name} 测试...")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            if result is not None and result is not False:
                results.append((test_name, True))
                print(f"✓ {test_name} 测试通过")
            else:
                results.append((test_name, False))
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有组件测试通过！可以尝试运行完整训练")
        
        # 提供运行建议
        print("\n建议的训练命令:")
        print("F:\\ProgramData\\anaconda3\\python.exe tools/train.py \\")
        print("  device.use_cuda=false \\")
        print("  training.epochs=1 \\")
        print("  training.log_interval=1 \\")
        print("  data.dataloader.batch_size=1 \\")
        print("  model.params.kwargs.depths=[1,1,1,1] \\")
        print("  model.params.kwargs.num_heads=[2,4,8,16] \\")
        print("  model.params.kwargs.embed_dim=48")
        
    else:
        print("⚠ 部分组件测试失败，需要进一步调试")

if __name__ == "__main__":
    main()