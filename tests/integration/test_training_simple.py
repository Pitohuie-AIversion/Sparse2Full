"""简化的训练测试脚本

用于验证基本的训练流程是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

def test_basic_training():
    """测试基本训练流程"""
    print("开始基本训练测试...")
    
    # 设置设备
    device = torch.device('cpu')  # 强制使用CPU避免设备问题
    print(f"使用设备: {device}")
    
    # 创建简单的测试数据
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    # 模拟输入数据
    baseline = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)
    
    print(f"输入数据形状: {baseline.shape}")
    print(f"目标数据形状: {target.shape}")
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, channels, 3, padding=1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    model = SimpleModel().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 简单的损失函数
    criterion = nn.MSELoss()
    
    # 训练几个步骤
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(baseline)
        loss = criterion(pred, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"步骤 {step+1}, 损失: {loss.item():.6f}")
    
    print("✓ 基本训练测试通过")
    return True

def test_data_loading():
    """测试数据加载"""
    print("\n开始数据加载测试...")
    
    try:
        from datasets.pdebench import PDEBenchDataModule
        from omegaconf import OmegaConf
        
        # 创建最小配置
        config = OmegaConf.create({
            'data_dir': 'data/pdebench',
            'task': 'sr',
            'scale_factor': 4,
            'image_size': 64,
            'keys': ['u'],
            'dataloader': {
                'batch_size': 2,
                'num_workers': 0,
                'pin_memory': False
            }
        })
        
        # 创建数据模块
        data_module = PDEBenchDataModule(config)
        
        # 检查是否有数据文件
        data_dir = Path('data/pdebench')
        if not data_dir.exists():
            print("⚠ 数据目录不存在，跳过数据加载测试")
            return True
            
        print("✓ 数据加载模块创建成功")
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n开始模型创建测试...")
    
    try:
        from models.swin_unet import SwinUNet
        
        # 创建模型配置
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
        
        # 测试前向传播
        device = torch.device('cpu')
        model = model.to(device)
        
        test_input = torch.randn(1, 3, 64, 64, device=device)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"模型输入形状: {test_input.shape}")
        print(f"模型输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        print("✓ 模型创建测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False

def test_loss_function():
    """测试损失函数"""
    print("\n开始损失函数测试...")
    
    try:
        from ops.total_loss import TotalLoss
        from omegaconf import OmegaConf
        
        # 创建损失函数配置
        loss_config = OmegaConf.create({
            'reconstruction_weight': 1.0,
            'spectral_weight': 0.5,
            'data_consistency_weight': 1.0
        })
        
        loss_fn = TotalLoss(loss_config)
        
        # 创建测试数据
        device = torch.device('cpu')
        batch_size = 2
        channels = 3
        height, width = 64, 64
        
        pred = torch.randn(batch_size, channels, height, width, device=device)
        target = torch.randn(batch_size, channels, height, width, device=device)
        
        # 创建观测数据
        obs_data = {
            'baseline': torch.randn(batch_size, channels, height, width, device=device),
            'coords': None,
            'mask': None,
            'observation': torch.randn(batch_size, channels, height//4, width//4, device=device),
            'h_params': {'task': 'sr', 'scale_factor': 4}
        }
        
        # 创建归一化统计量
        norm_stats = {
            'mean': torch.zeros(channels, device=device),
            'std': torch.ones(channels, device=device)
        }
        
        # 创建配置
        cfg = OmegaConf.create({
            'train': {
                'loss_weights': {
                    'reconstruction': 1.0,
                    'spectral': 0.5,
                    'data_consistency': 1.0
                },
                'spectral_loss': {
                    'low_freq_modes': 16,
                    'use_rfft': True,
                    'normalize': True
                }
            },
            'data': {
                'keys': ['u']  # 直接使用列表，不是方法
            }
        })
        
        # 计算损失
        loss_dict = loss_fn(pred, target, obs_data, norm_stats, cfg)
        
        print(f"损失组件: {list(loss_dict.keys())}")
        print(f"总损失: {loss_dict['total_loss'].item():.6f}")
        
        print("✓ 损失函数测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("PDEBench 稀疏观测重建系统 - 基础功能测试")
    print("=" * 50)
    
    tests = [
        ("基本训练", test_basic_training),
        ("数据加载", test_data_loading),
        ("模型创建", test_model_creation),
        ("损失函数", test_loss_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有基础功能测试通过！")
    else:
        print("⚠ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main()