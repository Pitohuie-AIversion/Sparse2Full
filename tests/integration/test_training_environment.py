"""训练环境完整性测试

验证PDEBench训练环境的各个组件是否正常工作
包括数据加载、模型初始化、损失计算、GPU资源等
"""

import os
import sys
import time
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets import PDEBenchDataModule
from models import create_model
from ops.loss import TotalLoss


def test_environment():
    """测试训练环境"""
    print("=" * 60)
    print("PDEBench训练环境完整性测试")
    print("=" * 60)
    
    # 1. 测试基础环境
    print("\n1. 基础环境检查:")
    print(f"  - Python版本: {sys.version}")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA设备数: {torch.cuda.device_count()}")
        print(f"  - 当前设备: {torch.cuda.get_device_name()}")
        print(f"  - 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - 使用设备: {device}")
    
    # 2. 测试数据路径
    print("\n2. 数据路径检查:")
    data_path = "data/2D"
    if os.path.exists(data_path):
        print(f"  ✓ 数据路径存在: {data_path}")
        subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        print(f"  - 子目录数量: {len(subdirs)}")
        print(f"  - 子目录: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
    else:
        print(f"  ✗ 数据路径不存在: {data_path}")
        return False
    
    # 3. 测试配置加载
    print("\n3. 配置加载测试:")
    try:
        with initialize(version_base=None, config_path="configs"):
            config = compose(config_name="config")
        print("  ✓ 配置加载成功")
        print(f"  - 实验名称: {config.experiment.name}")
        print(f"  - 数据集: {config.data._target_}")
        print(f"  - 模型: {config.model.name}")
        print(f"  - 训练epochs: {config.train.epochs}")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return False
    
    # 4. 测试数据模块
    print("\n4. 数据模块测试:")
    try:
        data_module = PDEBenchDataModule(config.data)
        data_module.setup()
        print("  ✓ 数据模块初始化成功")
        
        train_loader = data_module.train_dataloader()
        print(f"  - 训练批次数: {len(train_loader)}")
        print(f"  - 批次大小: {config.data.dataloader.batch_size}")
        
        # 测试数据加载
        batch = next(iter(train_loader))
        print("  ✓ 数据加载成功")
        print(f"  - 批次键: {list(batch.keys())}")
        
        # 检查张量形状
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"    {key}: {value.shape} ({value.dtype})")
        
    except Exception as e:
        print(f"  ✗ 数据模块测试失败: {e}")
        traceback.print_exc()
        return False
    
    # 5. 测试模型创建
    print("\n5. 模型创建测试:")
    try:
        model = create_model(config)
        model = model.to(device)
        print("  ✓ 模型创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"  ✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False
    
    # 6. 测试前向传播
    print("\n6. 前向传播测试:")
    try:
        # 获取输入张量
        input_key = None
        for key in ['baseline', 'input', 'lr']:
            if key in batch:
                input_key = key
                break
        
        if input_key is None:
            print(f"  ✗ 未找到合适的输入键，可用键: {list(batch.keys())}")
            return False
        
        input_tensor = batch[input_key].to(device)
        print(f"  - 使用输入键: {input_key}")
        print(f"  - 输入形状: {input_tensor.shape}")
        
        with torch.no_grad():
            pred = model(input_tensor)
        
        print("  ✓ 前向传播成功")
        print(f"  - 输出形状: {pred.shape}")
        print(f"  - 输出范围: [{pred.min():.4f}, {pred.max():.4f}]")
        
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        traceback.print_exc()
        return False
    
    # 7. 测试损失计算
    print("\n7. 损失计算测试:")
    try:
        # 创建损失函数
        loss_fn = TotalLoss(
            rec_weight=config.train.loss_weights.reconstruction,
            spec_weight=config.train.loss_weights.spectral,
            dc_weight=config.train.loss_weights.data_consistency
        )
        print("  ✓ 损失函数创建成功")
        
        # 获取目标张量
        target_key = None
        for key in ['target', 'gt', 'hr']:
            if key in batch:
                target_key = key
                break
        
        if target_key is None:
            print(f"  ✗ 未找到目标张量键，可用键: {list(batch.keys())}")
            return False
        
        target_tensor = batch[target_key].to(device)
        print(f"  - 使用目标键: {target_key}")
        
        # 获取观测张量
        observation_key = None
        for key in ['observation', 'lr_observation', 'baseline']:
            if key in batch:
                observation_key = key
                break
        
        if observation_key is None:
            print(f"  ✗ 未找到观测张量键")
            return False
        
        observation_tensor = batch[observation_key].to(device)
        print(f"  - 使用观测键: {observation_key}")
        
        # 构建任务参数
        task_params = batch.get('task_params', {
            'task': 'SR',
            'scale': 4,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        })
        
        # 计算损失
        total_loss, loss_dict = loss_fn(pred, target_tensor, observation_tensor, task_params)
        
        print("  ✓ 损失计算成功")
        for loss_name, loss_value in loss_dict.items():
            if torch.is_tensor(loss_value):
                print(f"    {loss_name}: {loss_value.item():.6f}")
        
        # 检查损失是否为NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  ⚠ 警告: 总损失为NaN或Inf: {total_loss}")
        else:
            print(f"  ✓ 损失值正常: {total_loss.item():.6f}")
        
    except Exception as e:
        print(f"  ✗ 损失计算失败: {e}")
        traceback.print_exc()
        return False
    
    # 8. 测试GPU内存使用
    print("\n8. GPU内存使用:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"  - 已分配: {allocated:.2f} GB")
        print(f"  - 缓存: {cached:.2f} GB")
    else:
        print("  - CPU模式，无GPU内存统计")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！训练环境准备就绪")
    print("=" * 60)
    
    return True


def main():
    """主函数"""
    try:
        success = test_environment()
        if success:
            print("\n🎉 训练环境测试完成，可以开始训练！")
            return 0
        else:
            print("\n❌ 训练环境测试失败，请检查配置")
            return 1
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())