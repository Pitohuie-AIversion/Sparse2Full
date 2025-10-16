#!/usr/bin/env python3
"""简单测试脚本，验证训练脚本的基本功能"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import create_dataloader
from models import create_model
from ops.loss import TotalLoss

def test_basic_functionality():
    """测试基本功能"""
    print("开始基本功能测试...")
    
    # 1. 测试配置加载
    print("1. 测试配置加载...")
    config = OmegaConf.load('configs/train.yaml')
    print(f"   配置加载成功: {config.data.dataset_name}")
    
    # 2. 测试数据加载
    print("2. 测试数据加载...")
    train_loader = create_dataloader(config, split='train')
    print(f"   训练数据加载器创建成功，批次大小: {config.training.batch_size}")
    
    # 3. 获取一个批次数据
    print("3. 测试数据批次...")
    batch = next(iter(train_loader))
    print(f"   批次数据键: {list(batch.keys())}")
    print(f"   target形状: {batch['target'].shape}")
    print(f"   observation形状: {batch['observation'].shape}")
    if 'original_observation' in batch:
        print(f"   original_observation形状: {batch['original_observation'].shape}")
    
    # 4. 测试模型创建
    print("4. 测试模型创建...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    model = model.to(device)
    print(f"   模型创建成功，设备: {device}")
    
    # 5. 测试前向传播
    print("5. 测试前向传播...")
    observation = batch['observation'].to(device)
    target = batch['target'].to(device)
    
    with torch.no_grad():
        pred = model(observation)
    print(f"   前向传播成功，预测形状: {pred.shape}")
    
    # 6. 测试损失计算
    print("6. 测试损失计算...")
    loss_fn = TotalLoss(
        rec_weight=config.loss.rec_weight,
        spec_weight=config.loss.spec_weight,
        dc_weight=config.loss.dc_weight,
        rec_loss_type=config.loss.rec_loss_type,
        spec_loss_type=config.loss.spec_loss_type,
        dc_loss_type=config.loss.dc_loss_type,
        low_freq_modes=config.loss.low_freq_modes
    )
    
    # 处理task_params
    sample_task_params = {}
    for key, value in batch['task_params'].items():
        if isinstance(value, (list, tuple)):
            sample_task_params[key] = value[0] if len(value) > 0 else value
        else:
            sample_task_params[key] = value
    
    # 确保参数是标量而不是张量
    for key, value in sample_task_params.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                sample_task_params[key] = value.item()
            else:
                print(f"Warning: {key} is a tensor with {value.numel()} elements, using first element")
                sample_task_params[key] = value.flatten()[0].item()
    
    # 使用原始观测数据进行损失计算
    obs_for_loss = batch.get('original_observation', batch['observation']).to(device)
    
    loss, loss_dict = loss_fn(
        pred=pred,
        target=target,
        observation=obs_for_loss,
        task_params=sample_task_params
    )
    print(f"   损失计算成功，总损失: {loss.item():.6f}")
    print(f"   损失组件: {loss_dict}")
    
    print("\n✅ 所有基本功能测试通过！")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)