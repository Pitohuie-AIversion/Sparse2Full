#!/usr/bin/env python3
"""
测试训练脚本与观测配置的兼容性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tools.training.train_real_data_ar import RealDataARTrainer
from omegaconf import OmegaConf

def test_training_compatibility():
    """测试训练脚本与观测配置的兼容性"""
    
    # 加载配置
    config_path = "configs/ar_training_config.yaml"
    config = OmegaConf.load(config_path)
    
    # 创建训练器
    print("🔍 创建训练器...")
    trainer = RealDataARTrainer(config_path)
    print("✅ 训练器创建成功")
    
    # 检查观测配置
    print(f"📊 观测配置: {trainer.config.data.observation}")
    print(f"📊 数据模块观测参数: {trainer.data_module.observation_params}")
    
    # 测试数据加载
    print("🔍 测试数据加载...")
    trainer.data_module.setup()
    train_loader = trainer.data_module.train_dataloader()
    
    # 获取一个批次
    sample_batch = next(iter(train_loader))
    print(f"📊 批次键: {list(sample_batch.keys())}")
    print(f"📊 输入序列形状: {sample_batch['input_sequence'].shape}")
    print(f"📊 目标序列形状: {sample_batch['target_sequence'].shape}")
    
    # 测试模型前向传播
    print("🔍 测试模型前向传播...")
    trainer.model.eval()
    with torch.no_grad():
        input_seq = sample_batch['input_sequence'][:2].to(trainer.device)  # 取2个样本
        target_seq = sample_batch['target_sequence'][:2].to(trainer.device)
        
        print(f"📊 输入到模型的形状: {input_seq.shape}")
        
        # 简单测试单步前向传播
        # 重塑为模型期望的格式 [B, T_in*C, H, W]
        B, T, C, H, W = input_seq.shape
        model_input = input_seq.reshape(B, T*C, H, W)
        
        print(f"📊 模型输入形状: {model_input.shape}")
        
        # 前向传播
        pred = trainer.model(model_input)  # [B, C, H, W]
        
        print(f"✅ 模型前向传播成功")
        print(f"📊 预测输出形状: {pred.shape}")
        print(f"📊 目标形状: {target_seq[:, 0].shape}")  # 第一个时间步的目标
    
    print("✅ 训练脚本兼容性测试完成！")
    
    return True

if __name__ == "__main__":
    try:
        test_training_compatibility()
        print("🎉 兼容性测试通过！")
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)