#!/usr/bin/env python3
"""测试观测配置和降采样功能"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import torch
from omegaconf import OmegaConf
from datasets.real_dr_dataset import RealDiffusionReactionDataModule

def test_observation_config():
    """测试观测配置"""
    print("🔍 测试观测配置和降采样功能...")
    
    # 加载配置
    config_path = "configs/ar_training_config.yaml"
    config = OmegaConf.load(config_path)
    
    print(f"✅ 配置加载成功: {config_path}")
    print(f"📊 观测配置: {config.data.observation}")
    
    # 创建数据模块
    data_module = RealDiffusionReactionDataModule(
        data_path=config.data.data_path,
        T_in=config.data.T_in,
        T_out=config.data.T_out,
        batch_size=4,  # 使用小批次进行测试
        num_workers=0,
        observation=config.data.observation
    )
    
    print("✅ 数据模块创建成功")
    print(f"📋 观测参数: {data_module.observation_params}")
    
    # 设置数据模块
    data_module.setup()
    print("✅ 数据模块设置完成")
    
    # 获取训练数据加载器
    train_loader = data_module.train_dataloader()
    print(f"📊 训练集批次数: {len(train_loader)}")
    
    # 测试一个批次
    print("🔍 测试数据批次...")
    batch = next(iter(train_loader))
    
    print(f"📊 批次键: {list(batch.keys())}")
    print(f"📊 输入序列形状: {batch['input_sequence'].shape}")
    print(f"📊 目标序列形状: {batch['target_sequence'].shape}")
    
    # 检查观测参数是否正确传递
    if 'metadata' in batch:
        metadata = batch['metadata']
        if isinstance(metadata, list):
            metadata = metadata[0]  # 取第一个样本的metadata
        print(f"📊 观测参数: {metadata.get('observation_params', 'None')}")
    
    print("✅ 观测配置测试完成！")
    
    return True

if __name__ == "__main__":
    try:
        test_observation_config()
        print("🎉 所有测试通过！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()