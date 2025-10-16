#!/usr/bin/env python3
"""测试训练环境设置"""

import sys
import os
sys.path.append('.')

from datasets import PDEBenchDataModule
# from models.base import create_model
from ops.loss import TotalLoss
from omegaconf import OmegaConf
import torch

def test_training_setup():
    """测试训练环境设置"""
    print("=== 测试训练环境设置 ===")
    
    try:
        # 1. 加载配置
        print("1. 加载配置...")
        from hydra import compose, initialize
        with initialize(config_path="configs", version_base=None):
            config = compose(config_name="config")
        print("✓ 配置加载成功")
        
        # 2. 测试数据模块
        print("\n2. 测试数据模块...")
        data_module = PDEBenchDataModule(config.data)
        print("✓ 数据模块创建成功")
        
        # 设置数据模块
        data_module.setup()
        print("✓ 数据模块设置完成")
        
        # 获取训练数据加载器
        train_loader = data_module.train_dataloader()
        print(f"✓ 训练数据加载器创建成功: {len(train_loader)} 批次")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print("✓ 批次数据加载成功")
        print(f"  - 批次键: {list(batch.keys())}")
        if 'input' in batch:
            print(f"  - 输入形状: {batch['input'].shape}")
        if 'target' in batch:
            print(f"  - 目标形状: {batch['target'].shape}")
        if 'observation' in batch:
            print(f"  - 观测形状: {batch['observation'].shape}")
        if 'gt' in batch:
            print(f"  - GT形状: {batch['gt'].shape}")
        if 'lr' in batch:
            print(f"  - LR形状: {batch['lr'].shape}")
        if 'hr' in batch:
            print(f"  - HR形状: {batch['hr'].shape}")
        
        # 3. 测试模型创建
        print("\n3. 测试模型创建...")
        from models import create_model
        model = create_model(config)
        print("✓ 模型创建成功")
        print(f"  - 模型类型: {type(model).__name__}")
        
        # 4. 测试损失函数
        print("\n4. 测试损失函数...")
        loss_fn = TotalLoss(
            rec_weight=config.train.loss_weights.reconstruction,
            spec_weight=config.train.loss_weights.spectral,
            dc_weight=config.train.loss_weights.data_consistency
        )
        print("✓ 损失函数创建成功")
        
        # 5. 测试前向传播
        print("\n5. 测试前向传播...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 根据实际的批次键来获取张量
        if 'baseline' in batch:
            input_tensor = batch['baseline'].to(device)
        elif 'input' in batch:
            input_tensor = batch['input'].to(device)
        elif 'lr' in batch:
            input_tensor = batch['lr'].to(device)
        else:
            raise ValueError("找不到输入张量")
            
        if 'target' in batch:
            target_tensor = batch['target'].to(device)
        elif 'gt' in batch:
            target_tensor = batch['gt'].to(device)
        elif 'hr' in batch:
            target_tensor = batch['hr'].to(device)
        else:
            raise ValueError("找不到目标张量")
            
        if 'observation' in batch:
            observation_tensor = batch['observation'].to(device)
        elif 'lr' in batch:
            observation_tensor = batch['lr'].to(device)
        else:
            raise ValueError("找不到观测张量")
        
        # 前向传播
        with torch.no_grad():
            output = model(input_tensor)
            print(f"✓ 模型前向传播成功: {output.shape}")
            
            # 测试损失计算
            task_params = batch.get('task_params', {
                'observation_operator': 'SR',
                'scale_factor': 4,
                'denorm_fn': None
            })
            
            loss, loss_dict = loss_fn(output, target_tensor, observation_tensor, task_params)
            print(f"✓ 损失计算成功: {loss.item():.6f}")
            print(f"  - 重建损失: {loss_dict['reconstruction'].item():.6f}")
            print(f"  - 频域损失: {loss_dict['spectral'].item():.6f}")
            print(f"  - 数据一致性损失: {loss_dict['data_consistency'].item():.6f}")
        
        # 6. 检查GPU内存
        if torch.cuda.is_available():
            print(f"\n6. GPU内存使用:")
            print(f"  - 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  - 缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        print("\n=== 所有测试通过！训练环境准备就绪 ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_setup()
    sys.exit(0 if success else 1)