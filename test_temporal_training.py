#!/usr/bin/env python3
"""
测试时序增强训练脚本
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import h5py
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.training.train_temporal_enhanced import TemporalEnhancedTrainer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    return {
        'experiment': {
            'name': 'test_temporal',
            'seed': 42,
            'output_dir': './test_runs',
            'save_interval': 10,
            'log_interval': 5
        },
        'data': {
            'name': 'real_diffusion_reaction',
            'data_path': './test_data.h5',  # 将使用模拟数据
            'T_in': 4,
            'T_out': 4,
            'dt': 0.1,
            'spatial_size': [64, 64],
            'img_size': 64,  # 添加img_size字段
            'channels': 2,   # 添加channels字段
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'time_step_start': 0,  # 添加时间步字段
            'time_step_end': 20,
            'time_step_stride': 1,
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
            'normalize': True,
            'augmentation': {
                'enabled': False,
                'flip_prob': 0.0,
                'rotate_prob': 0.0,
                'noise_std': 0.0
            }
        },
        'model': {
            'name': 'SwinTemporalWrapper',
            'in_channels': 2,
            'out_channels': 2,
            'img_size': 64,  # 使用标量而不是列表
            'T_in': 4,
            'T_out': 4,
            'prediction_mode': 'ar',
            'scheduled_sampling': {
                'enabled': True,
                'initial_prob': 1.0,  # 使用initial_prob而不是initial_ratio
                'final_prob': 0.0,    # 使用final_prob而不是final_ratio
                'decay_type': 'linear',
                'decay_steps': 100
            },
            'temporal_encoder': {
                'type': 'conv1d',
                'c_out': 64,  # 使用c_out而不是hidden_dim
                'k': 3,       # 添加k字段
                'causal': True  # 添加causal字段
            },
            'nar_head': {
                'type': 'simple',
                'd_model': 256,
                'max_timesteps': 20
            },
            'swin_config': {
                'patch_size': 4,  # 添加patch_size
                'window_size': 7,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'mlp_ratio': 4.0,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,  # 添加attn_drop_rate
                'drop_path_rate': 0.1
            }
        },
        'training': {
            'epochs': 10,  # 使用epochs而不是total_epochs
            'batch_size': 4,
            'accumulate_grad_batches': 1,  # 添加accumulate_grad_batches
            'optimizer': {
                'name': 'AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'betas': [0.9, 0.999]
            },
            'scheduler': {
                'name': 'CosineAnnealingLR',
                'T_max': 10,
                'eta_min': 1e-6,
                'warmup_epochs': 0
            },
            'gradient_clip_val': 1.0,
            'amp': {
                'enabled': True,
                'opt_level': 'O1'
            }
        },
        'loss': {
            'reconstruction': {
                'name': 'MSELoss',
                'weight': 1.0
            },
            'relative_l2': {
                'name': 'RelativeL2Loss',
                'weight': 1.0,
                'eps': 1e-8
            },
            'temporal_consistency': {
                'name': 'TemporalConsistencyLoss',
                'weight': 0.1
            },
            'spectral': {
                'name': 'SpectralLoss',
                'weight': 0.1,
                'freq_bands': [4, 8]
            }
        },
        'curriculum': {
            'enabled': True,
            'strategy': 'progressive',
            'stages': [
                {
                    'epochs': 5,
                    'T_out': 2,
                    'mode': 'ar',
                    'description': '阶段1: AR模式预测2步'
                },
                {
                    'epochs': 10,
                    'T_out': 4,
                    'mode': 'ar',
                    'description': '阶段2: AR模式预测4步'
                }
            ]
        },
        'validation': {
            'check_val_every_n_epoch': 5,
            'val_check_interval': 1.0,
            'metrics': ['mse', 'mae', 'rel_l2', 'temporal_consistency']
        },
        'hardware': {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False
        }
    }

def create_dummy_data(file_path, num_samples=100, T=20, H=64, W=64, C=2):
    """创建模拟数据文件"""
    logger.info(f"创建模拟数据文件: {file_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        # 创建模拟的时序数据
        data = np.random.randn(num_samples, T, C, H, W).astype(np.float32)
        
        # 添加一些时序相关性
        for i in range(1, T):
            data[:, i] = 0.8 * data[:, i-1] + 0.2 * data[:, i]
        
        # 保存数据
        f.create_dataset('u', data=data)
        
        # 创建时间坐标
        t = np.linspace(0, 2.0, T).astype(np.float32)
        f.create_dataset('t', data=t)
        
        # 创建空间坐标
        x = np.linspace(0, 1, W).astype(np.float32)
        y = np.linspace(0, 1, H).astype(np.float32)
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        
        logger.info(f"数据形状: {data.shape}")

def create_mock_data(file_path: str, n_timesteps: int = 20, n_samples: int = 100, 
                    height: int = 64, width: int = 64, n_channels: int = 2):
    """创建模拟数据文件，匹配真实数据格式"""
    logger.info(f"创建模拟数据文件: {file_path}")
    
    with h5py.File(file_path, 'w') as f:
        # 创建时间步组，格式为 '0000', '0001', ...
        for t in range(n_timesteps):
            time_key = f"{t:04d}"  # 格式化为4位数字符串
            group = f.create_group(time_key)
            
            # 创建数据集，形状为 (n_samples, height, width, n_channels)
            # 这匹配真实数据的格式 (101, 128, 128, 2)
            data = np.random.randn(n_samples, height, width, n_channels).astype(np.float32)
            
            # 添加一些时间相关的模式，使数据更真实
            t_norm = t / (n_timesteps - 1)  # 归一化时间 [0, 1]
            
            # 第一个通道：扩散模式
            data[:, :, :, 0] += np.sin(2 * np.pi * t_norm) * 0.5
            
            # 第二个通道：反应模式  
            data[:, :, :, 1] += np.cos(2 * np.pi * t_norm) * 0.3
            
            # 添加空间相关性
            for i in range(n_samples):
                # 创建空间梯度
                x = np.linspace(-1, 1, width)
                y = np.linspace(-1, 1, height)
                X, Y = np.meshgrid(x, y)
                
                # 添加空间模式
                spatial_pattern = np.exp(-(X**2 + Y**2) / 0.5) * np.sin(t_norm * np.pi)
                data[i, :, :, 0] += spatial_pattern * 0.2
                data[i, :, :, 1] += spatial_pattern * 0.1
            
            group.create_dataset('data', data=data)
            
            # 可选：添加网格信息（如果需要）
            if t == 0:  # 只在第一个时间步添加网格信息
                x_grid = np.linspace(0, 1, width)
                y_grid = np.linspace(0, 1, height)
                group.create_dataset('x', data=x_grid)
                group.create_dataset('y', data=y_grid)
    
    logger.info(f"数据形状: ({n_timesteps}, {n_samples}, {height}, {width}, {n_channels})")
    logger.info(f"时间步格式: 0000 ~ {n_timesteps-1:04d}")
    
    return file_path

def test_model_creation():
    """测试模型创建"""
    logger.info("🧪 测试模型创建...")
    
    try:
        # 创建模拟数据文件
        create_mock_data('./test_data.h5', n_timesteps=20, n_samples=100)
        
        config = create_test_config()
        trainer = TemporalEnhancedTrainer(config_dict=config)
        
        logger.info("✅ 模型创建测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 模型创建测试失败: {e}")
        return False

def test_loss_functions():
    """测试损失函数"""
    logger.info("🧪 测试损失函数...")
    
    try:
        # 确保数据文件存在
        if not os.path.exists('./test_data.h5'):
            create_mock_data('./test_data.h5', n_timesteps=20, n_samples=100)
        
        config = create_test_config()
        trainer = TemporalEnhancedTrainer(config_dict=config)
        
        # 创建模拟输入
        batch_size, T_out, C, H, W = 2, 4, 2, 64, 64
        pred = torch.randn(batch_size, T_out, C, H, W)
        target = torch.randn(batch_size, T_out, C, H, W)
        
        # 测试损失计算
        loss = trainer.compute_loss(pred, target)
        
        logger.info(f"损失值: {loss.item():.6f}")
        logger.info("✅ 损失函数测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 损失函数测试失败: {e}")
        return False

def test_curriculum_learning():
    """测试课程学习"""
    logger.info("🧪 测试课程学习...")
    
    try:
        # 确保数据文件存在
        if not os.path.exists('./test_data.h5'):
            create_mock_data('./test_data.h5', n_timesteps=20, n_samples=100)
        
        config = create_test_config()
        trainer = TemporalEnhancedTrainer(config_dict=config)
        
        # 测试课程学习更新
        for epoch in range(1, 6):
            trainer.update_curriculum(epoch)
            logger.info(f"Epoch {epoch}: T_out={trainer.current_T_out}, mode={trainer.current_mode}")
        
        logger.info("✅ 课程学习测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 课程学习测试失败: {e}")
        return False

def test_training_loop():
    """测试训练循环"""
    logger.info("🧪 测试训练循环...")
    
    try:
        # 确保数据文件存在
        if not os.path.exists('./test_data.h5'):
            create_mock_data('./test_data.h5', n_timesteps=20, n_samples=100)
        
        config = create_test_config()
        trainer = TemporalEnhancedTrainer(config_dict=config)
        
        # 获取一个批次进行测试
        train_loader = trainer.data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        # 测试前向传播
        input_seq = batch['input_sequence']  # [B, T_in, C, H, W]
        target_seq = batch['target_sequence']  # [B, T_out, C, H, W]
        
        logger.info(f"输入形状: {input_seq.shape}")
        logger.info(f"目标形状: {target_seq.shape}")
        
        # 测试模型前向传播
        with torch.no_grad():
            pred = trainer.model(input_seq)
            logger.info(f"预测形状: {pred.shape}")
        
        logger.info("✅ 训练循环测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 训练循环测试失败: {e}")
        return False

def test_full_training():
    """测试完整训练流程"""
    logger.info("🧪 测试完整训练流程...")
    
    try:
        # 确保数据文件存在
        if not os.path.exists('./test_data.h5'):
            create_mock_data('./test_data.h5', n_timesteps=20, n_samples=100)
        
        # 创建简化配置用于快速测试
        config = create_test_config()
        config['training']['epochs'] = 2  # 只训练2个epoch
        config['curriculum']['stages'][0]['epochs'] = 1
        config['curriculum']['stages'][1]['epochs'] = 2
        
        trainer = TemporalEnhancedTrainer(config_dict=config)
        
        # 运行训练
        trainer.train()
        
        logger.info("✅ 完整训练流程测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 完整训练流程测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始时序训练脚本测试...")
    
    try:
        test_model_creation()
        test_loss_functions()
        test_curriculum_learning()
        test_training_loop()
        test_full_training()
        
        logger.info("🎉 所有测试通过！")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()