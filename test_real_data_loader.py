#!/usr/bin/env python3
"""测试真实数据集加载器"""

import sys
sys.path.append('.')

from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
from omegaconf import DictConfig

def test_real_data_loader():
    """测试真实数据集加载器"""
    print('🧪 测试真实数据集加载器...')
    
    # 创建测试配置
    config = DictConfig({
        'data': {
            'data_path': 'E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
            'T_in': 1,
            'T_out': 20,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'time_step_start': 0,
            'time_step_end': 980,
            'time_step_stride': 1,
            'normalize': True,
            'augmentation': {
                'enabled': True,
                'flip_prob': 0.5,
                'rotate_prob': 0.3,
                'noise_std': 0.01
            }
        },
        'training': {
            'batch_size': 2  # 小批次测试
        },
        'hardware': {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False
        },
        'seed': 2025
    })
    
    try:
        # 创建数据模块
        data_module = RealDiffusionReactionDataModule(config)
        data_module.setup()
        
        # 测试训练数据加载器
        train_loader = data_module.train_dataloader()
        print(f'✅ 训练集批次数: {len(train_loader)}')
        
        # 获取一个批次
        batch = next(iter(train_loader))
        print(f'✅ 输入序列形状: {batch["input_sequence"].shape}')
        print(f'✅ 目标序列形状: {batch["target_sequence"].shape}')
        print(f'✅ 批次大小: {batch["input_sequence"].shape[0]}')
        
        # 检查数据范围
        input_data = batch["input_sequence"]
        target_data = batch["target_sequence"]
        print(f'✅ 输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]')
        print(f'✅ 目标数据范围: [{target_data.min():.6f}, {target_data.max():.6f}]')
        
        # 测试验证数据加载器
        val_loader = data_module.val_dataloader()
        print(f'✅ 验证集批次数: {len(val_loader)}')
        
        print('🎉 数据集加载器测试成功！')
        return True
        
    except Exception as e:
        print(f'❌ 数据集加载器测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_data_loader()