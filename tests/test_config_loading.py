#!/usr/bin/env python3
"""
测试配置文件加载和训练初始化
验证配置是否可以正常启动训练流程
"""

import os
import sys
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_config_loading(config_path: str, config_name: str) -> Dict[str, Any]:
    """测试单个配置文件的加载和初始化"""
    result = {
        'config_name': config_name,
        'success': False,
        'error': None,
        'details': {}
    }
    
    try:
        print(f"测试配置: {config_name}")
        
        # 1. 加载配置
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name=config_name)
        
        print(f"  配置加载成功")
        result['details']['config_loaded'] = True
        
        # 2. 测试数据模块初始化
        try:
            from datasets import PDEBenchDataModule
            # PDEBenchDataModule只接受config参数
            data_module = PDEBenchDataModule(cfg.data)
            print(f"  数据模块初始化成功")
            result['details']['data_module'] = True
        except Exception as e:
            print(f"  数据模块初始化失败: {e}")
            result['details']['data_module'] = False
            result['error'] = f"数据模块: {e}"
            return result
        
        # 3. 测试模型初始化
        try:
            from models import create_model
            # 检查模型配置结构
            print(f"    模型配置: {cfg.model}")
            model = create_model(cfg.model)
            print(f"  模型初始化成功: {model.__class__.__name__}")
            result['details']['model'] = True
            
            # 检查模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    总参数: {total_params:,}, 可训练: {trainable_params:,}")
            result['details']['total_params'] = total_params
            result['details']['trainable_params'] = trainable_params
            
        except Exception as e:
            print(f"  模型初始化失败: {e}")
            result['details']['model'] = False
            result['error'] = f"模型: {e}"
            return result
        
        # 4. 测试优化器初始化
        try:
            if cfg.training.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.training.learning_rate
                )
            print(f"  优化器初始化成功: {cfg.training.optimizer}")
            result['details']['optimizer'] = True
        except Exception as e:
            print(f"  优化器初始化失败: {e}")
            result['details']['optimizer'] = False
            result['error'] = f"优化器: {e}"
            return result
        
        # 5. 测试损失函数初始化
        try:
            from ops.losses import compute_total_loss
            # 创建虚拟数据测试损失计算
            batch_size = 1
            channels = model.in_channels
            img_size = model.img_size
            
            dummy_pred = torch.randn(batch_size, channels, img_size, img_size)
            dummy_target = torch.randn(batch_size, channels, img_size, img_size)
            dummy_observed = torch.randn(batch_size, channels, img_size, img_size)
            
            # 创建虚拟观测数据
            obs_data = {
                'baseline': dummy_observed,
                'mask': torch.ones_like(dummy_observed),
                'coords': torch.randn(batch_size, 2, img_size, img_size),
                'h_params': {'scale': 2, 'sigma': 1.0},
                'observation': dummy_observed
            }
            
            # 创建虚拟归一化统计量
            norm_stats = {
                'mean': torch.zeros(channels),
                'std': torch.ones(channels)
            }
            
            loss_dict = compute_total_loss(
                pred_z=dummy_pred,
                target_z=dummy_target,
                obs_data=obs_data,
                norm_stats=norm_stats,
                config=cfg
            )
            print(f"  损失函数测试成功")
            result['details']['loss'] = True
        except Exception as e:
            print(f"  损失函数测试失败: {e}")
            result['details']['loss'] = False
            result['error'] = f"损失函数: {e}"
            return result
        
        # 6. 测试前向传播
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, channels, img_size, img_size)
                output = model(dummy_input)
                expected_shape = (1, cfg.model.out_channels, img_size, img_size)
                if output.shape == expected_shape:
                    print(f"  ✅ 前向传播测试成功: {output.shape}")
                    result['details']['forward'] = True
                else:
                    print(f"  ❌ 输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}")
                    result['details']['forward'] = False
                    result['error'] = f"输出形状不匹配"
                    return result
        except Exception as e:
            print(f"  ❌ 前向传播测试失败: {e}")
            result['details']['forward'] = False
            result['error'] = f"前向传播: {e}"
            return result
        
        result['success'] = True
        print(f"  🎉 配置测试完全通过!")
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        result['error'] = str(e)
    
    return result

def main():
    """主函数"""
    print("开始测试配置文件加载和训练初始化...")
    
    config_dir = Path("configs/auto_generated")
    if not config_dir.exists():
        print(f"配置目录不存在: {config_dir}")
        return False
    
    # 获取所有YAML配置文件
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        print(f"未找到配置文件在: {config_dir}")
        return False
    
    print(f"找到 {len(config_files)} 个配置文件")
    
    results = []
    success_count = 0
    
    # 测试每个配置文件
    for config_file in sorted(config_files):
        config_name = config_file.stem
        result = test_config_loading("configs/auto_generated", config_name)
        results.append(result)
        
        if result['success']:
            success_count += 1
    
    # 输出总结
    print(f"\n测试结果总结:")
    print(f"  成功: {success_count}/{len(config_files)}")
    print(f"  失败: {len(config_files) - success_count}/{len(config_files)}")
    
    # 详细结果
    success_configs = []
    failed_configs = []
    
    for result in results:
        if result['success']:
            success_configs.append(result['config_name'])
        else:
            failed_configs.append(result)
    
    if success_configs:
        print(f"\n🎉 可以正常训练的配置:")
        for config_name in success_configs:
            print(f"  - {config_name}")
    
    if failed_configs:
        print(f"\n⚠️  需要修复的配置:")
        for result in failed_configs:
            print(f"  - {result['config_name']}: {result['error']}")
    
    # 提供训练命令
    if success_configs:
        print(f"\n🚀 训练命令示例:")
        example_config = success_configs[0]
        print(f"  python train.py --config-path configs/auto_generated --config-name {example_config}")
    
    return success_count == len(config_files)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)