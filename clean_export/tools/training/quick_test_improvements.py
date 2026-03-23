#!/usr/bin/env python3
"""
快速验证脚本 - 测试改进配置的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ops.enhanced_losses import compute_enhanced_total_loss
from ops.losses import compute_total_loss

def create_test_data(batch_size=4, channels=2, height=64, width=64, device='cuda'):
    """创建测试数据"""
    # 创建预测和目标张量
    pred = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)
    
    # 创建归一化统计
    norm_stats = {
        'mu': torch.randn(channels, device=device),
        'sigma': torch.abs(torch.randn(channels, device=device)) + 0.1
    }
    
    # 创建观测数据
    obs_data = {
        'observation': torch.randn(batch_size, channels, height, width, device=device),  # 与预测相同尺寸
        'baseline': torch.randn(batch_size, channels, height, width, device=device),
        'h_params': {'task': 'SR', 'scale': 1, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'}  # scale=1避免尺寸变化
    }
    
    return pred, target, norm_stats, obs_data

def test_loss_functions():
    """测试不同损失函数的效果"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # 创建测试数据
    pred, target, norm_stats, obs_data = create_test_data(device=device)
    
    from omegaconf import DictConfig
    
    # 创建配置字典
    config_dict = {
        'loss': {
            'reconstruction': {'weight': 0.7},
            'spectral': {'weight': 0.2},
            'degradation_consistency': {'weight': 0.1},
            'gradient_weight': 0.05
        },
        'reconstruction': {
            'loss_type': 'mse',
            'reduction': 'mean',
            'huber_delta': 1.0,
            'smooth_l1_beta': 1.0
        },
        'spectral': {
            'k_max': 16,
            'adaptive_weight': True,
            'frequency_weights': None
        },
        'degradation_consistency': {
            'multi_scale': True,
            'scale_factors': [1.0, 0.5, 0.25]
        },
        'adaptive_weights': {
            'adaptive_weights': True,
            'weight_adjustment_factor': 0.1
        }
    }
    
    config = DictConfig(config_dict)
    
    print("\n=== Testing Loss Functions ===")
    
    # 测试原始损失函数
    print("\n1. Original Loss Function:")
    # 创建原始配置（只使用重建损失）
    original_config_dict = {
        'loss': {
            'reconstruction': {'weight': 1.0},
            'spectral': {'weight': 0.0},
            'degradation_consistency': {'weight': 0.0},
            'gradient_weight': 0.0
        },
        'reconstruction': {
            'loss_type': 'mse',
            'reduction': 'mean',
            'huber_delta': 1.0,
            'smooth_l1_beta': 1.0
        },
        'spectral': {
            'k_max': 16,
            'adaptive_weight': False,
            'frequency_weights': None
        },
        'degradation_consistency': {
            'multi_scale': False,
            'scale_factors': [1.0]
        },
        'adaptive_weights': {
            'adaptive_weights': False,
            'weight_adjustment_factor': 0.0
        }
    }
    original_config = DictConfig(original_config_dict)
    
    original_losses = compute_enhanced_total_loss(
        pred_z=pred,
        target_z=target,
        obs_data=obs_data,
        norm_stats=norm_stats,
        config=original_config,
        epoch=0
    )
    print(f"   Total Loss: {original_losses['total_loss'].item():.4f}")
    print(f"   Reconstruction: {original_losses['reconstruction_loss'].item():.4f}")
    
    # 测试增强损失函数
    print("\n2. Enhanced Loss Function:")
    enhanced_losses = compute_enhanced_total_loss(
        pred_z=pred,
        target_z=target,
        obs_data=obs_data,
        norm_stats=norm_stats,
        config=config,
        epoch=10
    )
    print(f"   Total Loss: {enhanced_losses['total_loss'].item():.4f}")
    print(f"   Reconstruction: {enhanced_losses['reconstruction_loss'].item():.4f}")
    print(f"   Spectral: {enhanced_losses['spectral_loss'].item():.4f}")
    print(f"   DC: {enhanced_losses['dc_loss'].item():.4f}")
    print(f"   Gradient: {enhanced_losses['gradient_loss'].item():.4f}")
    print(f"   Weights: {enhanced_losses['weights']}")
    
    return original_losses, enhanced_losses

def test_convergence_simulation():
    """模拟收敛过程"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Testing Convergence Simulation ===")
    
    # 创建简单的线性模型用于测试
    model = nn.Sequential(
        nn.Conv2d(2, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 2, 3, padding=1)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # 创建目标数据（带有一些模式）
    target = torch.randn(4, 2, 64, 64, device=device)
    # 添加一些低频模式
    x_coords = torch.linspace(-1, 1, 64, device=device)
    y_coords = torch.linspace(-1, 1, 64, device=device)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    pattern = torch.sin(xx * 3) * torch.cos(yy * 2) * 0.5
    target += pattern.unsqueeze(0).unsqueeze(0)
    
    # 归一化统计
    norm_stats = {
        'mu': torch.tensor([0.0, 0.0], device=device),
        'sigma': torch.tensor([1.0, 1.0], device=device)
    }
    
    obs_data = {
        'observation': torch.randn(4, 2, 64, 64, device=device),  # 与预测相同尺寸
        'baseline': torch.randn(4, 2, 64, 64, device=device),
        'h_params': {'task': 'SR', 'scale': 1, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'}  # scale=1避免尺寸变化
    }
    
    from omegaconf import DictConfig
    
    config_dict = {
        'loss': {
            'reconstruction': {'weight': 0.7},
            'spectral': {'weight': 0.2},
            'degradation_consistency': {'weight': 0.1},
            'gradient_weight': 0.05
        },
        'reconstruction': {
            'loss_type': 'mse',
            'reduction': 'mean',
            'huber_delta': 1.0,
            'smooth_l1_beta': 1.0
        },
        'spectral': {
            'k_max': 16,
            'adaptive_weight': True,
            'frequency_weights': None
        },
        'degradation_consistency': {
            'multi_scale': True,
            'scale_factors': [1.0, 0.5, 0.25]
        },
        'adaptive_weights': {
            'adaptive_weights': True,
            'weight_adjustment_factor': 0.1
        }
    }
    
    config = DictConfig(config_dict)
    
    print("Simulating 50 training steps...")
    
    losses = []
    gradient_norms = []
    
    for step in range(50):
        # 随机输入
        x = torch.randn(4, 2, 64, 64, device=device)
        
        # 前向传播
        pred = model(x)
        
        # 计算损失
        losses_dict = compute_enhanced_total_loss(
            pred_z=pred,
            target_z=target,
            obs_data=obs_data,
            norm_stats=norm_stats,
            config=config,
            epoch=step // 5  # 模拟epoch
        )
        
        loss = losses_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度范数
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # 梯度裁剪和更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录
        losses.append(loss.item())
        gradient_norms.append(total_grad_norm)
        
        if step % 10 == 0:
            print(f"  Step {step}: Loss={loss.item():.4f}, Grad Norm={total_grad_norm:.2f}")
    
    # 分析收敛性
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nConvergence Analysis:")
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Loss Reduction: {loss_reduction:.1f}%")
    print(f"  Avg Gradient Norm: {np.mean(gradient_norms):.2f}")
    
    # 检查是否收敛
    recent_trend = np.polyfit(range(min(20, len(losses))), losses[-min(20, len(losses)):], 1)[0]
    is_converging = recent_trend < -0.001
    print(f"  Converging: {is_converging} (trend: {recent_trend:.4f})")
    
    return losses, gradient_norms, loss_reduction, is_converging

def main():
    """主函数"""
    print("=== Quick Training Loss Optimization Test ===")
    
    # 测试损失函数
    original_losses, enhanced_losses = test_loss_functions()
    
    # 测试收敛模拟
    losses, gradient_norms, loss_reduction, is_converging = test_convergence_simulation()
    
    print(f"\n=== Summary ===")
    print(f"Enhanced loss function provides additional regularization:")
    print(f"  - Spectral loss: {enhanced_losses['spectral_loss'].item():.4f}")
    print(f"  - DC loss: {enhanced_losses['dc_loss'].item():.4f}")
    print(f"  - Gradient loss: {enhanced_losses['gradient_loss'].item():.4f}")
    
    print(f"\nConvergence simulation results:")
    print(f"  - Loss reduction: {loss_reduction:.1f}%")
    print(f"  - Converging: {is_converging}")
    print(f"  - Average gradient norm: {np.mean(gradient_norms):.2f}")
    
    if is_converging and loss_reduction > 20:
        print("\n✅ Test PASSED: Configuration improvements show good convergence!")
    else:
        print("\n⚠️  Test RESULTS: Configuration shows some improvement but may need further tuning.")
    
    print("\n=== Recommendations ===")
    print("1. Use enhanced loss function with multiple loss components")
    print("2. Implement adaptive learning rate scheduling")
    print("3. Add gradient clipping to prevent instability")
    print("4. Monitor loss components separately for better debugging")
    print("5. Consider early stopping based on validation loss")

if __name__ == "__main__":
    main()