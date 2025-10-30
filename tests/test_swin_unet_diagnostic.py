#!/usr/bin/env python3
"""SwinUNet模型诊断测试脚本

用于诊断SwinUNet模型的问题，包括：
1. 模型创建测试
2. 前向传播测试
3. 梯度计算测试
4. 损失函数测试
5. 参数检查
"""

import torch
import torch.nn as nn
import numpy as np
import traceback
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.swin_unet import SwinUNet
from models.base import create_model


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("1. 测试SwinUNet模型创建")
    print("=" * 60)
    
    try:
        # 测试基本参数
        model = SwinUNet(
            in_channels=3,
            out_channels=3,
            img_size=256,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm  # 确保传入可调用对象而不是字符串
        )
        
        print("✓ SwinUNet模型创建成功")
        print(f"  - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
        
    except Exception as e:
        print(f"✗ SwinUNet模型创建失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return None


def test_forward_pass(model, device):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("2. 测试前向传播")
    print("=" * 60)
    
    if model is None:
        print("✗ 跳过前向传播测试（模型创建失败）")
        return None
    
    try:
        # 创建测试输入，确保在正确设备上
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
        
        print(f"输入张量形状: {input_tensor.shape}")
        print(f"输入设备: {input_tensor.device}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出张量形状: {output.shape}")
        print(f"  - 输出数值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        print(f"  - 输出均值: {output.mean().item():.6f}")
        print(f"  - 输出标准差: {output.std().item():.6f}")
        
        return output
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return None


def test_gradient_computation(model, device):
    """测试梯度计算"""
    print("\n" + "=" * 60)
    print("3. 测试梯度计算")
    print("=" * 60)
    
    if model is None:
        print("✗ 跳过梯度计算测试（模型创建失败）")
        return False
    
    try:
        # 创建测试输入和目标，确保在正确设备上
        input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True).to(device)
        target = torch.randn(1, 3, 256, 256).to(device)
        
        # 前向传播
        model.train()
        output = model(input_tensor)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        print(f"✓ 梯度计算成功")
        print(f"  - 损失值: {loss.item():.6f}")
        print(f"  - 有梯度的参数数量: {len(grad_norms)}")
        print(f"  - 梯度范数统计: 最小={min(grad_norms):.6f}, 最大={max(grad_norms):.6f}, 平均={np.mean(grad_norms):.6f}")
        
        # 检查是否有异常梯度
        if any(np.isnan(g) or np.isinf(g) for g in grad_norms):
            print("⚠ 警告: 发现NaN或Inf梯度")
            return False
        
        if max(grad_norms) > 100:
            print("⚠ 警告: 发现过大的梯度（可能存在梯度爆炸）")
        
        return True
        
    except Exception as e:
        print(f"✗ 梯度计算失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return False


def test_loss_computation(model, device):
    """测试损失函数计算"""
    print("\n" + "=" * 60)
    print("4. 测试损失函数计算")
    print("=" * 60)
    
    if model is None:
        print("✗ 跳过损失函数测试（模型创建失败）")
        return False
    
    try:
        # 创建测试数据，确保在正确设备上
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
        target = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # 测试不同损失函数
        losses = {}
        
        # L2损失
        losses['L2'] = nn.MSELoss()(output, target).item()
        
        # L1损失
        losses['L1'] = nn.L1Loss()(output, target).item()
        
        # 相对L2损失
        diff = output - target
        rel_l2 = torch.sqrt(torch.sum(diff ** 2, dim=(1,2,3))) / torch.sqrt(torch.sum(target ** 2, dim=(1,2,3)))
        losses['Rel_L2'] = rel_l2.mean().item()
        
        print("✓ 损失函数计算成功")
        for loss_name, loss_value in losses.items():
            print(f"  - {loss_name}: {loss_value:.6f}")
        
        # 检查损失值是否合理
        if any(np.isnan(v) or np.isinf(v) for v in losses.values()):
            print("⚠ 警告: 发现NaN或Inf损失值")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 损失函数计算失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return False


def test_parameter_analysis(model):
    """测试参数分析"""
    print("\n" + "=" * 60)
    print("5. 参数分析")
    print("=" * 60)
    
    if model is None:
        print("✗ 跳过参数分析（模型创建失败）")
        return
    
    try:
        # 统计参数
        total_params = 0
        trainable_params = 0
        param_stats = {}
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            # 按模块分类统计
            module_name = name.split('.')[0]
            if module_name not in param_stats:
                param_stats[module_name] = 0
            param_stats[module_name] += param_count
        
        print(f"✓ 参数分析完成")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 冻结参数: {total_params - trainable_params:,}")
        
        print("\n各模块参数分布:")
        for module_name, param_count in sorted(param_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = param_count / total_params * 100
            print(f"  - {module_name}: {param_count:,} ({percentage:.1f}%)")
        
        # 检查参数初始化
        param_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.norm().item()
                param_norms.append(param_norm)
        
        print(f"\n参数范数统计:")
        print(f"  - 最小: {min(param_norms):.6f}")
        print(f"  - 最大: {max(param_norms):.6f}")
        print(f"  - 平均: {np.mean(param_norms):.6f}")
        print(f"  - 标准差: {np.std(param_norms):.6f}")
        
    except Exception as e:
        print(f"✗ 参数分析失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")


def test_config_compatibility(device):
    """测试配置兼容性"""
    print("\n" + "=" * 60)
    print("6. 测试配置兼容性")
    print("=" * 60)
    
    try:
        # 测试通过create_model函数创建 - 修正参数格式
        model = create_model(
            'SwinUNet',  # 直接传入字符串
            in_channels=3,
            out_channels=3,
            img_size=256,
            patch_size=4,
            window_size=8,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm
        )
        
        model = model.to(device)
        print("✓ 通过create_model函数创建成功")
        
        # 测试前向传播
        input_tensor = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"✓ 配置兼容性测试通过")
        print(f"  - 输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置兼容性测试失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return False


def main():
    """主函数"""
    print("SwinUNet模型诊断测试")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行测试
    model = test_model_creation()
    
    if model is not None:
        model = model.to(device)
    
    output = test_forward_pass(model, device)
    gradient_ok = test_gradient_computation(model, device)
    loss_ok = test_loss_computation(model, device)
    test_parameter_analysis(model)
    config_ok = test_config_compatibility(device)
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断结果总结")
    print("=" * 60)
    
    tests = [
        ("模型创建", model is not None),
        ("前向传播", output is not None),
        ("梯度计算", gradient_ok),
        ("损失函数", loss_ok),
        ("配置兼容性", config_ok)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！SwinUNet模型状态正常。")
    else:
        print("⚠️  存在问题，需要进一步调试。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)