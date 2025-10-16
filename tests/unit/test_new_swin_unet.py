#!/usr/bin/env python3
"""测试新的对称Swin-UNet架构的数值健康性"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swin_unet import SwinUNet


def test_model_architecture():
    """测试模型架构的基本功能"""
    print("=" * 60)
    print("测试新的对称Swin-UNet架构")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
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
        skip_connections=True,
        use_fno_bottleneck=False,
        final_activation=None
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试前向传播
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"\n输入张量形状: {input_tensor.shape}")
    print(f"输入张量范围: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    
    # 前向传播
    try:
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"输出张量形状: {output.shape}")
        print(f"输出张量范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"输出张量均值: {output.mean():.4f}")
        print(f"输出张量标准差: {output.std():.4f}")
        
        # 检查是否有NaN或Inf
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值!")
            return False
        if torch.isinf(output).any():
            print("❌ 输出包含Inf值!")
            return False
        
        print("✅ 前向传播成功，无NaN/Inf值")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_gradient_flow():
    """测试梯度流动"""
    print("\n" + "=" * 60)
    print("测试梯度流动")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SwinUNet(
        in_channels=3,
        out_channels=3,
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        skip_connections=True,
        use_fno_bottleneck=False
    ).to(device)
    
    # 创建输入和目标
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    target = torch.randn(1, 3, 256, 256).to(device)
    
    # 前向传播
    output = model(input_tensor)
    
    # 计算损失
    loss = nn.MSELoss()(output, target)
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    try:
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        nan_grads = 0
        zero_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                    print(f"❌ {name} 包含NaN梯度")
                
                if grad_norm == 0:
                    zero_grads += 1
        
        if nan_grads > 0:
            print(f"❌ 发现 {nan_grads} 个参数的梯度为NaN")
            return False
        
        print(f"梯度范数统计:")
        print(f"  最小值: {min(grad_norms):.6f}")
        print(f"  最大值: {max(grad_norms):.6f}")
        print(f"  均值: {np.mean(grad_norms):.6f}")
        print(f"  标准差: {np.std(grad_norms):.6f}")
        print(f"  零梯度参数数量: {zero_grads}")
        
        if max(grad_norms) > 100:
            print("⚠️  发现较大的梯度值，可能存在梯度爆炸")
        
        print("✅ 梯度流动正常")
        
    except Exception as e:
        print(f"❌ 梯度流动测试 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_weight_initialization():
    """测试权重初始化"""
    print("\n" + "=" * 60)
    print("测试权重初始化")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SwinUNet(
        in_channels=3,
        out_channels=3,
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        skip_connections=True,
        use_fno_bottleneck=False
    ).to(device)
    
    # 检查权重初始化
    linear_weights = []
    conv_weights = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_std = param.std().item()
            weight_mean = param.mean().item()
            
            if 'Linear' in str(type(param)) or 'linear' in name.lower():
                linear_weights.append(weight_std)
            elif 'Conv' in str(type(param)) or 'conv' in name.lower():
                conv_weights.append(weight_std)
            
            # 检查异常值
            if weight_std > 1.0:
                print(f"⚠️  {name} 权重标准差较大: {weight_std:.6f}")
            if abs(weight_mean) > 0.1:
                print(f"⚠️  {name} 权重均值偏离零点: {weight_mean:.6f}")
    
    if linear_weights:
        print(f"Linear层权重标准差: 均值={np.mean(linear_weights):.6f}, 范围=[{min(linear_weights):.6f}, {max(linear_weights):.6f}]")
    
    if conv_weights:
        print(f"Conv层权重标准差: 均值={np.mean(conv_weights):.6f}, 范围=[{min(conv_weights):.6f}, {max(conv_weights):.6f}]")
    
    print("✅ 权重初始化检查完成")
    return True


def test_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 60)
    print("测试内存使用")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存测试")
        return True
    
    device = torch.device('cuda')
    
    # 清空缓存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # 创建模型
    model = SwinUNet(
        in_channels=3,
        out_channels=3,
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        skip_connections=True,
        use_fno_bottleneck=False
    ).to(device)
    
    model_memory = torch.cuda.memory_allocated() - initial_memory
    print(f"模型内存使用: {model_memory / 1024**2:.2f} MB")
    
    # 测试不同批次大小
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            before_forward = torch.cuda.memory_allocated()
            
            input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            after_forward = torch.cuda.memory_allocated()
            forward_memory = after_forward - before_forward
            
            print(f"批次大小 {batch_size}: 前向传播内存 {forward_memory / 1024**2:.2f} MB")
            
            # 清理
            del input_tensor, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            if "out of memory" in str(e):
                print(f"批次大小 {batch_size}: 内存不足")
            else:
                print(f"❌ 内存使用测试 执行失败: {e}")
                return False
    
    print("✅ 内存使用测试完成")
    return True


def main():
    """主测试函数"""
    print("开始测试新的对称Swin-UNet架构...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        ("架构测试", test_model_architecture),
        ("梯度流动测试", test_gradient_flow),
        ("权重初始化测试", test_weight_initialization),
        ("内存使用测试", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！新架构数值健康性良好。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)