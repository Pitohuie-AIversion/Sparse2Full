"""
稀疏注意力编码器单元测试

测试SparseAttentionEncoder和SparseSwinUNet的功能正确性
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch

# 测试稀疏注意力编码器
def test_sparse_attention_encoder_basic():
    """测试稀疏注意力编码器的基本功能"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    
    # 创建编码器
    encoder = SparseAttentionEncoder(
        in_channels=4,  # baseline(1) + coords(2) + mask(1)
        embed_dim=128,
        num_heads=4,
        sensor_dim=64,
        coord_dim=32,
        mask_dim=16,
        dropout=0.1,
        use_sparse_bias=True
    )
    
    # 创建测试输入
    B, C, H, W = 2, 4, 64, 64
    x = torch.randn(B, C, H, W)
    
    # 前向传播
    output = encoder(x)
    
    # 检查输出形状
    assert output.shape == (B, 128, H, W), f"Expected shape (B, 128, H, W), got {output.shape}"
    
    # 检查输出不是NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    print("✓ 基本功能测试通过")


def test_sparse_attention_encoder_with_separate_inputs():
    """测试使用单独输入的稀疏注意力编码器"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    
    encoder = SparseAttentionEncoder(
        in_channels=1,  # 只有baseline
        embed_dim=128,
        num_heads=4,
        sensor_dim=64,
        coord_dim=32,
        mask_dim=16,
        dropout=0.1
    )
    
    B, H, W = 2, 64, 64
    baseline = torch.randn(B, 1, H, W)
    coords = torch.randn(B, 2, H, W)
    mask = torch.ones(B, 1, H, W)
    
    # 前向传播
    output = encoder(baseline, coords=coords, mask=mask)
    
    # 检查输出形状
    assert output.shape == (B, 128, H, W)
    assert not torch.isnan(output).any()
    
    print("✓ 单独输入测试通过")


def test_sparse_attention_mask_creation():
    """测试稀疏注意力掩码创建"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    
    encoder = SparseAttentionEncoder(in_channels=4, embed_dim=128, num_heads=4)
    
    # 创建稀疏掩码（只有少数观测点）
    B, H, W = 1, 32, 32
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 10:15, 10:15] = 1.0  # 在中心区域设置观测点
    mask[:, :, 20:25, 20:25] = 1.0  # 在另一个区域设置观测点
    
    # 创建稀疏注意力掩码
    sparse_mask = encoder._create_sparse_attention_mask(mask, window_size=3)
    
    # 检查掩码形状
    assert sparse_mask.shape == (B, H * W, H * W)
    
    # 检查掩码值（观测点位置应该不是-inf）
    assert not torch.all(sparse_mask == float('-inf'))
    
    print("✓ 稀疏注意力掩码测试通过")


def test_sparse_swin_unet():
    """测试稀疏SwinUNet完整模型"""
    from models.sparse_attention_encoder import SparseSwinUNet
    
    # 创建模型
    model = SparseSwinUNet(
        in_channels=4,
        out_channels=1,
        img_size=128,  # 使用较小的图像大小以加快测试
        embed_dim=48,
        sparse_encoder_config={
            'embed_dim': 96,
            'num_heads': 4,
            'sensor_dim': 32,
            'coord_dim': 16,
            'mask_dim': 16,
            'dropout': 0.1,
            'use_sparse_bias': True
        },
        swin_unet_config={
            'depths': [1, 1, 3, 1],  # 减少层数以加快测试
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0
        }
    )
    
    # 创建测试输入
    B, C, H, W = 1, 4, 128, 128
    x = torch.randn(B, C, H, W)
    
    # 前向传播
    output = model(x)
    
    # 检查输出形状
    assert output.shape == (B, 1, H, W), f"Expected shape (B, 1, H, W), got {output.shape}"
    assert not torch.isnan(output).any()
    
    print("✓ 稀疏SwinUNet测试通过")


def test_gradient_flow():
    """测试梯度流是否正常"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    
    encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=64,
        num_heads=4,
        dropout=0.0  # 关闭dropout以便测试
    )
    
    B, C, H, W = 1, 4, 32, 32
    x = torch.randn(B, C, H, W, requires_grad=True)
    
    # 前向传播
    output = encoder(x)
    
    # 创建简单的损失函数
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查输入梯度
    assert x.grad is not None, "Input gradients are None"
    assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
    assert x.grad.abs().sum() > 0, "Input gradients are zero"
    
    print("✓ 梯度流测试通过")


def test_different_input_sizes():
    """测试不同输入大小的适应性"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    
    encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=64,
        num_heads=4
    )
    
    # 测试不同大小
    sizes = [(32, 32), (64, 64), (128, 128)]
    
    for H, W in sizes:
        x = torch.randn(1, 4, H, W)
        output = encoder(x)
        
        assert output.shape == (1, 64, H, W)
        assert not torch.isnan(output).any()
    
    print("✓ 不同输入大小测试通过")


def test_sparse_vs_dense_attention():
    """测试稀疏注意力与全注意力的性能差异"""
    from models.sparse_attention_encoder import SparseAttentionEncoder
    import time
    
    H, W = 64, 64
    B = 2
    
    # 创建稀疏和密集注意力编码器
    sparse_encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=128,
        num_heads=4,
        use_sparse_bias=True
    )
    
    dense_encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=128,
        num_heads=4,
        use_sparse_bias=False
    )
    
    # 创建稀疏输入（只有10%的观测点）
    x = torch.randn(B, 4, H, W)
    mask = torch.zeros(B, 1, H, W)
    # 随机设置10%的观测点
    num_obs = int(0.1 * H * W)
    for b in range(B):
        indices = np.random.choice(H * W, num_obs, replace=False)
        for idx in indices:
            i, j = idx // W, idx % W
            mask[b, 0, i, j] = 1.0
    
    # 更新输入的掩码通道
    x[:, 3:4, :, :] = mask
    
    # 测试推理时间
    def measure_time(encoder, x):
        start = time.time()
        with torch.no_grad():
            for _ in range(5):  # 运行5次取平均
                _ = encoder(x)
        end = time.time()
        return (end - start) / 5
    
    sparse_time = measure_time(sparse_encoder, x)
    dense_time = measure_time(dense_encoder, x)
    
    print(f"稀疏注意力平均时间: {sparse_time:.4f}s")
    print(f"全注意力平均时间: {dense_time:.4f}s")
    print(f"加速比: {dense_time / sparse_time:.2f}x")
    
    # 稀疏注意力应该更快
    assert sparse_time < dense_time, "稀疏注意力应该比全注意力更快"
    
    print("✓ 稀疏vs密集注意力性能测试通过")


if __name__ == "__main__":
    print("开始运行稀疏注意力编码器单元测试...")
    
    # 运行所有测试
    test_sparse_attention_encoder_basic()
    test_sparse_attention_encoder_with_separate_inputs()
    test_sparse_attention_mask_creation()
    test_sparse_swin_unet()
    test_gradient_flow()
    test_different_input_sizes()
    test_sparse_vs_dense_attention()
    
    print("\n🎉 所有稀疏注意力编码器测试通过！")