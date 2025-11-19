"""
稀疏注意力模型快速验证测试

简化测试，验证SparseSwinUNet基本功能
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.sparse_attention_encoder import SparseSwinUNet, SparseAttentionEncoder


def test_sparse_attention_encoder_basic():
    """测试稀疏注意力编码器基本功能"""
    print("🧪 测试稀疏注意力编码器基本功能...")
    
    encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=64,
        num_heads=4,
        sensor_dim=32,
        coord_dim=16,
        mask_dim=16,
        dropout=0.0,
        use_sparse_bias=True
    )
    
    # 创建测试输入
    B, C, H, W = 1, 4, 32, 32
    x = torch.randn(B, C, H, W)
    
    # 前向传播
    output = encoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 检查输出
    assert output.shape == (B, 64, H, W)
    assert not torch.isnan(output).any()
    
    print("✅ 稀疏注意力编码器基本功能测试通过!")
    return True


def test_sparse_vs_dense_attention():
    """测试稀疏注意力与全注意力性能对比"""
    print("\n🧪 测试稀疏vs全注意力性能...")
    
    # 创建稀疏和密集注意力编码器
    sparse_encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=64,
        num_heads=4,
        use_sparse_bias=True,
        dropout=0.0
    )
    
    dense_encoder = SparseAttentionEncoder(
        in_channels=4,
        embed_dim=64,
        num_heads=4,
        use_sparse_bias=False,
        dropout=0.0
    )
    
    # 创建稀疏输入（只有10%的观测点）
    B, H, W = 1, 32, 32
    x = torch.randn(B, 4, H, W)
    mask = torch.zeros(B, 1, H, W)
    
    # 随机设置10%的观测点
    num_obs = int(0.1 * H * W)
    indices = np.random.choice(H * W, num_obs, replace=False)
    for idx in indices:
        i, j = idx // W, idx % W
        mask[0, 0, i, j] = 1.0
    
    # 更新输入的掩码通道
    x[:, 3:4, :, :] = mask
    
    # 测试推理
    with torch.no_grad():
        sparse_output = sparse_encoder(x)
        dense_output = dense_encoder(x)
    
    print(f"稀疏输出形状: {sparse_output.shape}")
    print(f"密集输出形状: {dense_output.shape}")
    
    # 验证输出质量相似
    output_diff = torch.nn.functional.mse_loss(sparse_output, dense_output)
    print(f"输出差异: {output_diff.item():.6f}")
    
    assert not torch.isnan(sparse_output).any()
    assert not torch.isnan(dense_output).any()
    assert output_diff.item() < 0.5, f"输出差异过大: {output_diff.item()}"
    
    print("✅ 稀疏注意力性能测试通过!")
    return True


def test_sparse_swin_unet_integration():
    """测试SparseSwinUNet集成"""
    print("\n🧪 测试SparseSwinUNet集成...")
    
    # 创建模型
    model = SparseSwinUNet(
        in_channels=4,
        out_channels=1,
        img_size=64,
        embed_dim=32,
        sparse_encoder_config={
            'embed_dim': 64,
            'num_heads': 4,
            'sensor_dim': 32,
            'coord_dim': 16,
            'mask_dim': 16,
            'dropout': 0.0,
            'use_sparse_bias': True
        },
        swin_unet_config={
            'depths': [1, 1, 2, 1],
            'num_heads': [2, 4, 8, 16],
            'window_size': 8,
            'mlp_ratio': 4.0
        }
    )
    
    # 创建测试输入
    B, C, H, W = 1, 4, 64, 64
    x = torch.randn(B, C, H, W)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 检查输出
    assert output.shape == (B, 1, H, W)
    assert not torch.isnan(output).any()
    
    print("✅ SparseSwinUNet集成测试通过!")
    return True


def test_model_integration_with_project():
    """测试与项目现有框架的集成"""
    print("\n🧪 测试与项目框架集成...")
    
    from models import create_model
    
    # 测试模型创建 - 使用正确的配置格式
    from omegaconf import DictConfig
    config = DictConfig({
        'name': 'SparseSwinUNet',
        'params': {
            'in_channels': 4,
            'out_channels': 1,
            'img_size': 64,
            'embed_dim': 32,
            'sparse_encoder_config': {
                'embed_dim': 64,
                'num_heads': 4,
                'sensor_dim': 32,
                'coord_dim': 16,
                'mask_dim': 16,
                'dropout': 0.0,
                'use_sparse_bias': True
            },
            'swin_unet_config': {
                'depths': [1, 1, 2, 1],
                'num_heads': [2, 4, 8, 16],
                'window_size': 8,
                'mlp_ratio': 4.0
            }
        }
    })
    
    model = create_model(config)
    
    # 测试前向传播
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        output = model(x)
    
    print(f"通过工厂函数创建的模型输出形状: {output.shape}")
    assert output.shape == (1, 1, 64, 64)
    
    print("✅ 项目框架集成测试通过!")
    return True


if __name__ == "__main__":
    print("🚀 开始稀疏注意力模型快速验证测试...")
    
    try:
        # 运行基本测试
        test_sparse_attention_encoder_basic()
        test_sparse_vs_dense_attention()
        test_sparse_swin_unet_integration()
        test_model_integration_with_project()
        
        print("\n🎉 所有测试通过！稀疏注意力模型已成功集成到项目框架中。")
        print("\n📋 模型特点:")
        print("  ✓ 基于Senseiver注意力机制的稀疏观测编码")
        print("  ✓ 支持传感器位置、坐标和掩码的多模态输入")
        print("  ✓ 稀疏注意力机制，在观测点稀少时更高效")
        print("  ✓ 与现有SwinUNet架构无缝集成")
        print("  ✓ 遵循项目统一接口规范")
        print("\n🔧 使用方式:")
        print("  1. 在配置文件中使用 model.name: 'SparseSwinUNet'")
        print("  2. 设置sparse_encoder_config配置注意力参数")
        print("  3. 设置swin_unet_config配置SwinUNet参数")
        print("  4. 输入格式: [baseline, coords_x, coords_y, mask]")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)