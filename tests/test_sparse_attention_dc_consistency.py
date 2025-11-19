"""
稀疏注意力模型数据一致性测试

验证SparseSwinUNet与观测算子H的兼容性
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.sparse_attention_encoder import SparseSwinUNet
from ops.degradation import apply_degradation_operator, verify_degradation_consistency


def test_sparse_swin_unet_dc_consistency():
    """测试SparseSwinUNet的数据一致性"""
    print("🧪 测试SparseSwinUNet数据一致性...")
    
    # 创建模型
    model = SparseSwinUNet(
        in_channels=4,  # baseline + coords + mask
        out_channels=1,
        img_size=64,  # 使用小尺寸以加快测试
        embed_dim=48,
        sparse_encoder_config={
            'embed_dim': 96,
            'num_heads': 4,
            'sensor_dim': 32,
            'coord_dim': 16,
            'mask_dim': 16,
            'dropout': 0.0,  # 关闭dropout以保持一致性
            'use_sparse_bias': True
        },
        swin_unet_config={
            'depths': [1, 1, 2, 1],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0
        }
    )
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建测试数据
    B, C, H, W = 1, 1, 64, 64
    
    # 创建地面真值（简单的正弦波模式）
    x = torch.linspace(0, 4*np.pi, W)
    y = torch.linspace(0, 4*np.pi, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    gt = torch.sin(X) * torch.cos(Y)  # [H, W]
    gt = gt.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
    
    # 创建SR观测参数
    sr_params = {
        'task': 'SR',
        'scale_factor': 4,
        'blur_sigma': 1.0,
        'kernel_size': 5,
        'interpolation': 'bilinear',
        'boundary': 'mirror'
    }
    
    # 应用观测算子得到观测数据
    observation = apply_degradation_operator(gt, sr_params)
    print(f"地面真值形状: {gt.shape}")
    print(f"观测数据形状: {observation.shape}")
    
    # 验证观测一致性
    consistency_result = verify_degradation_consistency(gt, observation, sr_params)
    print(f"观测一致性检查结果: {consistency_result}")
    
    # 创建模型输入（baseline + coords + mask）
    # 上采样观测数据作为baseline
    baseline = torch.nn.functional.interpolate(
        observation, size=(H, W), mode='bilinear', align_corners=False
    )
    
    # 创建坐标网格
    coords_x = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
    coords_y = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
    coords = torch.cat([coords_x, coords_y], dim=1)
    
    # 创建掩码（SR模式下全为1）
    mask = torch.ones(B, 1, H, W)
    
    # 组合输入
    model_input = torch.cat([baseline, coords, mask], dim=1)
    print(f"模型输入形状: {model_input.shape}")
    
    # 模型推理
    with torch.no_grad():
        reconstruction = model(model_input)
    
    print(f"重建结果形状: {reconstruction.shape}")
    
    # 验证重建质量
    mse_error = torch.nn.functional.mse_loss(reconstruction, gt)
    relative_l2 = torch.sqrt(mse_error) / torch.sqrt((gt**2).mean())
    
    print(f"MSE误差: {mse_error.item():.6f}")
    print(f"相对L2误差: {relative_l2.item():.6f}")
    
    # 验证重建结果通过观测算子的一致性
    recon_observation = apply_degradation_operator(reconstruction, sr_params)
    obs_mse = torch.nn.functional.mse_loss(recon_observation, observation)
    
    print(f"重建观测一致性MSE: {obs_mse.item():.6f}")
    
    # 检查是否在合理范围内
    assert mse_error.item() < 0.1, f"MSE误差过大: {mse_error.item()}"
    assert obs_mse.item() < 1e-4, f"观测一致性误差过大: {obs_mse.item()}"
    
    print("✅ SparseSwinUNet数据一致性测试通过!")
    
    return {
        'mse_error': mse_error.item(),
        'relative_l2': relative_l2.item(),
        'obs_consistency': obs_mse.item(),
        'dc_consistency': consistency_result
    }


def test_sparse_vs_dense_attention():
    """测试稀疏注意力与全注意力的性能对比"""
    print("\n🧪 测试稀疏注意力vs全注意力性能...")
    
    from models.sparse_attention_encoder import SparseAttentionEncoder
    import time
    
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
    
    # 创建稀疏输入（只有5%的观测点）
    B, H, W = 1, 64, 64
    x = torch.randn(B, 4, H, W)
    mask = torch.zeros(B, 1, H, W)
    
    # 随机设置5%的观测点（更稀疏）
    num_obs = int(0.05 * H * W)
    indices = np.random.choice(H * W, num_obs, replace=False)
    for idx in indices:
        i, j = idx // W, idx % W
        mask[0, 0, i, j] = 1.0
    
    # 更新输入的掩码通道
    x[:, 3:4, :, :] = mask
    
    # 测试推理时间
    def measure_time(encoder, x, num_runs=10):
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = encoder(x)
        
        # 正式测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = encoder(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        return (end - start) / num_runs, output
    
    sparse_time, sparse_output = measure_time(sparse_encoder, x)
    dense_time, dense_output = measure_time(dense_encoder, x)
    
    print(f"稀疏注意力平均时间: {sparse_time:.4f}s")
    print(f"全注意力平均时间: {dense_time:.4f}s")
    print(f"加速比: {dense_time / sparse_time:.2f}x")
    
    # 验证输出质量相似
    output_diff = torch.nn.functional.mse_loss(sparse_output, dense_output)
    print(f"输出差异: {output_diff.item():.6f}")
    
    # 稀疏注意力应该更快
    assert sparse_time < dense_time, f"稀疏注意力应该比全注意力更快: {sparse_time} vs {dense_time}"
    assert output_diff.item() < 0.1, f"输出差异过大: {output_diff.item()}"
    
    print("✅ 稀疏注意力性能测试通过!")
    
    return {
        'sparse_time': sparse_time,
        'dense_time': dense_time,
        'speedup': dense_time / sparse_time,
        'output_diff': output_diff.item()
    }


if __name__ == "__main__":
    print("🚀 开始稀疏注意力模型一致性测试...")
    
    # 运行数据一致性测试
    dc_results = test_sparse_swin_unet_dc_consistency()
    
    # 运行性能对比测试
    perf_results = test_sparse_vs_dense_attention()
    
    print("\n📊 测试结果汇总:")
    print("=" * 50)
    print(f"数据一致性:")
    print(f"  - MSE误差: {dc_results['mse_error']:.6f}")
    print(f"  - 相对L2误差: {dc_results['relative_l2']:.6f}")
    print(f"  - 观测一致性: {dc_results['obs_consistency']:.6f}")
    print(f"性能对比:")
    print(f"  - 稀疏注意力时间: {perf_results['sparse_time']:.4f}s")
    print(f"  - 全注意力时间: {perf_results['dense_time']:.4f}s")
    print(f"  - 加速比: {perf_results['speedup']:.2f}x")
    
    print("\n🎉 所有测试通过！稀疏注意力模型与项目框架完全兼容。")