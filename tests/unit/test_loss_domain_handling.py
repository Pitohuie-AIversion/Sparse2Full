"""
测试损失函数值域处理

验证损失函数在z-score域和原值域的正确处理：
1. 重建损失在z-score域计算
2. DC损失和频谱损失在原值域计算
3. 反归一化函数正确性
"""

import torch
import numpy as np
from typing import Dict, Optional
from ops.loss import TotalLoss, SpectralLoss, DataConsistencyLoss, ReconstructionLoss
from ops.degradation import apply_degradation_operator
from ops.losses import _denormalize_tensor


def create_test_data():
    """创建测试数据"""
    torch.manual_seed(42)
    
    # 原值域数据
    gt_original = torch.randn(2, 3, 64, 64) * 10 + 5  # 均值5，标准差10
    pred_original = gt_original + torch.randn_like(gt_original) * 0.1
    
    # 归一化统计
    norm_stats = {
        'mean': torch.tensor([5.0, 5.0, 5.0]).view(1, 3, 1, 1),
        'std': torch.tensor([10.0, 10.0, 10.0]).view(1, 3, 1, 1)
    }
    
    # z-score域数据
    gt_zscore = (gt_original - norm_stats['mean']) / norm_stats['std']
    pred_zscore = (pred_original - norm_stats['mean']) / norm_stats['std']
    
    # 观测数据（在原值域）
    sr_params = {'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5}
    obs_data = {
        'degraded': apply_degradation_operator(gt_original, sr_params),
        'mask': torch.ones_like(gt_original[:, :1, :, :])
    }
    
    return {
        'gt_original': gt_original,
        'pred_original': pred_original,
        'gt_zscore': gt_zscore,
        'pred_zscore': pred_zscore,
        'obs_data': obs_data,
        'norm_stats': norm_stats
    }


def test_reconstruction_loss_domain():
    """测试重建损失在z-score域计算"""
    print("测试重建损失值域处理...")
    
    data = create_test_data()
    recon_loss = ReconstructionLoss(loss_type='l2')
    
    # 在z-score域计算重建损失
    loss_zscore = recon_loss(data['pred_zscore'], data['gt_zscore'])
    
    # 验证：应该在z-score域计算
    expected_loss = torch.nn.functional.mse_loss(data['pred_zscore'], data['gt_zscore'])
    
    assert torch.allclose(loss_zscore, expected_loss, atol=1e-6), \
        f"重建损失计算错误: {loss_zscore} vs {expected_loss}"
    
    print(f"✓ 重建损失在z-score域计算正确: {loss_zscore:.6f}")


def test_spectral_loss_domain():
    """测试频谱损失在原值域计算"""
    print("测试频谱损失值域处理...")
    
    data = create_test_data()
    spectral_loss = SpectralLoss(low_freq_modes=8)
    
    # 频谱损失应该在原值域计算
    # 需要先反归一化到原值域
    # 创建简单的反归一化函数
    def denormalize_simple(x_z, mean, std):
        return x_z * std + mean
    
    pred_original_from_zscore = denormalize_simple(
        data['pred_zscore'], 
        data['norm_stats']['mean'], 
        data['norm_stats']['std']
    )
    gt_original_from_zscore = denormalize_simple(
        data['gt_zscore'],
        data['norm_stats']['mean'],
        data['norm_stats']['std']
    )
    
    loss_original = spectral_loss(pred_original_from_zscore, gt_original_from_zscore)
    loss_direct = spectral_loss(data['pred_original'], data['gt_original'])
    
    # 验证：从z-score反归一化后的结果应该与原始数据一致
    assert torch.allclose(pred_original_from_zscore, data['pred_original'], atol=1e-5), \
        "反归一化结果不一致"
    
    assert torch.allclose(loss_original, loss_direct, atol=1e-5), \
        f"频谱损失计算不一致: {loss_original} vs {loss_direct}"
    
    print(f"✓ 频谱损失在原值域计算正确: {loss_original:.6f}")


def test_data_consistency_loss_domain():
    """测试数据一致性损失在原值域计算"""
    print("测试数据一致性损失值域处理...")
    
    data = create_test_data()
    
    # 创建反归一化函数
    def denormalize_fn(x_zscore):
        return x_zscore * data['norm_stats']['std'] + data['norm_stats']['mean']
    
    dc_loss = DataConsistencyLoss(denormalize_fn=denormalize_fn)
    
    # DC损失应该在原值域计算
    sr_params = {'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5}
    loss = dc_loss(data['pred_zscore'], data['obs_data']['degraded'], sr_params)
    
    # 手动验证：
    # 1. 将z-score域预测反归一化到原值域
    pred_original = denormalize_fn(data['pred_zscore'])
    # 2. 应用观测算子
    sr_params = {'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5}
    pred_degraded = apply_degradation_operator(pred_original, sr_params)
    # 3. 与观测数据比较
    expected_loss = torch.nn.functional.mse_loss(pred_degraded, data['obs_data']['degraded'])
    
    assert torch.allclose(loss, expected_loss, atol=1e-5), \
        f"数据一致性损失计算错误: {loss} vs {expected_loss}"
    
    print(f"✓ 数据一致性损失在原值域计算正确: {loss:.6f}")


def test_total_loss_domain_handling():
    """测试总损失的值域处理"""
    print("测试总损失值域处理...")
    
    data = create_test_data()
    
    # 创建反归一化函数
    def denormalize_fn(x_zscore):
        return x_zscore * data['norm_stats']['std'] + data['norm_stats']['mean']
    
    total_loss = TotalLoss(
        rec_weight=1.0,
        spec_weight=0.5,
        dc_weight=1.0,
        denormalize_fn=denormalize_fn
    )
    
    # 计算总损失
    sr_params = {'task': 'sr', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5}
    total_loss_value, loss_dict = total_loss(data['pred_zscore'], data['gt_zscore'], data['obs_data']['degraded'], sr_params)
    
    # 验证各组件
    assert 'reconstruction' in loss_dict, "缺少重建损失"
    assert 'spectral' in loss_dict, "缺少频谱损失"
    assert 'data_consistency' in loss_dict, "缺少数据一致性损失"
    assert 'total' in loss_dict, "缺少总损失"
    
    # 验证总损失计算
    expected_total = (
        1.0 * loss_dict['reconstruction'] +
        0.5 * loss_dict['spectral'] +
        1.0 * loss_dict['data_consistency']
    )
    
    assert torch.allclose(loss_dict['total'], expected_total, atol=1e-6), \
        f"总损失计算错误: {loss_dict['total']} vs {expected_total}"
    
    print(f"✓ 总损失计算正确:")
    print(f"  - 重建损失: {loss_dict['reconstruction']:.6f}")
    print(f"  - 频谱损失: {loss_dict['spectral']:.6f}")
    print(f"  - 数据一致性损失: {loss_dict['data_consistency']:.6f}")
    print(f"  - 总损失: {loss_dict['total']:.6f}")


def test_denormalization_correctness():
    """测试反归一化函数正确性"""
    print("测试反归一化函数正确性...")
    
    data = create_test_data()
    
    # 反归一化
    pred_recovered = data['pred_zscore'] * data['norm_stats']['std'] + data['norm_stats']['mean']
    gt_recovered = data['gt_zscore'] * data['norm_stats']['std'] + data['norm_stats']['mean']
    
    # 验证反归一化正确性
    assert torch.allclose(pred_recovered, data['pred_original'], atol=1e-5), \
        "预测值反归一化错误"
    
    assert torch.allclose(gt_recovered, data['gt_original'], atol=1e-5), \
        "真实值反归一化错误"
    
    print("✓ 反归一化函数正确")


def main():
    """主测试函数"""
    print("PDEBench稀疏观测重建系统 - 损失函数值域处理测试")
    print("=" * 60)
    
    try:
        test_denormalization_correctness()
        test_reconstruction_loss_domain()
        test_spectral_loss_domain()
        test_data_consistency_loss_domain()
        test_total_loss_domain_handling()
        
        print("\n" + "=" * 60)
        print("✓ 所有损失函数值域处理测试通过！")
        print("✓ 重建损失在z-score域计算")
        print("✓ 频谱损失和DC损失在原值域计算")
        print("✓ 反归一化函数工作正确")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()