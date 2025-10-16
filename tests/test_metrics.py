"""
PDEBench稀疏观测重建系统 - 指标计算单元测试

测试各种评估指标的计算准确性，包括：
- Rel-L2, MAE, PSNR, SSIM
- fRMSE (low/mid/high频段)
- bRMSE (边界带误差)
- cRMSE (中心区域误差)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import (
    rel_l2_error, mae_error, psnr_metric, ssim_metric,
    frequency_rmse, boundary_rmse, center_rmse,
    compute_all_metrics
)


class TestBasicMetrics:
    """基础指标测试"""
    
    def test_rel_l2_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的Rel-L2误差"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        error = rel_l2_error(pred, target)
        
        assert error < tolerance_config['atol'], f"Perfect match should have near-zero error, got {error}"
    
    def test_rel_l2_known_values(self, device, tolerance_config):
        """测试已知值的Rel-L2误差"""
        # 创建简单的测试案例
        target = torch.ones(1, 1, 4, 4, device=device)
        pred = torch.ones(1, 1, 4, 4, device=device) * 2  # 预测值是真实值的2倍
        
        error = rel_l2_error(pred, target)
        expected = 1.0  # ||2*ones - ones|| / ||ones|| = ||ones|| / ||ones|| = 1
        
        assert abs(error - expected) < tolerance_config['rtol'], f"Expected {expected}, got {error}"
    
    def test_rel_l2_zero_target(self, device):
        """测试目标为零的情况"""
        target = torch.zeros(1, 1, 4, 4, device=device)
        pred = torch.ones(1, 1, 4, 4, device=device)
        
        # 应该处理除零情况
        error = rel_l2_error(pred, target)
        assert torch.isfinite(torch.tensor(error)), "Should handle zero target gracefully"
    
    def test_mae_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的MAE"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        error = mae_error(pred, target)
        
        assert error < tolerance_config['atol'], f"Perfect match should have near-zero MAE, got {error}"
    
    def test_mae_known_values(self, device, tolerance_config):
        """测试已知值的MAE"""
        target = torch.zeros(1, 1, 2, 2, device=device)
        pred = torch.ones(1, 1, 2, 2, device=device)
        
        error = mae_error(pred, target)
        expected = 1.0  # mean(|1 - 0|) = 1
        
        assert abs(error - expected) < tolerance_config['rtol'], f"Expected {expected}, got {error}"
    
    def test_psnr_perfect_match(self, sample_tensor):
        """测试完美匹配的PSNR"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        psnr = psnr_metric(pred, target)
        
        # 完美匹配应该有很高的PSNR
        assert psnr > 100, f"Perfect match should have high PSNR, got {psnr}"
    
    def test_psnr_known_values(self, device, tolerance_config):
        """测试已知值的PSNR"""
        # 创建简单的测试案例
        target = torch.zeros(1, 1, 2, 2, device=device)
        pred = torch.ones(1, 1, 2, 2, device=device) * 0.1
        
        psnr = psnr_metric(pred, target, max_val=1.0)
        
        # MSE = 0.01, PSNR = 10 * log10(1^2 / 0.01) = 20
        expected = 20.0
        assert abs(psnr - expected) < tolerance_config['rtol'], f"Expected {expected}, got {psnr}"
    
    def test_ssim_perfect_match(self, sample_tensor):
        """测试完美匹配的SSIM"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        ssim = ssim_metric(pred, target)
        
        # 完美匹配应该有SSIM=1
        assert abs(ssim - 1.0) < 1e-4, f"Perfect match should have SSIM=1, got {ssim}"
    
    def test_ssim_opposite_values(self, device):
        """测试相反值的SSIM"""
        target = torch.ones(1, 1, 32, 32, device=device)
        pred = -target
        
        ssim = ssim_metric(pred, target)
        
        # 相反值应该有较低的SSIM
        assert ssim < 0.5, f"Opposite values should have low SSIM, got {ssim}"


class TestFrequencyMetrics:
    """频域指标测试"""
    
    def test_frequency_rmse_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的频域RMSE"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        low_error = frequency_rmse(pred, target, freq_range='low')
        mid_error = frequency_rmse(pred, target, freq_range='mid')
        high_error = frequency_rmse(pred, target, freq_range='high')
        
        assert low_error < tolerance_config['atol'], f"Low freq error should be near zero, got {low_error}"
        assert mid_error < tolerance_config['atol'], f"Mid freq error should be near zero, got {mid_error}"
        assert high_error < tolerance_config['atol'], f"High freq error should be near zero, got {high_error}"
    
    def test_frequency_rmse_low_freq_signal(self, device, test_utils):
        """测试低频信号的频域RMSE"""
        # 创建低频信号
        pred = test_utils.create_structured_data(1, 1, 64, 64, device, pattern='low_freq')
        target = pred.clone()
        
        # 添加高频噪声到预测值
        noise = test_utils.create_structured_data(1, 1, 64, 64, device, pattern='high_freq') * 0.1
        pred_noisy = pred + noise
        
        low_error = frequency_rmse(pred_noisy, target, freq_range='low')
        high_error = frequency_rmse(pred_noisy, target, freq_range='high')
        
        # 低频误差应该小于高频误差
        assert low_error < high_error, f"Low freq error ({low_error}) should be less than high freq error ({high_error})"
    
    def test_frequency_rmse_different_ranges(self, sample_tensor):
        """测试不同频率范围的RMSE"""
        pred = sample_tensor((1, 1, 64, 64))
        target = sample_tensor((1, 1, 64, 64))
        
        # 测试所有频率范围
        for freq_range in ['low', 'mid', 'high']:
            error = frequency_rmse(pred, target, freq_range=freq_range)
            assert error >= 0, f"Frequency RMSE should be non-negative for {freq_range} range"
            assert torch.isfinite(torch.tensor(error)), f"Frequency RMSE should be finite for {freq_range} range"


class TestSpatialMetrics:
    """空间指标测试"""
    
    def test_boundary_rmse_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的边界RMSE"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        error = boundary_rmse(pred, target, boundary_width=8)
        
        assert error < tolerance_config['atol'], f"Perfect match should have near-zero boundary RMSE, got {error}"
    
    def test_boundary_rmse_edge_error(self, device, test_utils):
        """测试边界误差的边界RMSE"""
        # 创建中心区域正确，边界区域错误的数据
        target = test_utils.create_structured_data(1, 1, 64, 64, device, pattern='gaussian')
        pred = target.clone()
        
        # 在边界添加误差
        boundary_width = 8
        pred[:, :, :boundary_width, :] += 1.0  # 上边界
        pred[:, :, -boundary_width:, :] += 1.0  # 下边界
        pred[:, :, :, :boundary_width] += 1.0  # 左边界
        pred[:, :, :, -boundary_width:] += 1.0  # 右边界
        
        boundary_error = boundary_rmse(pred, target, boundary_width=boundary_width)
        center_error = center_rmse(pred, target, boundary_width=boundary_width)
        
        # 边界误差应该大于中心误差
        assert boundary_error > center_error, f"Boundary error ({boundary_error}) should be greater than center error ({center_error})"
    
    def test_center_rmse_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的中心RMSE"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        error = center_rmse(pred, target, boundary_width=8)
        
        assert error < tolerance_config['atol'], f"Perfect match should have near-zero center RMSE, got {error}"
    
    def test_boundary_width_effects(self, sample_tensor):
        """测试不同边界宽度的影响"""
        pred = sample_tensor((1, 1, 64, 64))
        target = sample_tensor((1, 1, 64, 64))
        
        # 测试不同边界宽度
        widths = [4, 8, 16]
        boundary_errors = []
        center_errors = []
        
        for width in widths:
            b_error = boundary_rmse(pred, target, boundary_width=width)
            c_error = center_rmse(pred, target, boundary_width=width)
            
            boundary_errors.append(b_error)
            center_errors.append(c_error)
            
            assert b_error >= 0, f"Boundary RMSE should be non-negative for width {width}"
            assert c_error >= 0, f"Center RMSE should be non-negative for width {width}"


class TestMetricsIntegration:
    """指标集成测试"""
    
    def test_compute_all_metrics_perfect_match(self, sample_tensor, tolerance_config):
        """测试完美匹配的所有指标"""
        pred = sample_tensor((2, 3, 64, 64))
        target = pred.clone()
        
        metrics = compute_all_metrics(pred, target)
        
        # 检查所有指标
        assert metrics['rel_l2'] < tolerance_config['atol'], "Rel-L2 should be near zero"
        assert metrics['mae'] < tolerance_config['atol'], "MAE should be near zero"
        assert metrics['psnr'] > 100, "PSNR should be high"
        assert abs(metrics['ssim'] - 1.0) < 1e-4, "SSIM should be near 1"
        
        # 频域指标
        assert metrics['frmse_low'] < tolerance_config['atol'], "Low freq RMSE should be near zero"
        assert metrics['frmse_mid'] < tolerance_config['atol'], "Mid freq RMSE should be near zero"
        assert metrics['frmse_high'] < tolerance_config['atol'], "High freq RMSE should be near zero"
        
        # 空间指标
        assert metrics['brmse'] < tolerance_config['atol'], "Boundary RMSE should be near zero"
        assert metrics['crmse'] < tolerance_config['atol'], "Center RMSE should be near zero"
    
    def test_compute_all_metrics_structure(self, sample_tensor):
        """测试所有指标的结构"""
        pred = sample_tensor((2, 3, 64, 64))
        target = sample_tensor((2, 3, 64, 64))
        
        metrics = compute_all_metrics(pred, target)
        
        # 检查返回的指标结构
        expected_keys = [
            'rel_l2', 'mae', 'psnr', 'ssim',
            'frmse_low', 'frmse_mid', 'frmse_high',
            'brmse', 'crmse'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
            assert torch.isfinite(torch.tensor(metrics[key])), f"Metric {key} should be finite"
    
    def test_metrics_batch_consistency(self, sample_tensor, tolerance_config):
        """测试批处理一致性"""
        # 创建批处理数据
        batch_pred = sample_tensor((4, 3, 32, 32))
        batch_target = sample_tensor((4, 3, 32, 32))
        
        # 计算批处理指标
        batch_metrics = compute_all_metrics(batch_pred, batch_target)
        
        # 计算单个样本指标的平均值
        individual_metrics = []
        for i in range(4):
            single_metrics = compute_all_metrics(
                batch_pred[i:i+1], batch_target[i:i+1]
            )
            individual_metrics.append(single_metrics)
        
        # 计算平均值
        avg_metrics = {}
        for key in batch_metrics.keys():
            avg_metrics[key] = np.mean([m[key] for m in individual_metrics])
        
        # 检查一致性（允许一定的数值误差）
        for key in batch_metrics.keys():
            diff = abs(batch_metrics[key] - avg_metrics[key])
            assert diff < tolerance_config['rtol'], f"Batch inconsistency for {key}: {diff}"


class TestMetricsEdgeCases:
    """指标边界情况测试"""
    
    def test_zero_input(self, device, tolerance_config):
        """测试零输入"""
        pred = torch.zeros(1, 1, 32, 32, device=device)
        target = torch.zeros(1, 1, 32, 32, device=device)
        
        metrics = compute_all_metrics(pred, target)
        
        # 零输入应该有特定的指标值
        assert metrics['rel_l2'] < tolerance_config['atol'], "Zero input Rel-L2 should be near zero"
        assert metrics['mae'] < tolerance_config['atol'], "Zero input MAE should be near zero"
    
    def test_constant_input(self, device):
        """测试常数输入"""
        pred = torch.ones(1, 1, 32, 32, device=device)
        target = torch.ones(1, 1, 32, 32, device=device) * 2
        
        metrics = compute_all_metrics(pred, target)
        
        # 常数输入应该有可预测的指标值
        assert metrics['rel_l2'] == 0.5, "Constant input Rel-L2 should be 0.5"
        assert metrics['mae'] == 1.0, "Constant input MAE should be 1.0"
    
    def test_single_pixel(self, device):
        """测试单像素输入"""
        pred = torch.tensor([[[[1.0]]]], device=device)
        target = torch.tensor([[[[2.0]]]], device=device)
        
        metrics = compute_all_metrics(pred, target)
        
        # 单像素应该能正常计算指标
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value)), f"Single pixel {key} should be finite"
    
    def test_large_values(self, device):
        """测试大数值"""
        pred = torch.ones(1, 1, 16, 16, device=device) * 1e6
        target = torch.ones(1, 1, 16, 16, device=device) * 1e6
        
        metrics = compute_all_metrics(pred, target)
        
        # 大数值应该能正常处理
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value)), f"Large values {key} should be finite"
            assert not torch.isnan(torch.tensor(value)), f"Large values {key} should not be NaN"
    
    def test_different_dtypes(self, device):
        """测试不同数据类型"""
        for dtype in [torch.float32, torch.float64]:
            pred = torch.randn(1, 1, 16, 16, device=device, dtype=dtype)
            target = torch.randn(1, 1, 16, 16, device=device, dtype=dtype)
            
            metrics = compute_all_metrics(pred, target)
            
            # 不同数据类型应该能正常计算
            for key, value in metrics.items():
                assert torch.isfinite(torch.tensor(value)), f"Dtype {dtype} {key} should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])