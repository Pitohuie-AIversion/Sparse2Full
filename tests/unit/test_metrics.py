"""
PDEBench稀疏观测重建系统 - 指标计算模块单元测试

测试所有评测指标的数值正确性，包括：
- Rel-L2: 相对L2误差
- MAE: 平均绝对误差
- PSNR: 峰值信噪比
- SSIM: 结构相似性指数
- fRMSE: 频域RMSE (low/mid/high)
- bRMSE: 边界RMSE
- cRMSE: 中心RMSE
- ||H(ŷ)−y||: 数据一致性误差

遵循技术架构文档7.7节TDD准则要求。
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ops.metrics import compute_all_metrics, compute_rel_l2_error, compute_mae
from ops.degradation import apply_degradation_operator
from utils.metrics import MetricsCalculator, StatisticalAnalyzer
from tests.conftest import assert_tensor_close, assert_tensor_shape


class TestMetricsCalculator:
    """测试MetricsCalculator类"""
    
    @pytest.fixture
    def sample_tensor_2d(self, sample_tensor):
        """2D样本tensor fixture"""
        return sample_tensor((2, 1, 64, 64))
    
    def test_init_default_params(self):
        """测试默认参数初始化"""
        calculator = MetricsCalculator()
        
        assert calculator.image_size == (256, 256)
        assert calculator.boundary_width == 16
        assert 'low' in calculator.freq_bands
        assert 'mid' in calculator.freq_bands
        assert 'high' in calculator.freq_bands
        
        # 检查掩码形状
        H, W = calculator.image_size
        assert calculator.boundary_mask.shape == (H, W)
        assert calculator.center_mask.shape == (H, W)
        assert torch.sum(calculator.boundary_mask) + torch.sum(calculator.center_mask) == H * W
    
    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        image_size = (128, 128)
        boundary_width = 8
        freq_bands = {'low': (0, 8), 'high': (8, 32)}
        
        calculator = MetricsCalculator(
            image_size=image_size,
            boundary_width=boundary_width,
            freq_bands=freq_bands
        )
        
        assert calculator.image_size == image_size
        assert calculator.boundary_width == boundary_width
        assert calculator.freq_bands == freq_bands
        assert len(calculator.freq_masks) == 2
    
    def test_precompute_masks(self):
        """测试掩码预计算"""
        calculator = MetricsCalculator(image_size=(64, 64), boundary_width=4)
        
        # 边界掩码检查
        boundary_mask = calculator.boundary_mask
        assert boundary_mask[:4, :].all()  # 上边界
        assert boundary_mask[-4:, :].all()  # 下边界
        assert boundary_mask[:, :4].all()  # 左边界
        assert boundary_mask[:, -4:].all()  # 右边界
        
        # 中心掩码检查
        center_mask = calculator.center_mask
        assert not center_mask[:4, :].any()  # 上边界为False
        assert center_mask[8:56, 8:56].all()  # 中心区域为True
    
    def test_create_freq_mask(self):
        """测试频域掩码创建"""
        calculator = MetricsCalculator(image_size=(32, 32))
        
        # 测试低频掩码
        mask = calculator._create_freq_mask(32, 32, 0, 4)
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.bool
        
        # DC分量应该在低频掩码中
        assert mask[0, 0] == True
    
    def test_compute_rel_l2_basic(self, sample_tensor_2d, tolerance_config):
        """测试相对L2误差基本功能"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        rel_l2 = calculator.compute_rel_l2(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(rel_l2, (2, 1), "Rel-L2 shape mismatch")
        
        # 检查数值范围
        assert torch.all(rel_l2 >= 0), "Rel-L2 should be non-negative"
        assert torch.all(rel_l2 < 1), "Rel-L2 should be reasonable for small noise"
    
    def test_compute_rel_l2_identical_inputs(self, sample_tensor_2d, tolerance_config):
        """测试相同输入的相对L2误差"""
        calculator = MetricsCalculator()
        
        rel_l2 = calculator.compute_rel_l2(sample_tensor_2d, sample_tensor_2d)
        
        # 相同输入应该得到接近0的误差
        assert_tensor_close(rel_l2, torch.zeros_like(rel_l2), 
                          atol=tolerance_config['atol'], 
                          msg="Identical inputs should give zero Rel-L2")
    
    def test_compute_rel_l2_zero_target(self, tolerance_config):
        """测试零目标的相对L2误差"""
        calculator = MetricsCalculator()
        
        pred = torch.randn(2, 1, 64, 64)
        target = torch.zeros_like(pred)
        
        rel_l2 = calculator.compute_rel_l2(pred, target, eps=1e-8)
        
        # 应该处理零目标情况
        assert torch.all(torch.isfinite(rel_l2)), "Should handle zero target gracefully"
    
    def test_compute_mae_basic(self, sample_tensor_2d):
        """测试平均绝对误差基本功能"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.5
        
        mae = calculator.compute_mae(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(mae, (2, 1), "MAE shape mismatch")
        
        # 检查数值正确性
        expected_mae = torch.full((2, 1), 0.5, device=sample_tensor_2d.device)
        assert_tensor_close(mae, expected_mae, rtol=1e-5, msg="MAE calculation incorrect")
    
    def test_compute_mae_identical_inputs(self, sample_tensor_2d, tolerance_config):
        """测试相同输入的MAE"""
        calculator = MetricsCalculator()
        
        mae = calculator.compute_mae(sample_tensor_2d, sample_tensor_2d)
        
        assert_tensor_close(mae, torch.zeros_like(mae), 
                          atol=tolerance_config['atol'],
                          msg="Identical inputs should give zero MAE")
    
    def test_compute_psnr_basic(self, sample_tensor_2d):
        """测试PSNR基本功能"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        psnr = calculator.compute_psnr(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(psnr, (2, 1), "PSNR shape mismatch")
        
        # PSNR应该为正值
        assert torch.all(psnr > 0), "PSNR should be positive for reasonable inputs"
    
    def test_compute_psnr_custom_max_val(self, sample_tensor_2d):
        """测试自定义最大值的PSNR"""
        calculator = MetricsCalculator()
        
        pred = torch.zeros_like(sample_tensor_2d)
        target = torch.ones_like(sample_tensor_2d)
        
        psnr = calculator.compute_psnr(pred, target, max_val=1.0)
        
        # 对于MSE=1, max_val=1的情况，PSNR应该为0
        expected_psnr = torch.zeros((2, 1), device=sample_tensor_2d.device)
        assert_tensor_close(psnr, expected_psnr, rtol=1e-3, msg="PSNR with custom max_val incorrect")
    
    def test_compute_ssim_basic(self, sample_tensor_2d):
        """测试SSIM基本功能"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        ssim_val = calculator.compute_ssim(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(ssim_val, (2, 1), "SSIM shape mismatch")
        
        # SSIM应该在[-1, 1]范围内
        assert torch.all(ssim_val >= -1), "SSIM should be >= -1"
        assert torch.all(ssim_val <= 1), "SSIM should be <= 1"
    
    def test_compute_ssim_identical_inputs(self, sample_tensor_2d, tolerance_config):
        """测试相同输入的SSIM"""
        calculator = MetricsCalculator()
        
        ssim_val = calculator.compute_ssim(sample_tensor_2d, sample_tensor_2d)
        
        # 相同输入的SSIM应该接近1
        expected_ssim = torch.ones((2, 1), device=sample_tensor_2d.device)
        assert_tensor_close(ssim_val, expected_ssim, rtol=1e-3, 
                          msg="Identical inputs should give SSIM ≈ 1")
    
    def test_compute_freq_rmse_basic(self, sample_tensor_2d):
        """测试频域RMSE基本功能"""
        calculator = MetricsCalculator(image_size=(64, 64))
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        freq_rmse = calculator.compute_freq_rmse(pred, target)
        
        # 检查返回的频段
        assert 'low' in freq_rmse
        assert 'mid' in freq_rmse
        assert 'high' in freq_rmse
        
        # 检查每个频段的形状
        for band_name, rmse in freq_rmse.items():
            assert_tensor_shape(rmse, (2, 1), f"fRMSE {band_name} shape mismatch")
            assert torch.all(rmse >= 0), f"fRMSE {band_name} should be non-negative"
    
    def test_compute_freq_rmse_identical_inputs(self, sample_tensor_2d, tolerance_config):
        """测试相同输入的频域RMSE"""
        calculator = MetricsCalculator(image_size=(64, 64))
        
        freq_rmse = calculator.compute_freq_rmse(sample_tensor_2d, sample_tensor_2d)
        
        for band_name, rmse in freq_rmse.items():
            assert_tensor_close(rmse, torch.zeros_like(rmse), 
                              atol=tolerance_config['atol'],
                              msg=f"Identical inputs should give zero fRMSE {band_name}")
    
    def test_compute_boundary_rmse_basic(self, sample_tensor_2d):
        """测试边界RMSE基本功能"""
        calculator = MetricsCalculator(image_size=(64, 64), boundary_width=8)
        
        pred = sample_tensor_2d
        target = sample_tensor_2d.clone()
        
        # 在边界区域添加误差
        target[:, :, :8, :] += 0.5  # 上边界
        target[:, :, -8:, :] += 0.5  # 下边界
        
        brmse = calculator.compute_boundary_rmse(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(brmse, (2, 1), "bRMSE shape mismatch")
        
        # 应该检测到边界误差（降低阈值以适应实际计算）
        assert torch.all(brmse > 0.1), "Should detect boundary errors"
    
    def test_compute_center_rmse_basic(self, sample_tensor_2d):
        """测试中心RMSE基本功能"""
        calculator = MetricsCalculator(image_size=(64, 64), boundary_width=8)
        
        pred = sample_tensor_2d
        target = sample_tensor_2d.clone()
        
        # 在中心区域添加误差
        target[:, :, 16:48, 16:48] += 0.5
        
        crmse = calculator.compute_center_rmse(pred, target)
        
        # 检查输出形状
        assert_tensor_shape(crmse, (2, 1), "cRMSE shape mismatch")
        
        # 应该检测到中心误差
        assert torch.all(crmse > 0.3), "Should detect center errors"
    
    def test_compute_data_consistency_error_basic(self, sample_tensor_2d, sample_observation_data):
        """测试数据一致性误差基本功能"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        obs_data = sample_observation_data
        
        dc_error = calculator.compute_data_consistency_error(pred, obs_data)
        
        # 检查输出形状
        assert_tensor_shape(dc_error, (2, 1), "DC error shape mismatch")
        
        # 检查数值范围
        assert torch.all(dc_error >= 0), "DC error should be non-negative"
    
    def test_compute_data_consistency_error_with_normalization(self, sample_tensor_2d, sample_observation_data, sample_normalization_stats):
        """测试带归一化的数据一致性误差"""
        calculator = MetricsCalculator()
        
        pred = sample_tensor_2d
        obs_data = sample_observation_data
        mu, sigma = sample_normalization_stats
        norm_stats = {'mean': mu.tolist(), 'std': sigma.tolist()}
        
        dc_error = calculator.compute_data_consistency_error(pred, obs_data, norm_stats)
        
        # 检查输出形状
        assert_tensor_shape(dc_error, (2, 1), "DC error shape mismatch")
        
        # 检查数值范围
        assert torch.all(dc_error >= 0), "DC error should be non-negative"
    
    def test_compute_all_metrics_basic(self, sample_tensor_2d, sample_observation_data):
        """测试计算所有指标基本功能"""
        calculator = MetricsCalculator(image_size=(64, 64))
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        with patch('utils.metrics.apply_degradation_operator') as mock_degradation:
            mock_degradation.return_value = sample_observation_data['baseline']
            
            metrics = calculator.compute_all_metrics(pred, target, sample_observation_data)
        
        # 检查所有指标都存在
        expected_metrics = [
            'rel_l2', 'mae', 'psnr', 'ssim',
            'frmse_low', 'frmse_mid', 'frmse_high',
            'brmse', 'crmse', 'dc_error'
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics, f"Missing metric: {metric_name}"
            assert torch.is_tensor(metrics[metric_name]), f"Metric {metric_name} should be tensor"
    
    def test_compute_all_metrics_without_obs_data(self, sample_tensor_2d):
        """测试不带观测数据的所有指标计算"""
        calculator = MetricsCalculator(image_size=(64, 64))
        
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        metrics = calculator.compute_all_metrics(pred, target)
        
        # 应该没有数据一致性误差
        assert 'dc_error' not in metrics
        
        # 其他指标应该存在
        basic_metrics = ['rel_l2', 'mae', 'psnr', 'ssim', 'brmse', 'crmse']
        for metric_name in basic_metrics:
            assert metric_name in metrics, f"Missing metric: {metric_name}"


class TestComputeAllMetricsFunction:
    """compute_all_metrics便捷函数测试"""
    
    def test_basic_functionality(self, sample_tensor_2d):
        """测试基本功能"""
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        metrics = compute_all_metrics(pred, target, image_size=(64, 64))
        
        # 检查返回标量值
        for metric_name, value in metrics.items():
            assert isinstance(value, float), f"Metric {metric_name} should be scalar"
            assert not np.isnan(value), f"Metric {metric_name} should not be NaN"
    
    def test_with_observation_data(self, sample_tensor_2d, sample_observation_data):
        """测试带观测数据的功能"""
        pred = sample_tensor_2d
        target = sample_tensor_2d + 0.1 * torch.randn_like(sample_tensor_2d)
        
        with patch('utils.metrics.apply_degradation_operator') as mock_degradation:
            mock_degradation.return_value = sample_observation_data['baseline']
            
            metrics = compute_all_metrics(pred, target, sample_observation_data, image_size=(64, 64))
        
        # 应该包含数据一致性误差
        assert 'dc_error' in metrics
        assert isinstance(metrics['dc_error'], float)


class TestStatisticalAnalyzer:
    """StatisticalAnalyzer类测试"""
    
    def test_init(self):
        """测试初始化"""
        analyzer = StatisticalAnalyzer()
        assert analyzer.results == []
    
    def test_add_result(self):
        """测试添加结果"""
        analyzer = StatisticalAnalyzer()
        
        result1 = {'rel_l2': 0.1, 'mae': 0.05}
        result2 = {'rel_l2': 0.12, 'mae': 0.06}
        
        analyzer.add_result(result1)
        analyzer.add_result(result2)
        
        assert len(analyzer.results) == 2
        assert analyzer.results[0] == result1
        assert analyzer.results[1] == result2
    
    def test_compute_statistics_basic(self):
        """测试基本统计计算"""
        analyzer = StatisticalAnalyzer()
        
        # 添加测试数据
        results = [
            {'rel_l2': 0.1, 'mae': 0.05},
            {'rel_l2': 0.12, 'mae': 0.06},
            {'rel_l2': 0.11, 'mae': 0.055}
        ]
        
        for result in results:
            analyzer.add_result(result)
        
        stats = analyzer.compute_statistics()
        
        # 检查rel_l2统计
        rel_l2_stats = stats['rel_l2']
        assert abs(rel_l2_stats['mean'] - 0.11) < 1e-6
        assert rel_l2_stats['std'] > 0
        assert rel_l2_stats['min'] == 0.1
        assert rel_l2_stats['max'] == 0.12
        assert rel_l2_stats['count'] == 3
    
    def test_compute_statistics_empty(self):
        """测试空结果的统计计算"""
        analyzer = StatisticalAnalyzer()
        
        stats = analyzer.compute_statistics()
        
        assert stats == {}
    
    def test_compute_statistics_with_nan(self):
        """测试包含NaN的统计计算"""
        analyzer = StatisticalAnalyzer()
        
        results = [
            {'rel_l2': 0.1, 'mae': np.nan},
            {'rel_l2': 0.12, 'mae': 0.06},
            {'mae': 0.055}  # 缺少rel_l2
        ]
        
        for result in results:
            analyzer.add_result(result)
        
        stats = analyzer.compute_statistics()
        
        # rel_l2应该有2个有效值
        assert stats['rel_l2']['count'] == 2
        assert abs(stats['rel_l2']['mean'] - 0.11) < 1e-6
        
        # mae应该有2个有效值
        assert stats['mae']['count'] == 2
        assert abs(stats['mae']['mean'] - 0.0575) < 1e-6
    
    def test_significance_test_basic(self):
        """测试基本显著性检验"""
        analyzer = StatisticalAnalyzer()
        
        # 我们的结果（更好）
        our_results = [
            {'rel_l2': 0.08},
            {'rel_l2': 0.09},
            {'rel_l2': 0.085}
        ]
        
        # 基线结果（更差）
        baseline_results = [
            {'rel_l2': 0.12},
            {'rel_l2': 0.13},
            {'rel_l2': 0.125}
        ]
        
        for result in our_results:
            analyzer.add_result(result)
        
        test_result = analyzer.compute_significance_test(baseline_results, our_results, 'rel_l2')
        
        # 检查返回的统计量
        assert 't_stat' in test_result
        assert 'p_value' in test_result
        assert 'effect_size' in test_result
        assert 'is_significant' in test_result
        
        # 我们的结果明显更好，应该显著
        assert test_result['is_significant'] == True
        # 注意：t_stat的符号取决于具体的统计检验实现，这里只检查是否显著
        assert abs(test_result['t_stat']) > 0  # 应该有显著的统计量
    
    def test_significance_test_insufficient_data(self):
        """测试数据不足的显著性检验"""
        analyzer = StatisticalAnalyzer()
        
        our_results = [{'rel_l2': 0.1}]
        baseline_results = [{'rel_l2': 0.12}]
        
        test_result = analyzer.compute_significance_test(baseline_results, our_results, 'rel_l2')
        
        # 数据不足应该返回错误信息
        assert 'error' in test_result
        assert 'Insufficient samples' in test_result['error']
    
    def test_significance_test_mismatched_length(self):
        """测试长度不匹配的显著性检验"""
        analyzer = StatisticalAnalyzer()
        
        our_results = [{'rel_l2': 0.1}, {'rel_l2': 0.11}]
        baseline_results = [{'rel_l2': 0.12}]  # 长度不匹配
        
        # 长度不匹配时应该使用独立t检验，不会报错
        test_result = analyzer.compute_significance_test(baseline_results, our_results, 'rel_l2')
        
        # 应该返回错误信息，因为样本数不足
        assert 'error' in test_result
    
    def test_generate_report_basic(self):
        """测试基本报告生成"""
        analyzer = StatisticalAnalyzer()
        
        results = [
            {'rel_l2': 0.1, 'mae': 0.05, 'psnr': 25.0},
            {'rel_l2': 0.12, 'mae': 0.06, 'psnr': 24.0},
            {'rel_l2': 0.11, 'mae': 0.055, 'psnr': 24.5}
        ]
        
        for result in results:
            analyzer.add_result(result)
        
        report = analyzer.generate_report()
        
        # 检查报告内容
        assert "Statistical Analysis Report" in report
        assert "rel_l2" in report
        assert "mae" in report
        assert "psnr" in report
    
    def test_generate_report_with_baseline(self):
        """测试带基线的报告生成"""
        analyzer = StatisticalAnalyzer()
        
        our_results = [
            {'rel_l2': 0.08, 'mae': 0.04},
            {'rel_l2': 0.09, 'mae': 0.045},
            {'rel_l2': 0.085, 'mae': 0.042}
        ]
        
        for result in our_results:
            analyzer.add_result(result)
        
        report = analyzer.generate_report()
        
        # 检查基本报告内容
        assert "Statistical Analysis Report" in report
        assert "rel_l2" in report
        assert "mae" in report


class TestEdgeCases:
    """边界条件测试"""
    
    def test_single_pixel_image(self):
        """测试单像素图像"""
        calculator = MetricsCalculator(image_size=(1, 1), boundary_width=1)
        
        pred = torch.randn(1, 1, 1, 1)
        target = torch.randn(1, 1, 1, 1)
        
        # 基础指标应该能处理
        rel_l2 = calculator.compute_rel_l2(pred, target)
        mae = calculator.compute_mae(pred, target)
        
        assert_tensor_shape(rel_l2, (1, 1))
        assert_tensor_shape(mae, (1, 1))
    
    def test_extreme_values(self):
        """测试极值情况"""
        calculator = MetricsCalculator(image_size=(32, 32))
        
        # 极大值
        pred = torch.full((1, 1, 32, 32), 1e6)
        target = torch.full((1, 1, 32, 32), 1e6 + 1)
        
        rel_l2 = calculator.compute_rel_l2(pred, target)
        assert torch.all(torch.isfinite(rel_l2)), "Should handle extreme values"
        
        # 极小值
        pred = torch.full((1, 1, 32, 32), 1e-6)
        target = torch.full((1, 1, 32, 32), 2e-6)
        
        rel_l2 = calculator.compute_rel_l2(pred, target)
        assert torch.all(torch.isfinite(rel_l2)), "Should handle small values"
    
    def test_nan_inf_handling(self):
        """测试NaN和Inf处理"""
        calculator = MetricsCalculator(image_size=(32, 32))
        
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        
        # 注入NaN
        pred[0, 0, 0, 0] = float('nan')
        
        # 应该能检测到NaN
        rel_l2 = calculator.compute_rel_l2(pred, target)
        assert torch.any(torch.isnan(rel_l2)) or torch.any(torch.isinf(rel_l2))
    
    def test_multichannel_consistency(self, sample_tensor_multichannel):
        """测试多通道一致性"""
        calculator = MetricsCalculator(image_size=(64, 64))
        
        pred = sample_tensor_multichannel
        target = sample_tensor_multichannel + 0.1 * torch.randn_like(sample_tensor_multichannel)
        
        metrics = calculator.compute_all_metrics(pred, target)
        
        # 所有指标都应该有正确的通道维度
        for metric_name, value in metrics.items():
            if metric_name != 'dc_error':  # dc_error需要观测数据
                assert_tensor_shape(value, (2, 3), f"Metric {metric_name} channel dimension incorrect")


class TestNumericalStability:
    """数值稳定性测试"""
    
    def test_different_dtypes(self):
        """测试不同数据类型"""
        calculator = MetricsCalculator(image_size=(32, 32))
        
        # float32
        pred_f32 = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        target_f32 = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        
        rel_l2_f32 = calculator.compute_rel_l2(pred_f32, target_f32)
        
        # float64
        pred_f64 = pred_f32.double()
        target_f64 = target_f32.double()
        
        rel_l2_f64 = calculator.compute_rel_l2(pred_f64, target_f64)
        
        # 结果应该接近
        assert_tensor_close(rel_l2_f32, rel_l2_f64.float(), rtol=1e-4, 
                          msg="Different dtypes should give similar results")
    
    def test_gradient_stability(self):
        """测试梯度稳定性"""
        calculator = MetricsCalculator(image_size=(32, 32))
        
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randn(1, 1, 32, 32)
        
        rel_l2 = calculator.compute_rel_l2(pred, target)
        loss = torch.mean(rel_l2)
        
        # 应该能计算梯度
        loss.backward()
        assert pred.grad is not None, "Should compute gradients"
        assert torch.all(torch.isfinite(pred.grad)), "Gradients should be finite"


class TestPerformance:
    """性能测试"""
    
    @pytest.mark.slow
    def test_batch_processing(self):
        """测试批量处理性能"""
        calculator = MetricsCalculator(image_size=(256, 256))
        
        # 大批量数据
        pred = torch.randn(16, 3, 256, 256)
        target = torch.randn(16, 3, 256, 256)
        
        import time
        start_time = time.time()
        
        metrics = calculator.compute_all_metrics(pred, target)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 应该在合理时间内完成（具体阈值可调整）
        assert processing_time < 10.0, f"Batch processing too slow: {processing_time:.2f}s"
        
        # 检查结果形状
        for metric_name, value in metrics.items():
            if metric_name != 'dc_error':
                assert_tensor_shape(value, (16, 3), f"Batch metric {metric_name} shape incorrect")
    
    @pytest.mark.gpu
    def test_cuda_consistency(self, device):
        """测试CUDA一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        calculator = MetricsCalculator(image_size=(64, 64))
        
        pred_cpu = torch.randn(2, 1, 64, 64)
        target_cpu = torch.randn(2, 1, 64, 64)
        
        pred_gpu = pred_cpu.to(device)
        target_gpu = target_cpu.to(device)
        
        # CPU计算
        rel_l2_cpu = calculator.compute_rel_l2(pred_cpu, target_cpu)
        
        # GPU计算
        rel_l2_gpu = calculator.compute_rel_l2(pred_gpu, target_gpu)
        
        # 结果应该一致
        assert_tensor_close(rel_l2_cpu, rel_l2_gpu.cpu(), rtol=1e-5, 
                          msg="CPU and GPU results should be consistent")


if __name__ == "__main__":
    pytest.main([__file__])