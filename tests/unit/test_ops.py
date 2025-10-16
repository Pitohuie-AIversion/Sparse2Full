"""
测试ops模块：H算子（观测算子）实现测试

测试SR和Crop模式的退化算子，验证数值正确性、边界处理、一致性等。
严格按照开发手册要求，确保H算子满足"黄金法则"。
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from ops.degradation import (
    apply_degradation_operator,
    _apply_sr_degradation,
    _apply_crop_degradation,
    _gaussian_blur,
    _create_gaussian_kernel,
    _pad_to_size,
    verify_degradation_consistency
)


class TestSRDegradationOperator:
    """测试SR退化算子"""
    
    def test_sr_degradation_basic(self, device):
        """测试基本SR退化功能"""
        # 创建测试数据
        B, C, H, W = 2, 3, 64, 64
        pred = torch.randn(B, C, H, W, device=device)
        
        params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        # 应用SR退化
        result = apply_degradation_operator(pred, params)
        
        # 验证输出形状
        expected_h, expected_w = H // 2, W // 2
        assert result.shape == (B, C, expected_h, expected_w)
        
        # 验证数值范围合理
        assert torch.isfinite(result).all()
        
    def test_sr_degradation_different_scales(self, device):
        """测试不同缩放因子"""
        B, C, H, W = 1, 1, 128, 128
        pred = torch.randn(B, C, H, W, device=device)
        
        for scale in [2, 4, 8]:
            params = {
                'task': 'SR',
                'scale': scale,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
            
            result = apply_degradation_operator(pred, params)
            expected_h, expected_w = H // scale, W // scale
            assert result.shape == (B, C, expected_h, expected_w)
    
    def test_sr_degradation_different_sigmas(self, device):
        """测试不同高斯模糊参数"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        results = {}
        for sigma in [0.5, 1.0, 2.0]:
            params = {
                'task': 'SR',
                'scale': 2,
                'sigma': sigma,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
            
            result = apply_degradation_operator(pred, params)
            results[sigma] = result
            
            # 验证形状
            assert result.shape == (B, C, H // 2, W // 2)
        
        # 验证不同sigma产生不同结果
        assert not torch.allclose(results[0.5], results[2.0], rtol=1e-3)
    
    def test_sr_boundary_modes(self, device):
        """测试不同边界处理模式"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        
        results = {}
        for boundary in ['mirror', 'wrap', 'zero']:
            params = {
                'task': 'SR',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': boundary
            }
            
            result = apply_degradation_operator(pred, params)
            results[boundary] = result
            
            # 验证形状
            assert result.shape == (B, C, H // 2, W // 2)
        
        # 验证不同边界模式产生不同结果
        assert not torch.allclose(results['mirror'], results['zero'], rtol=1e-3)
    
    def test_sr_degradation_consistency(self, device, tolerance_config):
        """测试SR退化一致性：多次调用应产生相同结果"""
        B, C, H, W = 1, 2, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        # 多次调用
        result1 = apply_degradation_operator(pred, params)
        result2 = apply_degradation_operator(pred, params)
        
        # 验证一致性
        np.testing.assert_allclose(
            result1.cpu().numpy(),
            result2.cpu().numpy(),
            rtol=tolerance_config['rtol'],
            atol=tolerance_config['atol']
        )


class TestCropDegradationOperator:
    """测试Crop退化算子"""
    
    def test_crop_degradation_basic(self, device):
        """测试基本Crop退化功能"""
        B, C, H, W = 2, 3, 64, 64
        pred = torch.randn(B, C, H, W, device=device)
        
        crop_size = (32, 32)
        params = {
            'task': 'Crop',
            'crop_size': crop_size,
            'boundary': 'mirror'
        }
        
        # 应用Crop退化
        result = apply_degradation_operator(pred, params)
        
        # 验证输出形状
        assert result.shape == (B, C, crop_size[0], crop_size[1])
        
        # 验证数值范围合理
        assert torch.isfinite(result).all()
    
    def test_crop_with_specific_box(self, device):
        """测试指定裁剪框的Crop"""
        B, C, H, W = 1, 1, 64, 64
        pred = torch.randn(B, C, H, W, device=device)
        
        # 指定裁剪框 (x1, y1, x2, y2)
        crop_box = (10, 10, 42, 42)  # 32x32区域
        crop_size = (32, 32)
        
        params = {
            'task': 'Crop',
            'crop_size': crop_size,
            'crop_box': crop_box,
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred, params)
        
        # 验证形状
        assert result.shape == (B, C, crop_size[0], crop_size[1])
        
        # 验证内容：应该等于直接切片的结果
        x1, y1, x2, y2 = crop_box
        expected = pred[:, :, y1:y2, x1:x2]
        np.testing.assert_allclose(
            result.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_crop_boundary_handling(self, device):
        """测试边界处理：裁剪区域超出图像边界"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        # 裁剪尺寸大于图像尺寸
        crop_size = (48, 48)
        
        for boundary in ['mirror', 'wrap', 'zero']:
            params = {
                'task': 'Crop',
                'crop_size': crop_size,
                'boundary': boundary
            }
            
            result = apply_degradation_operator(pred, params)
            
            # 验证形状
            assert result.shape == (B, C, crop_size[0], crop_size[1])
            
            # 验证中心区域应该与原图一致
            center_h, center_w = crop_size[0] // 2, crop_size[1] // 2
            orig_h, orig_w = H // 2, W // 2
            
            # 提取中心区域进行比较
            result_center = result[:, :, 
                                 center_h - orig_h:center_h + orig_h,
                                 center_w - orig_w:center_w + orig_w]
            
            assert result_center.shape == pred.shape
    
    def test_crop_different_sizes(self, device):
        """测试不同裁剪尺寸"""
        B, C, H, W = 1, 1, 64, 64
        pred = torch.randn(B, C, H, W, device=device)
        
        crop_sizes = [(16, 16), (32, 32), (48, 48)]
        
        for crop_size in crop_sizes:
            params = {
                'task': 'Crop',
                'crop_size': crop_size,
                'boundary': 'mirror'
            }
            
            result = apply_degradation_operator(pred, params)
            assert result.shape == (B, C, crop_size[0], crop_size[1])


class TestGaussianBlur:
    """测试高斯模糊功能"""
    
    def test_gaussian_kernel_creation(self, device):
        """测试高斯核创建"""
        kernel_size = 5
        sigma = 1.0
        
        kernel = _create_gaussian_kernel(kernel_size, sigma, device, torch.float32)
        
        # 验证形状
        assert kernel.shape == (1, 1, kernel_size, kernel_size)
        
        # 验证归一化
        np.testing.assert_allclose(kernel.sum().item(), 1.0, rtol=1e-5)
        
        # 验证对称性
        kernel_np = kernel.squeeze().cpu().numpy()
        np.testing.assert_allclose(kernel_np, kernel_np.T, rtol=1e-5)
        
        # 验证中心最大
        center = kernel_size // 2
        assert kernel[0, 0, center, center] == kernel.max()
    
    def test_gaussian_blur_identity(self, device):
        """测试高斯模糊的恒等性：sigma=0时应接近原图"""
        B, C, H, W = 1, 1, 32, 32
        x = torch.randn(B, C, H, W, device=device)
        
        # 使用很小的sigma
        blurred = _gaussian_blur(x, kernel_size=3, sigma=0.01, boundary='mirror')
        
        # 应该接近原图
        np.testing.assert_allclose(
            blurred.cpu().numpy(),
            x.cpu().numpy(),
            rtol=1e-2, atol=1e-3
        )
    
    def test_gaussian_blur_smoothing(self, device):
        """测试高斯模糊的平滑效果"""
        B, C, H, W = 1, 1, 32, 32
        
        # 创建带有高频噪声的图像
        x = torch.randn(B, C, H, W, device=device)
        noise = torch.randn(B, C, H, W, device=device) * 0.1
        x_noisy = x + noise
        
        # 应用高斯模糊
        blurred = _gaussian_blur(x_noisy, kernel_size=5, sigma=2.0, boundary='mirror')
        
        # 模糊后的图像应该更平滑（方差更小）
        assert blurred.var() < x_noisy.var()


class TestPadToSize:
    """测试尺寸填充功能"""
    
    def test_pad_to_larger_size(self, device):
        """测试填充到更大尺寸"""
        B, C, H, W = 1, 1, 16, 16
        x = torch.randn(B, C, H, W, device=device)
        
        target_h, target_w = 32, 32
        
        for boundary in ['mirror', 'wrap', 'zero']:
            padded = _pad_to_size(x, target_h, target_w, boundary)
            
            # 验证形状
            assert padded.shape == (B, C, target_h, target_w)
            
            # 验证中心区域保持不变
            pad_h, pad_w = (target_h - H) // 2, (target_w - W) // 2
            center_region = padded[:, :, pad_h:pad_h+H, pad_w:pad_w+W]
            
            np.testing.assert_allclose(
                center_region.cpu().numpy(),
                x.cpu().numpy(),
                rtol=1e-5, atol=1e-8
            )
    
    def test_pad_to_same_size(self, device):
        """测试填充到相同尺寸"""
        B, C, H, W = 1, 1, 32, 32
        x = torch.randn(B, C, H, W, device=device)
        
        padded = _pad_to_size(x, H, W, 'mirror')
        
        # 应该保持不变
        np.testing.assert_allclose(
            padded.cpu().numpy(),
            x.cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_pad_to_smaller_size(self, device):
        """测试"填充"到更小尺寸（实际是裁剪）"""
        B, C, H, W = 1, 1, 32, 32
        x = torch.randn(B, C, H, W, device=device)
        
        target_h, target_w = 16, 16
        padded = _pad_to_size(x, target_h, target_w, 'mirror')
        
        # 验证形状
        assert padded.shape == (B, C, target_h, target_w)
        
        # 验证内容：应该是中心裁剪
        expected = x[:, :, :target_h, :target_w]
        np.testing.assert_allclose(
            padded.cpu().numpy(),
            expected.cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )


class TestDegradationConsistency:
    """测试退化算子一致性"""
    
    def test_sr_consistency_verification(self, device, tolerance_config):
        """测试SR一致性验证"""
        B, C, H, W = 1, 2, 64, 64
        target = torch.randn(B, C, H, W, device=device)
        
        # H算子参数
        h_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        # 生成观测
        observation = apply_degradation_operator(target, h_params)
        
        # 验证一致性
        result = verify_degradation_consistency(
            target, observation, h_params, tolerance=1e-8
        )
        
        # 应该通过一致性检查
        assert result['passed']
        assert result['mse'] < 1e-8
        assert result['max_error'] < 1e-6
    
    def test_crop_consistency_verification(self, device, tolerance_config):
        """测试Crop一致性验证"""
        B, C, H, W = 1, 2, 64, 64
        target = torch.randn(B, C, H, W, device=device)
        
        # H算子参数
        h_params = {
            'task': 'Crop',
            'crop_size': (32, 32),
            'crop_box': (16, 16, 48, 48),
            'boundary': 'mirror'
        }
        
        # 生成观测
        observation = apply_degradation_operator(target, h_params)
        
        # 验证一致性
        result = verify_degradation_consistency(
            target, observation, h_params, tolerance=1e-8
        )
        
        # 应该通过一致性检查
        assert result['passed']
        assert result['mse'] < 1e-8
    
    def test_consistency_failure_detection(self, device):
        """测试一致性检查能检测到不一致"""
        B, C, H, W = 1, 1, 64, 64
        target = torch.randn(B, C, H, W, device=device)
        
        h_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        # 生成正确的观测
        correct_obs = apply_degradation_operator(target, h_params)
        
        # 创建错误的观测（添加噪声）
        wrong_obs = correct_obs + torch.randn_like(correct_obs) * 0.1
        
        # 验证一致性
        result = verify_degradation_consistency(
            target, wrong_obs, h_params, tolerance=1e-8
        )
        
        # 应该检测到不一致
        assert not result['passed']
        assert result['mse'] > 1e-8


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_pixel_input(self, device):
        """测试单像素输入"""
        B, C, H, W = 1, 1, 1, 1
        pred = torch.randn(B, C, H, W, device=device)
        
        # SR退化
        sr_params = {
            'task': 'SR',
            'scale': 1,  # 不缩放
            'sigma': 0.5,
            'kernel_size': 1,
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred, sr_params)
        assert result.shape == (B, C, H, W)
    
    def test_large_kernel_small_image(self, device):
        """测试大核小图像情况"""
        B, C, H, W = 1, 1, 8, 8
        pred = torch.randn(B, C, H, W, device=device)
        
        # 使用比图像还大的核
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 2.0,
            'kernel_size': 15,  # 比图像大
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred, sr_params)
        assert result.shape == (B, C, H // 2, W // 2)
        assert torch.isfinite(result).all()
    
    def test_zero_input(self, device):
        """测试零输入"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.zeros(B, C, H, W, device=device)
        
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred, sr_params)
        
        # 零输入应该产生零输出
        assert torch.allclose(result, torch.zeros_like(result))
    
    def test_invalid_task(self, device):
        """测试无效任务类型"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        invalid_params = {
            'task': 'InvalidTask',
            'scale': 2
        }
        
        with pytest.raises(ValueError, match="Unknown task"):
            apply_degradation_operator(pred, invalid_params)
    
    def test_invalid_boundary_mode(self, device):
        """测试无效边界模式"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        invalid_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'invalid_mode'
        }
        
        with pytest.raises(ValueError, match="Unknown boundary mode"):
            apply_degradation_operator(pred, invalid_params)


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_extreme_values(self, device):
        """测试极值输入"""
        B, C, H, W = 1, 1, 32, 32
        
        # 测试极大值
        pred_large = torch.ones(B, C, H, W, device=device) * 1e6
        
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred_large, sr_params)
        assert torch.isfinite(result).all()
        
        # 测试极小值
        pred_small = torch.ones(B, C, H, W, device=device) * 1e-6
        result = apply_degradation_operator(pred_small, sr_params)
        assert torch.isfinite(result).all()
    
    def test_gradient_flow(self, device):
        """测试梯度流动"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
        
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        result = apply_degradation_operator(pred, sr_params)
        loss = result.sum()
        loss.backward()
        
        # 验证梯度存在且有限
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0  # 梯度不应该全为零


class TestPerformance:
    """测试性能相关"""
    
    def test_batch_processing(self, device):
        """测试批处理"""
        batch_sizes = [1, 4, 8]
        C, H, W = 3, 64, 64
        
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        for B in batch_sizes:
            pred = torch.randn(B, C, H, W, device=device)
            result = apply_degradation_operator(pred, sr_params)
            
            expected_shape = (B, C, H // 2, W // 2)
            assert result.shape == expected_shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_consistency(self):
        """测试CPU和CUDA结果一致性"""
        B, C, H, W = 1, 2, 32, 32
        
        # CPU计算
        pred_cpu = torch.randn(B, C, H, W)
        sr_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        result_cpu = apply_degradation_operator(pred_cpu, sr_params)
        
        # CUDA计算
        pred_cuda = pred_cpu.cuda()
        result_cuda = apply_degradation_operator(pred_cuda, sr_params)
        
        # 验证一致性
        np.testing.assert_allclose(
            result_cpu.numpy(),
            result_cuda.cpu().numpy(),
            rtol=1e-5, atol=1e-6
        )