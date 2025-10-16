"""
测试losses模块：损失函数数值正确性测试

测试重建损失、频谱损失、数据一致性损失的数值正确性。
严格按照开发手册要求，确保损失函数满足"黄金法则"。
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from omegaconf import DictConfig, OmegaConf

from ops.losses import (
    compute_total_loss,
    _compute_reconstruction_loss,
    _compute_spectral_loss,
    _compute_data_consistency_loss,
    _compute_gradient_loss,
    _compute_relative_l2_loss,
    _denormalize_tensor,
    _mirror_extend,
    compute_loss_weights_schedule
)


class TestTotalLoss:
    """测试总损失计算"""
    
    def test_total_loss_basic(self, device, sample_loss_config):
        """测试基本总损失计算"""
        B, C, H, W = 2, 3, 32, 32
        pred_z = torch.randn(B, C, H, W, device=device)
        target_z = torch.randn(B, C, H, W, device=device)
        
        # 创建观测数据
        obs_data = {
            'baseline': torch.randn(B, C, H, W, device=device),
            'coords': torch.randn(B, 2, H, W, device=device),
            'mask': torch.ones(B, 1, H, W, device=device),
            'h_params': {
                'task': 'SR',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
        }
        
        # 创建归一化统计量
        norm_stats = {
            'u_mean': torch.tensor([0.0], device=device),
            'u_std': torch.tensor([1.0], device=device),
            'v_mean': torch.tensor([0.0], device=device),
            'v_std': torch.tensor([1.0], device=device),
            'p_mean': torch.tensor([0.0], device=device),
            'p_std': torch.tensor([1.0], device=device)
        }
        
        # 计算总损失
        losses = compute_total_loss(pred_z, target_z, obs_data, norm_stats, sample_loss_config)
        
        # 验证返回的损失项
        expected_keys = ['reconstruction_loss', 'spectral_loss', 'dc_loss', 'total_loss']
        for key in expected_keys:
            assert key in losses
            assert torch.isfinite(losses[key])
            assert losses[key] >= 0  # 损失应该非负
        
        # 验证总损失是各分量的加权和
        w_rec = sample_loss_config.train.loss_weights.reconstruction
        w_spec = sample_loss_config.train.loss_weights.spectral
        w_dc = sample_loss_config.train.loss_weights.data_consistency
        
        expected_total = (
            w_rec * losses['reconstruction_loss'] +
            w_spec * losses['spectral_loss'] +
            w_dc * losses['dc_loss']
        )
        
        np.testing.assert_allclose(
            losses['total_loss'].item(),
            expected_total.item(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_total_loss_zero_weights(self, device, sample_loss_config):
        """测试零权重情况"""
        B, C, H, W = 1, 1, 16, 16
        pred_z = torch.randn(B, C, H, W, device=device)
        target_z = torch.randn(B, C, H, W, device=device)
        
        obs_data = {
            'baseline': torch.randn(B, C, H, W, device=device),
            'h_params': {
                'task': 'SR',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
        }
        
        # 设置零权重
        config = sample_loss_config.copy()
        config.train.loss_weights.spectral = 0.0
        config.train.loss_weights.data_consistency = 0.0
        
        losses = compute_total_loss(pred_z, target_z, obs_data, None, config)
        
        # 零权重的损失应该为0
        assert losses['spectral_loss'].item() == 0.0
        assert losses['dc_loss'].item() == 0.0
        
        # 总损失应该只包含重建损失
        expected_total = config.train.loss_weights.reconstruction * losses['reconstruction_loss']
        np.testing.assert_allclose(
            losses['total_loss'].item(),
            expected_total.item(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_total_loss_gradient_flow(self, device, sample_loss_config):
        """测试梯度流动"""
        B, C, H, W = 1, 2, 16, 16
        pred_z = torch.randn(B, C, H, W, device=device, requires_grad=True)
        target_z = torch.randn(B, C, H, W, device=device)
        
        obs_data = {
            'baseline': torch.randn(B, C, H, W, device=device),
            'h_params': {
                'task': 'SR',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
        }
        
        losses = compute_total_loss(pred_z, target_z, obs_data, None, sample_loss_config)
        
        # 反向传播
        losses['total_loss'].backward()
        
        # 验证梯度存在且有限
        assert pred_z.grad is not None
        assert torch.isfinite(pred_z.grad).all()
        assert pred_z.grad.abs().sum() > 0


class TestReconstructionLoss:
    """测试重建损失"""
    
    def test_reconstruction_loss_basic(self, device):
        """测试基本重建损失"""
        B, C, H, W = 2, 2, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        obs_data = {}  # 重建损失不需要观测数据
        
        loss = _compute_reconstruction_loss(pred, target, obs_data)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_reconstruction_loss_identical_inputs(self, device):
        """测试相同输入的重建损失"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = pred.clone()
        
        obs_data = {}
        
        loss = _compute_reconstruction_loss(pred, target, obs_data)
        
        # 相同输入的损失应该接近0
        assert loss < 1e-6
    
    def test_reconstruction_loss_consistency(self, device, tolerance_config):
        """测试重建损失一致性"""
        B, C, H, W = 1, 2, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        obs_data = {}
        
        # 多次计算
        loss1 = _compute_reconstruction_loss(pred, target, obs_data)
        loss2 = _compute_reconstruction_loss(pred, target, obs_data)
        
        # 应该完全一致
        np.testing.assert_allclose(
            loss1.item(), loss2.item(),
            rtol=tolerance_config['rtol'],
            atol=tolerance_config['atol']
        )


class TestSpectralLoss:
    """测试频谱损失"""
    
    def test_spectral_loss_basic(self, device, sample_loss_config):
        """测试基本频谱损失"""
        B, C, H, W = 2, 2, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        loss = _compute_spectral_loss(pred, target, sample_loss_config)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_spectral_loss_identical_inputs(self, device, sample_loss_config):
        """测试相同输入的频谱损失"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        target = pred.clone()
        
        loss = _compute_spectral_loss(pred, target, sample_loss_config)
        
        # 相同输入的损失应该接近0
        assert loss < 1e-6
    
    def test_spectral_loss_low_freq_focus(self, device, sample_loss_config):
        """测试频谱损失只关注低频"""
        B, C, H, W = 1, 1, 64, 64
        
        # 创建只有低频成分的信号
        low_freq_signal = torch.zeros(B, C, H, W, device=device)
        low_freq_signal[:, :, :8, :8] = torch.randn(B, C, 8, 8, device=device)
        
        # 创建有高频噪声的信号
        high_freq_noise = torch.randn(B, C, H, W, device=device) * 0.1
        noisy_signal = low_freq_signal + high_freq_noise
        
        # 频谱损失应该主要关注低频差异
        loss_clean = _compute_spectral_loss(low_freq_signal, low_freq_signal, sample_loss_config)
        loss_noisy = _compute_spectral_loss(noisy_signal, low_freq_signal, sample_loss_config)
        
        # 高频噪声对频谱损失的影响应该较小
        assert loss_clean < 1e-6
        assert loss_noisy < loss_clean + 0.1  # 噪声影响有限
    
    def test_spectral_loss_different_modes(self, device, sample_loss_config):
        """测试不同低频模式数"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        # 测试不同的低频模式数
        for low_freq_modes in [8, 16, 24]:
            config = sample_loss_config.copy()
            config.train.spectral_loss.low_freq_modes = low_freq_modes
            
            loss = _compute_spectral_loss(pred, target, config)
            
            assert torch.isfinite(loss)
            assert loss >= 0


class TestDataConsistencyLoss:
    """测试数据一致性损失"""
    
    def test_dc_loss_sr_mode(self, device):
        """测试SR模式的数据一致性损失"""
        B, C, H, W = 1, 2, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        # 创建SR观测数据
        obs_data = {
            'baseline': torch.randn(B, C, H, W, device=device),
            'lr_observation': torch.randn(B, C, H//2, W//2, device=device),
            'h_params': {
                'task': 'SR',
                'scale': 2,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
        }
        
        loss = _compute_data_consistency_loss(pred, obs_data)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_dc_loss_crop_mode(self, device):
        """测试Crop模式的数据一致性损失"""
        B, C, H, W = 1, 2, 64, 64
        pred = torch.randn(B, C, H, W, device=device)
        
        # 创建Crop观测数据
        crop_size = (32, 32)
        crop_box = (16, 16, 48, 48)
        
        obs_data = {
            'baseline': torch.randn(B, C, H, W, device=device),
            'crop_box': crop_box,
            'cropped_observation': torch.randn(B, C, crop_size[0], crop_size[1], device=device),
            'h_params': {
                'task': 'Crop',
                'crop_size': crop_size,
                'crop_box': crop_box,
                'boundary': 'mirror'
            }
        }
        
        loss = _compute_data_consistency_loss(pred, obs_data)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_dc_loss_perfect_consistency(self, device):
        """测试完美一致性的DC损失"""
        B, C, H, W = 1, 1, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        
        # 从pred生成完美一致的观测
        from ops.degradation import apply_degradation_operator
        
        h_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror'
        }
        
        perfect_obs = apply_degradation_operator(pred, h_params)
        
        obs_data = {
            'lr_observation': perfect_obs,
            'h_params': h_params
        }
        
        loss = _compute_data_consistency_loss(pred, obs_data)
        
        # 完美一致性的损失应该接近0
        assert loss < 1e-6
    
    def test_dc_loss_invalid_task(self, device):
        """测试无效任务类型"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        
        obs_data = {
            'h_params': {
                'task': 'InvalidTask'
            }
        }
        
        with pytest.raises(ValueError, match="Unknown task"):
            _compute_data_consistency_loss(pred, obs_data)


class TestGradientLoss:
    """测试梯度损失"""
    
    def test_gradient_loss_basic(self, device):
        """测试基本梯度损失"""
        B, C, H, W = 2, 2, 32, 32
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        loss = _compute_gradient_loss(pred, target)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_gradient_loss_identical_inputs(self, device):
        """测试相同输入的梯度损失"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = pred.clone()
        
        loss = _compute_gradient_loss(pred, target)
        
        # 相同输入的损失应该为0
        assert loss < 1e-6
    
    def test_gradient_loss_smooth_vs_noisy(self, device):
        """测试平滑信号vs噪声信号的梯度损失"""
        B, C, H, W = 1, 1, 32, 32
        
        # 创建平滑信号
        smooth_signal = torch.sin(torch.linspace(0, 2*np.pi, W, device=device)).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        
        # 创建噪声信号
        noisy_signal = smooth_signal + torch.randn(B, C, H, W, device=device) * 0.1
        
        # 梯度损失应该能区分平滑和噪声
        loss_smooth = _compute_gradient_loss(smooth_signal, smooth_signal)
        loss_noisy = _compute_gradient_loss(noisy_signal, smooth_signal)
        
        assert loss_smooth < 1e-6
        assert loss_noisy > loss_smooth


class TestRelativeL2Loss:
    """测试相对L2损失"""
    
    def test_relative_l2_basic(self, device):
        """测试基本相对L2损失"""
        B, C, H, W = 2, 2, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        loss = _compute_relative_l2_loss(pred, target)
        
        # 验证损失为标量且非负
        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)
    
    def test_relative_l2_identical_inputs(self, device):
        """测试相同输入的相对L2损失"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = pred.clone()
        
        loss = _compute_relative_l2_loss(pred, target)
        
        # 相同输入的损失应该为0
        assert loss < 1e-6
    
    def test_relative_l2_scale_invariance(self, device):
        """测试相对L2损失的尺度不变性"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        # 计算原始损失
        loss_orig = _compute_relative_l2_loss(pred, target)
        
        # 缩放输入
        scale = 10.0
        loss_scaled = _compute_relative_l2_loss(pred * scale, target * scale)
        
        # 相对L2损失应该对缩放不变
        np.testing.assert_allclose(
            loss_orig.item(), loss_scaled.item(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_relative_l2_zero_target(self, device):
        """测试零目标的相对L2损失"""
        B, C, H, W = 1, 1, 8, 8
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.zeros(B, C, H, W, device=device)
        
        loss = _compute_relative_l2_loss(pred, target, eps=1e-8)
        
        # 应该能处理零目标情况
        assert torch.isfinite(loss)
        assert loss >= 0


class TestDenormalization:
    """测试反归一化功能"""
    
    def test_denormalize_tensor_basic(self, device):
        """测试基本反归一化"""
        B, C, H, W = 2, 3, 16, 16
        tensor_z = torch.randn(B, C, H, W, device=device)
        
        # 创建归一化统计量
        keys = ['u', 'v', 'p']
        norm_stats = {}
        means = [1.0, 2.0, 3.0]
        stds = [0.5, 1.0, 1.5]
        
        for i, key in enumerate(keys):
            norm_stats[f"{key}_mean"] = torch.tensor([means[i]], device=device)
            norm_stats[f"{key}_std"] = torch.tensor([stds[i]], device=device)
        
        # 反归一化
        tensor_orig = _denormalize_tensor(tensor_z, norm_stats, keys)
        
        # 验证形状不变
        assert tensor_orig.shape == tensor_z.shape
        
        # 验证反归一化公式：x_orig = x_z * std + mean
        for i, key in enumerate(keys):
            expected = tensor_z[:, i:i+1] * stds[i] + means[i]
            np.testing.assert_allclose(
                tensor_orig[:, i:i+1].cpu().numpy(),
                expected.cpu().numpy(),
                rtol=1e-5, atol=1e-8
            )
    
    def test_denormalize_tensor_none_stats(self, device):
        """测试无归一化统计量的情况"""
        B, C, H, W = 1, 2, 8, 8
        tensor_z = torch.randn(B, C, H, W, device=device)
        
        tensor_orig = _denormalize_tensor(tensor_z, None, ['u', 'v'])
        
        # 应该返回原张量
        np.testing.assert_allclose(
            tensor_orig.cpu().numpy(),
            tensor_z.cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_denormalize_tensor_roundtrip(self, device):
        """测试归一化-反归一化往返"""
        B, C, H, W = 1, 2, 16, 16
        tensor_orig = torch.randn(B, C, H, W, device=device)
        
        # 模拟归一化统计量
        keys = ['u', 'v']
        means = [tensor_orig[:, i].mean().item() for i in range(C)]
        stds = [tensor_orig[:, i].std().item() for i in range(C)]
        
        norm_stats = {}
        for i, key in enumerate(keys):
            norm_stats[f"{key}_mean"] = torch.tensor([means[i]], device=device)
            norm_stats[f"{key}_std"] = torch.tensor([stds[i]], device=device)
        
        # 归一化
        tensor_z = tensor_orig.clone()
        for i, key in enumerate(keys):
            tensor_z[:, i:i+1] = (tensor_orig[:, i:i+1] - means[i]) / stds[i]
        
        # 反归一化
        tensor_recovered = _denormalize_tensor(tensor_z, norm_stats, keys)
        
        # 应该恢复原始张量
        np.testing.assert_allclose(
            tensor_recovered.cpu().numpy(),
            tensor_orig.cpu().numpy(),
            rtol=1e-4, atol=1e-6
        )


class TestMirrorExtend:
    """测试镜像延拓功能"""
    
    def test_mirror_extend_basic(self, device):
        """测试基本镜像延拓"""
        B, C, H, W = 1, 1, 8, 8
        x = torch.randn(B, C, H, W, device=device)
        
        extended = _mirror_extend(x, factor=2)
        
        # 验证输出尺寸
        assert extended.shape == (B, C, H*2, W*2)
        
        # 验证左上角区域保持不变
        np.testing.assert_allclose(
            extended[:, :, :H, :W].cpu().numpy(),
            x.cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
        
        # 验证水平镜像
        np.testing.assert_allclose(
            extended[:, :, :H, W:].cpu().numpy(),
            torch.flip(x, dims=[-1]).cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
        
        # 验证垂直镜像
        np.testing.assert_allclose(
            extended[:, :, H:, :W].cpu().numpy(),
            torch.flip(x, dims=[-2]).cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
    
    def test_mirror_extend_symmetry(self, device):
        """测试镜像延拓的对称性"""
        B, C, H, W = 1, 1, 4, 4
        x = torch.randn(B, C, H, W, device=device)
        
        extended = _mirror_extend(x)
        
        # 验证对称性
        # 水平对称
        left_half = extended[:, :, :, :W]
        right_half = extended[:, :, :, W:]
        np.testing.assert_allclose(
            left_half.cpu().numpy(),
            torch.flip(right_half, dims=[-1]).cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )
        
        # 垂直对称
        top_half = extended[:, :, :H, :]
        bottom_half = extended[:, :, H:, :]
        np.testing.assert_allclose(
            top_half.cpu().numpy(),
            torch.flip(bottom_half, dims=[-2]).cpu().numpy(),
            rtol=1e-5, atol=1e-8
        )


class TestLossWeightsSchedule:
    """测试损失权重调度"""
    
    def test_loss_weights_schedule_basic(self):
        """测试基本权重调度"""
        base_weights = {
            'reconstruction': 1.0,
            'spectral': 0.5,
            'data_consistency': 1.0
        }
        
        total_epochs = 100
        
        # 测试不同epoch的权重
        for epoch in [0, 25, 50, 75, 99]:
            weights = compute_loss_weights_schedule(epoch, total_epochs, base_weights)
            
            # 验证所有权重都存在且为正
            for key in base_weights:
                assert key in weights
                assert weights[key] >= 0
                assert np.isfinite(weights[key])
    
    def test_loss_weights_schedule_progression(self):
        """测试权重调度的进展"""
        base_weights = {
            'reconstruction': 1.0,
            'spectral': 0.5,
            'data_consistency': 1.0
        }
        
        total_epochs = 100
        
        # 获取不同阶段的权重
        early_weights = compute_loss_weights_schedule(0, total_epochs, base_weights)
        mid_weights = compute_loss_weights_schedule(50, total_epochs, base_weights)
        late_weights = compute_loss_weights_schedule(99, total_epochs, base_weights)
        
        # DC权重应该随训练进度增加
        assert early_weights['data_consistency'] < late_weights['data_consistency']
        
        # 频谱权重应该在中期达到峰值
        assert mid_weights['spectral'] >= early_weights['spectral']
        assert mid_weights['spectral'] >= late_weights['spectral']
    
    def test_loss_weights_schedule_edge_cases(self):
        """测试边界情况"""
        base_weights = {'reconstruction': 1.0}
        
        # 测试单个epoch
        weights = compute_loss_weights_schedule(0, 1, base_weights)
        assert 'reconstruction' in weights
        
        # 测试空权重
        empty_weights = compute_loss_weights_schedule(0, 100, {})
        assert isinstance(empty_weights, dict)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_pixel_losses(self, device, sample_loss_config):
        """测试单像素输入的损失"""
        B, C, H, W = 1, 1, 1, 1
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        # 重建损失
        rec_loss = _compute_reconstruction_loss(pred, target, {})
        assert torch.isfinite(rec_loss)
        
        # 频谱损失
        spec_loss = _compute_spectral_loss(pred, target, sample_loss_config)
        assert torch.isfinite(spec_loss)
        
        # 梯度损失
        grad_loss = _compute_gradient_loss(pred, target)
        assert torch.isfinite(grad_loss)
    
    def test_extreme_values_losses(self, device, sample_loss_config):
        """测试极值输入的损失"""
        B, C, H, W = 1, 1, 8, 8
        
        # 极大值
        pred_large = torch.ones(B, C, H, W, device=device) * 1e6
        target_large = torch.ones(B, C, H, W, device=device) * 1e6
        
        rec_loss = _compute_reconstruction_loss(pred_large, target_large, {})
        assert torch.isfinite(rec_loss)
        
        # 极小值
        pred_small = torch.ones(B, C, H, W, device=device) * 1e-6
        target_small = torch.ones(B, C, H, W, device=device) * 1e-6
        
        rec_loss = _compute_reconstruction_loss(pred_small, target_small, {})
        assert torch.isfinite(rec_loss)
    
    def test_nan_inf_handling(self, device, sample_loss_config):
        """测试NaN和Inf处理"""
        B, C, H, W = 1, 1, 8, 8
        pred = torch.randn(B, C, H, W, device=device)
        target = torch.randn(B, C, H, W, device=device)
        
        # 注入NaN
        pred_nan = pred.clone()
        pred_nan[0, 0, 0, 0] = float('nan')
        
        # 损失函数应该能检测到NaN
        rec_loss = _compute_reconstruction_loss(pred_nan, target, {})
        assert torch.isnan(rec_loss) or torch.isinf(rec_loss)


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_loss_numerical_stability(self, device, sample_loss_config):
        """测试损失函数的数值稳定性"""
        B, C, H, W = 1, 2, 16, 16
        
        # 使用不同的数值精度
        for dtype in [torch.float32, torch.float64]:
            pred = torch.randn(B, C, H, W, device=device, dtype=dtype)
            target = torch.randn(B, C, H, W, device=device, dtype=dtype)
            
            # 重建损失
            rec_loss = _compute_reconstruction_loss(pred, target, {})
            assert torch.isfinite(rec_loss)
            
            # 相对L2损失
            rel_l2 = _compute_relative_l2_loss(pred, target)
            assert torch.isfinite(rel_l2)
    
    def test_loss_gradient_stability(self, device, sample_loss_config):
        """测试损失梯度的数值稳定性"""
        B, C, H, W = 1, 1, 16, 16
        pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
        target = torch.randn(B, C, H, W, device=device)
        
        # 计算损失
        rec_loss = _compute_reconstruction_loss(pred, target, {})
        
        # 计算梯度
        rec_loss.backward()
        
        # 验证梯度稳定性
        assert torch.isfinite(pred.grad).all()
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()


class TestPerformance:
    """测试性能相关"""
    
    def test_loss_batch_processing(self, device, sample_loss_config):
        """测试批处理性能"""
        C, H, W = 3, 32, 32
        
        for B in [1, 4, 8, 16]:
            pred = torch.randn(B, C, H, W, device=device)
            target = torch.randn(B, C, H, W, device=device)
            
            # 重建损失
            rec_loss = _compute_reconstruction_loss(pred, target, {})
            assert torch.isfinite(rec_loss)
            
            # 频谱损失
            spec_loss = _compute_spectral_loss(pred, target, sample_loss_config)
            assert torch.isfinite(spec_loss)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_cuda_consistency(self, sample_loss_config):
        """测试CPU和CUDA损失一致性"""
        B, C, H, W = 1, 2, 16, 16
        
        # CPU计算
        pred_cpu = torch.randn(B, C, H, W)
        target_cpu = torch.randn(B, C, H, W)
        
        rec_loss_cpu = _compute_reconstruction_loss(pred_cpu, target_cpu, {})
        spec_loss_cpu = _compute_spectral_loss(pred_cpu, target_cpu, sample_loss_config)
        
        # CUDA计算
        pred_cuda = pred_cpu.cuda()
        target_cuda = target_cpu.cuda()
        
        rec_loss_cuda = _compute_reconstruction_loss(pred_cuda, target_cuda, {})
        spec_loss_cuda = _compute_spectral_loss(pred_cuda, target_cuda, sample_loss_config)
        
        # 验证一致性
        np.testing.assert_allclose(
            rec_loss_cpu.item(), rec_loss_cuda.cpu().item(),
            rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            spec_loss_cpu.item(), spec_loss_cuda.cpu().item(),
            rtol=1e-5, atol=1e-6
        )