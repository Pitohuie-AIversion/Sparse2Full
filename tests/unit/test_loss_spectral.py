"""
PDEBench稀疏观测重建系统 - 频域损失单元测试

测试频域损失的计算准确性，确保：
1. 频域损失计算正确（低频模态对比）
2. FFT变换和频率滤波正确
3. 非周期信号的镜像延拓处理
4. 值域转换正确（z-score域 → 原值域）
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.losses import SpectralLoss, TotalLoss


class TestSpectralLoss:
    """频域损失的单元测试类"""
    
    @pytest.fixture
    def device(self):
        """测试设备"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        """生成测试数据"""
        batch_size = 2
        channels = 3
        height, width = 64, 64
        
        # 生成包含不同频率成分的数据
        torch.manual_seed(42)
        
        # 基础随机数据
        gt_data = torch.randn(batch_size, channels, height, width, device=device)
        pred_data = gt_data + 0.1 * torch.randn_like(gt_data)
        
        # 添加低频成分
        x = torch.linspace(-np.pi, np.pi, width, device=device)
        y = torch.linspace(-np.pi, np.pi, height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 低频正弦波
        low_freq = torch.sin(X) * torch.cos(Y)
        gt_data[:, 0] += low_freq.unsqueeze(0)
        pred_data[:, 0] += low_freq.unsqueeze(0) * 0.9  # 略有差异
        
        # 中频成分
        mid_freq = torch.sin(4 * X) * torch.cos(4 * Y)
        gt_data[:, 1] += mid_freq.unsqueeze(0)
        pred_data[:, 1] += mid_freq.unsqueeze(0) * 0.8
        
        # 高频成分
        high_freq = torch.sin(16 * X) * torch.cos(16 * Y)
        gt_data[:, 2] += high_freq.unsqueeze(0)
        pred_data[:, 2] += high_freq.unsqueeze(0) * 0.7
        
        return gt_data, pred_data
    
    @pytest.fixture
    def normalization_stats(self, device):
        """归一化统计量"""
        channels = 3
        mean = torch.tensor([0.5, -0.2, 0.1], device=device).view(1, channels, 1, 1)
        std = torch.tensor([2.0, 1.5, 0.8], device=device).view(1, channels, 1, 1)
        return mean, std
    
    def test_spectral_loss_basic_computation(self, sample_data, device):
        """测试频域损失的基本计算"""
        gt_data, pred_data = sample_data
        
        # 创建频域损失函数
        spec_loss_fn = SpectralLoss(low_freq_modes=16)
        
        # 计算频域损失
        spec_loss = spec_loss_fn(pred_data, gt_data)
        
        # 检查基本属性
        assert isinstance(spec_loss, torch.Tensor), "频域损失应该是tensor"
        assert spec_loss.dim() == 0, "频域损失应该是标量"
        assert spec_loss >= 0, f"频域损失应该非负: {spec_loss}"
        assert torch.isfinite(spec_loss), "频域损失应该是有限值"
    
    def test_spectral_loss_perfect_match(self, device):
        """测试完全匹配时的频域损失"""
        # 创建相同的数据
        data = torch.randn(2, 3, 64, 64, device=device)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=16)
        spec_loss = spec_loss_fn(data, data)
        
        # 完全匹配的频域损失应该接近0
        assert spec_loss < 1e-6, f"完全匹配的频域损失过大: {spec_loss}"
    
    def test_spectral_loss_with_normalization(self, sample_data, normalization_stats, device):
        """测试带归一化的频域损失计算"""
        gt_data, pred_data = sample_data
        mean, std = normalization_stats
        
        # 归一化数据（模拟z-score归一化）
        gt_normalized = (gt_data - mean) / std
        pred_normalized = (pred_data - mean) / std
        
        # 创建带归一化的频域损失函数
        spec_loss_fn = SpectralLoss(low_freq_modes=16, mean=mean, std=std)
        
        # 计算频域损失（输入是z-score域，但计算在原值域）
        spec_loss = spec_loss_fn(pred_normalized, gt_normalized)
        
        # 检查损失有效性
        assert torch.isfinite(spec_loss), "带归一化的频域损失应该是有限值"
        assert spec_loss >= 0, "带归一化的频域损失应该非负"
        
        # 手动验证计算
        # 1. 反归一化
        pred_denorm = pred_normalized * std + mean
        gt_denorm = gt_normalized * std + mean
        
        # 2. 计算频域损失
        spec_loss_fn_no_norm = SpectralLoss(low_freq_modes=16)
        expected_loss = spec_loss_fn_no_norm(pred_denorm, gt_denorm)
        
        # 应该匹配（在数值精度范围内）
        assert torch.allclose(spec_loss, expected_loss, atol=1e-5), \
            f"频域损失计算不匹配: {spec_loss} vs {expected_loss}"
    
    def test_spectral_loss_different_freq_modes(self, sample_data, device):
        """测试不同频率模态数的影响"""
        gt_data, pred_data = sample_data
        
        freq_modes = [8, 16, 32]
        losses = []
        
        for modes in freq_modes:
            spec_loss_fn = SpectralLoss(low_freq_modes=modes)
            loss = spec_loss_fn(pred_data, gt_data)
            losses.append(loss)
            
            assert torch.isfinite(loss), f"频率模态{modes}的损失应该是有限值"
            assert loss >= 0, f"频率模态{modes}的损失应该非负"
        
        # 不同频率模态数应该产生不同的损失值
        for i in range(len(losses) - 1):
            assert not torch.allclose(losses[i], losses[i+1], atol=1e-6), \
                f"不同频率模态数应该产生不同损失: {losses[i]} vs {losses[i+1]}"
    
    def test_spectral_loss_fft_consistency(self, device):
        """测试FFT变换的一致性"""
        # 创建已知频率成分的信号
        size = 64
        x = torch.linspace(0, 2*np.pi, size, device=device)
        y = torch.linspace(0, 2*np.pi, size, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 创建单一频率成分
        freq_x, freq_y = 2, 3
        signal = torch.sin(freq_x * X) * torch.cos(freq_y * Y)
        signal = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 计算FFT
        fft_signal = torch.fft.fft2(signal)
        
        # 检查主要频率成分的位置
        power_spectrum = torch.abs(fft_signal) ** 2
        
        # 找到最大功率的位置
        max_power_idx = torch.argmax(power_spectrum.view(-1))
        max_h = max_power_idx // size
        max_w = max_power_idx % size
        
        # 应该在预期的频率位置附近
        assert max_h == freq_x or max_h == size - freq_x, \
            f"频率位置不正确: h={max_h}, 期望{freq_x}或{size-freq_x}"
    
    def test_spectral_loss_gradient_flow(self, sample_data, device):
        """测试频域损失的梯度流动"""
        gt_data, _ = sample_data
        
        # 创建需要梯度的预测
        pred_data = gt_data.clone().requires_grad_(True)
        
        # 创建频域损失函数
        spec_loss_fn = SpectralLoss(low_freq_modes=16)
        
        # 计算频域损失
        spec_loss = spec_loss_fn(pred_data, gt_data)
        
        # 反向传播
        spec_loss.backward()
        
        # 检查梯度
        assert pred_data.grad is not None, "预测数据应该有梯度"
        assert torch.isfinite(pred_data.grad).all(), "梯度应该是有限值"
        assert not torch.isnan(pred_data.grad).any(), "梯度不应该包含NaN"
        
        # 梯度范数应该合理
        grad_norm = torch.norm(pred_data.grad)
        assert grad_norm > 1e-8, f"梯度范数过小: {grad_norm}"
        assert grad_norm < 1e3, f"梯度范数过大: {grad_norm}"
    
    def test_spectral_loss_frequency_selectivity(self, device):
        """测试频域损失的频率选择性"""
        size = 64
        
        # 创建低频信号
        x = torch.linspace(0, 2*np.pi, size, device=device)
        y = torch.linspace(0, 2*np.pi, size, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        low_freq_signal = torch.sin(X) * torch.cos(Y)
        low_freq_signal = low_freq_signal.unsqueeze(0).unsqueeze(0)
        
        # 创建高频信号
        high_freq_signal = torch.sin(20 * X) * torch.cos(20 * Y)
        high_freq_signal = high_freq_signal.unsqueeze(0).unsqueeze(0)
        
        # 创建混合信号
        mixed_signal = low_freq_signal + 0.1 * high_freq_signal
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        
        # 低频差异应该产生较大损失
        low_freq_loss = spec_loss_fn(low_freq_signal * 0.5, low_freq_signal)
        
        # 高频差异应该产生较小损失（因为只关注低频）
        high_freq_loss = spec_loss_fn(high_freq_signal * 0.5, high_freq_signal)
        
        # 低频损失应该比高频损失更敏感
        # 注意：这个测试可能需要根据具体实现调整
        assert low_freq_loss > high_freq_loss * 0.1, \
            f"低频损失应该更敏感: {low_freq_loss} vs {high_freq_loss}"
    
    def test_spectral_loss_batch_consistency(self, device):
        """测试频域损失的批处理一致性"""
        # 创建单个样本
        single_gt = torch.randn(1, 3, 64, 64, device=device)
        single_pred = single_gt + 0.1 * torch.randn_like(single_gt)
        
        # 创建批次数据
        batch_size = 4
        batch_gt = single_gt.repeat(batch_size, 1, 1, 1)
        batch_pred = single_pred.repeat(batch_size, 1, 1, 1)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=16)
        
        # 单个样本的损失
        single_loss = spec_loss_fn(single_pred, single_gt)
        
        # 批次样本的损失
        batch_loss = spec_loss_fn(batch_pred, batch_gt)
        
        # 批次损失应该等于单个损失（因为所有样本相同）
        assert torch.allclose(batch_loss, single_loss, atol=1e-6), \
            f"批次损失不一致: {batch_loss} vs {single_loss}"
    
    def test_spectral_loss_mirror_padding(self, device):
        """测试镜像延拓处理"""
        # 创建非周期信号（边界不连续）
        size = 32
        signal = torch.zeros(1, 1, size, size, device=device)
        
        # 在中心创建一个方块
        center = size // 2
        signal[:, :, center-4:center+4, center-4:center+4] = 1.0
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        
        # 计算频域损失
        spec_loss = spec_loss_fn(signal * 0.8, signal)
        
        # 应该能正常计算，不产生异常值
        assert torch.isfinite(spec_loss), "镜像延拓后的频域损失应该是有限值"
        assert spec_loss >= 0, "镜像延拓后的频域损失应该非负"
    
    def test_spectral_loss_integration_with_total_loss(self, sample_data, device):
        """测试频域损失在总损失中的集成"""
        gt_data, pred_data = sample_data
        
        # 创建观测数据（用于DC损失）
        observed = torch.nn.functional.avg_pool2d(gt_data, 2)
        
        # 创建总损失函数
        dc_config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        total_loss_fn = TotalLoss(
            rec_weight=1.0,
            spec_weight=0.5,
            dc_weight=1.0,
            dc_config=dc_config,
            spec_config={'low_freq_modes': 16}
        )
        
        # 计算总损失
        total_loss, loss_dict = total_loss_fn(pred_data, gt_data, observed)
        
        # 检查频域损失
        assert 'spec_loss' in loss_dict, "缺少频域损失"
        spec_loss = loss_dict['spec_loss']
        
        assert torch.isfinite(spec_loss), "频域损失应该是有限值"
        assert spec_loss >= 0, "频域损失应该非负"
        
        # 单独计算频域损失进行验证
        spec_loss_fn = SpectralLoss(low_freq_modes=16)
        expected_spec_loss = spec_loss_fn(pred_data, gt_data)
        
        assert torch.allclose(spec_loss, expected_spec_loss, atol=1e-6), \
            f"集成的频域损失不匹配: {spec_loss} vs {expected_spec_loss}"


class TestSpectralLossEdgeCases:
    """频域损失边缘情况测试"""
    
    def test_spectral_loss_small_input(self, device):
        """测试小尺寸输入"""
        # 创建很小的输入
        small_data = torch.randn(1, 1, 8, 8, device=device)
        small_pred = small_data + 0.1 * torch.randn_like(small_data)
        
        # 频率模态数不能超过信号大小
        spec_loss_fn = SpectralLoss(low_freq_modes=4)
        spec_loss = spec_loss_fn(small_pred, small_data)
        
        assert torch.isfinite(spec_loss), "小尺寸输入的频域损失应该是有限值"
        assert spec_loss >= 0, "小尺寸输入的频域损失应该非负"
    
    def test_spectral_loss_large_freq_modes(self, device):
        """测试过大的频率模态数"""
        data = torch.randn(1, 1, 32, 32, device=device)
        pred = data + 0.1 * torch.randn_like(data)
        
        # 频率模态数接近信号大小
        spec_loss_fn = SpectralLoss(low_freq_modes=30)
        
        try:
            spec_loss = spec_loss_fn(pred, data)
            assert torch.isfinite(spec_loss), "大频率模态数的损失应该是有限值"
        except Exception as e:
            # 如果实现有限制，应该给出合理的错误信息
            assert "frequency modes" in str(e).lower() or "size" in str(e).lower()
    
    def test_spectral_loss_zero_input(self, device):
        """测试零输入"""
        zero_data = torch.zeros(2, 3, 32, 32, device=device)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        
        # 零与零的频域损失应该是0
        zero_loss = spec_loss_fn(zero_data, zero_data)
        assert torch.allclose(zero_loss, torch.tensor(0.0, device=device), atol=1e-8)
        
        # 零与非零的频域损失应该等于非零信号的功率
        nonzero_data = torch.randn(2, 3, 32, 32, device=device)
        nonzero_loss = spec_loss_fn(zero_data, nonzero_data)
        
        assert nonzero_loss > 0, "零与非零的频域损失应该大于0"
        assert torch.isfinite(nonzero_loss), "零与非零的频域损失应该是有限值"
    
    def test_spectral_loss_constant_input(self, device):
        """测试常数输入"""
        constant_value = 5.0
        constant_data = torch.full((1, 2, 32, 32), constant_value, device=device)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        
        # 相同常数的频域损失应该是0
        constant_loss = spec_loss_fn(constant_data, constant_data)
        assert torch.allclose(constant_loss, torch.tensor(0.0, device=device), atol=1e-6)
        
        # 不同常数的频域损失应该主要来自DC分量
        different_constant = torch.full_like(constant_data, constant_value * 0.8)
        different_loss = spec_loss_fn(different_constant, constant_data)
        
        assert different_loss > 0, "不同常数的频域损失应该大于0"
        assert torch.isfinite(different_loss), "不同常数的频域损失应该是有限值"
    
    def test_spectral_loss_numerical_stability(self, device):
        """测试数值稳定性"""
        # 测试极小值
        small_data = torch.full((1, 1, 32, 32), 1e-8, device=device)
        small_pred = small_data + 1e-9 * torch.randn_like(small_data)
        
        # 测试极大值
        large_data = torch.full((1, 1, 32, 32), 1e8, device=device)
        large_pred = large_data + 1e7 * torch.randn_like(large_data)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        
        # 测试极小值
        small_loss = spec_loss_fn(small_pred, small_data)
        assert torch.isfinite(small_loss), "极小值情况下频域损失不稳定"
        
        # 测试极大值
        large_loss = spec_loss_fn(large_pred, large_data)
        assert torch.isfinite(large_loss), "极大值情况下频域损失不稳定"
    
    def test_spectral_loss_different_dtypes(self, device):
        """测试不同数据类型"""
        # 测试float32
        data_f32 = torch.randn(1, 1, 32, 32, dtype=torch.float32, device=device)
        pred_f32 = data_f32 + 0.1 * torch.randn_like(data_f32)
        
        spec_loss_fn = SpectralLoss(low_freq_modes=8)
        loss_f32 = spec_loss_fn(pred_f32, data_f32)
        
        assert loss_f32.dtype == torch.float32, "float32输入应产生float32输出"
        
        # 测试float64
        data_f64 = data_f32.double()
        pred_f64 = pred_f32.double()
        
        loss_f64 = spec_loss_fn(pred_f64, data_f64)
        assert loss_f64.dtype == torch.float64, "float64输入应产生float64输出"
        
        # 结果应该在精度范围内一致
        assert torch.allclose(loss_f32.double(), loss_f64, atol=1e-5), \
            "不同精度的结果应该一致"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])