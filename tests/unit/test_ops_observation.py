"""
PDEBench稀疏观测重建系统 - 观测算子H单元测试

测试观测算子的一致性和正确性，确保：
1. H算子与DC损失使用完全相同的实现
2. 观测生成的结果符合预期
3. 不同参数配置下的行为正确
4. 边界条件处理正确
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

from ops.degradation import apply_degradation_operator
from utils.losses import TotalLoss


class TestObservationOperators:
    """观测算子H的单元测试类"""
    
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
        
        # 生成随机数据
        torch.manual_seed(42)
        data = torch.randn(batch_size, channels, height, width, device=device)
        
        # 添加一些结构化模式
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 添加正弦波模式
        pattern = torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
        data[:, 0] += pattern.unsqueeze(0)
        
        # 添加高斯模式
        gaussian = torch.exp(-(X**2 + Y**2) / 0.5)
        data[:, 1] += gaussian.unsqueeze(0)
        
        return data
    
    @pytest.fixture
    def sr_configs(self):
        """超分辨率配置"""
        return [
            {
                'task': 'super_resolution',
                'scale_factor': 2,
                'kernel_size': 5,
                'sigma': 1.0,
                'interpolation': 'bilinear'
            },
            {
                'task': 'super_resolution',
                'scale_factor': 4,
                'kernel_size': 7,
                'sigma': 1.5,
                'interpolation': 'bicubic'
            }
        ]
    
    @pytest.fixture
    def crop_configs(self):
        """裁剪重建配置"""
        return [
            {
                'task': 'crop_reconstruction',
                'crop_ratio': 0.5,
                'crop_mode': 'center',
                'boundary_mode': 'reflect'
            },
            {
                'task': 'crop_reconstruction',
                'crop_ratio': 0.3,
                'crop_mode': 'random',
                'boundary_mode': 'zero'
            }
        ]
    
    def test_sr_operator_consistency(self, sample_data, sr_configs, device):
        """测试超分辨率算子的一致性"""
        for config in sr_configs:
            # 应用观测算子
            observed = apply_degradation_operator(sample_data, config)
            
            # 检查输出形状
            scale_factor = config['scale_factor']
            expected_h = sample_data.shape[2] // scale_factor
            expected_w = sample_data.shape[3] // scale_factor
            
            assert observed.shape[0] == sample_data.shape[0], "批次大小不匹配"
            assert observed.shape[1] == sample_data.shape[1], "通道数不匹配"
            assert observed.shape[2] == expected_h, f"高度不匹配: {observed.shape[2]} vs {expected_h}"
            assert observed.shape[3] == expected_w, f"宽度不匹配: {observed.shape[3]} vs {expected_w}"
            
            # 检查数值范围合理性
            assert torch.isfinite(observed).all(), "输出包含非有限值"
            assert not torch.isnan(observed).any(), "输出包含NaN"
            
            # 检查能量守恒（近似）
            original_energy = torch.sum(sample_data**2)
            observed_energy = torch.sum(observed**2) * (scale_factor**2)
            energy_ratio = observed_energy / original_energy
            
            # 能量比应该在合理范围内（考虑模糊和下采样的影响）
            assert 0.1 < energy_ratio < 2.0, f"能量比异常: {energy_ratio}"
    
    def test_crop_operator_consistency(self, sample_data, crop_configs, device):
        """测试裁剪算子的一致性"""
        for config in crop_configs:
            # 应用观测算子
            observed = apply_degradation_operator(sample_data, config)
            
            # 检查输出形状
            crop_ratio = config['crop_ratio']
            expected_h = int(sample_data.shape[2] * crop_ratio)
            expected_w = int(sample_data.shape[3] * crop_ratio)
            
            assert observed.shape[0] == sample_data.shape[0], "批次大小不匹配"
            assert observed.shape[1] == sample_data.shape[1], "通道数不匹配"
            assert observed.shape[2] == expected_h, f"高度不匹配: {observed.shape[2]} vs {expected_h}"
            assert observed.shape[3] == expected_w, f"宽度不匹配: {observed.shape[3]} vs {expected_w}"
            
            # 检查数值范围合理性
            assert torch.isfinite(observed).all(), "输出包含非有限值"
            assert not torch.isnan(observed).any(), "输出包含NaN"
            
            # 对于中心裁剪，检查是否确实是中心区域
            if config['crop_mode'] == 'center':
                h_start = (sample_data.shape[2] - expected_h) // 2
                w_start = (sample_data.shape[3] - expected_w) // 2
                expected_crop = sample_data[:, :, h_start:h_start+expected_h, w_start:w_start+expected_w]
                
                # 应该完全匹配（在数值精度范围内）
                assert torch.allclose(observed, expected_crop, atol=1e-6), "中心裁剪结果不匹配"
    
    def test_operator_determinism(self, sample_data, sr_configs, device):
        """测试算子的确定性（相同输入应产生相同输出）"""
        config = sr_configs[0]
        
        # 多次应用相同配置
        results = []
        for _ in range(3):
            result = apply_degradation_operator(sample_data, config)
            results.append(result)
        
        # 所有结果应该完全相同
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-8), f"第{i}次结果与第0次不一致"
    
    def test_operator_with_dc_loss_consistency(self, sample_data, sr_configs, device):
        """测试观测算子与DC损失的一致性"""
        config = sr_configs[0]
        
        # 创建损失函数
        loss_fn = TotalLoss(
            rec_weight=1.0,
            spec_weight=0.5,
            dc_weight=1.0,
            dc_config=config
        )
        
        # 应用观测算子
        observed = apply_degradation_operator(sample_data, config)
        
        # 创建预测（这里使用原始数据作为"预测"）
        prediction = sample_data.clone()
        
        # 计算DC损失
        total_loss, loss_dict = loss_fn(prediction, sample_data, observed)
        dc_loss = loss_dict['dc_loss']
        
        # DC损失应该接近0（因为H(GT) = observed）
        assert dc_loss < 1e-6, f"DC损失过大: {dc_loss}, 表明H算子不一致"
    
    def test_boundary_conditions(self, device):
        """测试边界条件处理"""
        # 创建边界有特殊值的测试数据
        data = torch.zeros(1, 1, 32, 32, device=device)
        data[:, :, 0, :] = 1.0  # 上边界
        data[:, :, -1, :] = 2.0  # 下边界
        data[:, :, :, 0] = 3.0  # 左边界
        data[:, :, :, -1] = 4.0  # 右边界
        
        # 测试不同边界模式
        boundary_modes = ['reflect', 'zero', 'wrap']
        
        for boundary_mode in boundary_modes:
            config = {
                'task': 'crop_reconstruction',
                'crop_ratio': 0.8,
                'crop_mode': 'center',
                'boundary_mode': boundary_mode
            }
            
            try:
                result = apply_degradation_operator(data, config)
                
                # 检查结果有效性
                assert torch.isfinite(result).all(), f"边界模式 {boundary_mode} 产生非有限值"
                assert not torch.isnan(result).any(), f"边界模式 {boundary_mode} 产生NaN"
                
            except Exception as e:
                pytest.fail(f"边界模式 {boundary_mode} 处理失败: {e}")
    
    def test_extreme_parameters(self, sample_data, device):
        """测试极端参数情况"""
        # 测试极大缩放因子
        config_large_scale = {
            'task': 'super_resolution',
            'scale_factor': 8,
            'kernel_size': 15,
            'sigma': 3.0,
            'interpolation': 'bilinear'
        }
        
        try:
            result = apply_degradation_operator(sample_data, config_large_scale)
            assert result.shape[2] == sample_data.shape[2] // 8
            assert result.shape[3] == sample_data.shape[3] // 8
        except Exception as e:
            pytest.fail(f"极大缩放因子处理失败: {e}")
        
        # 测试极小裁剪比例
        config_small_crop = {
            'task': 'crop_reconstruction',
            'crop_ratio': 0.1,
            'crop_mode': 'center',
            'boundary_mode': 'reflect'
        }
        
        try:
            result = apply_degradation_operator(sample_data, config_small_crop)
            expected_size = int(sample_data.shape[2] * 0.1)
            assert result.shape[2] == expected_size
            assert result.shape[3] == expected_size
        except Exception as e:
            pytest.fail(f"极小裁剪比例处理失败: {e}")
    
    def test_batch_consistency(self, device):
        """测试批处理一致性"""
        # 创建不同大小的批次
        batch_sizes = [1, 2, 4, 8]
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        # 创建单个样本
        single_sample = torch.randn(1, 3, 64, 64, device=device)
        single_result = apply_degradation_operator(single_sample, config)
        
        for batch_size in batch_sizes:
            # 创建批次数据（重复单个样本）
            batch_data = single_sample.repeat(batch_size, 1, 1, 1)
            batch_result = apply_degradation_operator(batch_data, config)
            
            # 检查形状
            assert batch_result.shape[0] == batch_size
            assert batch_result.shape[1:] == single_result.shape[1:]
            
            # 检查每个样本的结果是否一致
            for i in range(batch_size):
                assert torch.allclose(batch_result[i], single_result[0], atol=1e-6), \
                    f"批次中第{i}个样本结果不一致"
    
    def test_gradient_flow(self, sample_data, sr_configs, device):
        """测试梯度流动"""
        config = sr_configs[0]
        
        # 设置需要梯度
        input_data = sample_data.clone().requires_grad_(True)
        
        # 应用观测算子
        observed = apply_degradation_operator(input_data, config)
        
        # 计算损失并反向传播
        loss = torch.sum(observed**2)
        loss.backward()
        
        # 检查梯度
        assert input_data.grad is not None, "输入数据没有梯度"
        assert torch.isfinite(input_data.grad).all(), "梯度包含非有限值"
        assert not torch.isnan(input_data.grad).any(), "梯度包含NaN"
        
        # 梯度应该不全为零（除非输入对输出没有影响）
        grad_norm = torch.norm(input_data.grad)
        assert grad_norm > 1e-8, f"梯度范数过小: {grad_norm}"


class TestObservationOperatorEdgeCases:
    """观测算子边缘情况测试"""
    
    def test_single_pixel_input(self, device):
        """测试单像素输入"""
        data = torch.randn(1, 1, 1, 1, device=device)
        
        config = {
            'task': 'crop_reconstruction',
            'crop_ratio': 1.0,
            'crop_mode': 'center',
            'boundary_mode': 'reflect'
        }
        
        result = apply_degradation_operator(data, config)
        assert result.shape == data.shape
        assert torch.allclose(result, data)
    
    def test_large_input(self, device):
        """测试大尺寸输入"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA进行大尺寸测试")
        
        # 创建较大的输入（但不要太大以免内存不足）
        data = torch.randn(1, 3, 512, 512, device=device)
        
        config = {
            'task': 'super_resolution',
            'scale_factor': 4,
            'kernel_size': 7,
            'sigma': 1.5,
            'interpolation': 'bilinear'
        }
        
        try:
            result = apply_degradation_operator(data, config)
            assert result.shape == (1, 3, 128, 128)
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("内存不足，跳过大尺寸测试")
            else:
                raise
    
    def test_zero_input(self, device):
        """测试零输入"""
        data = torch.zeros(2, 3, 32, 32, device=device)
        
        configs = [
            {
                'task': 'super_resolution',
                'scale_factor': 2,
                'kernel_size': 5,
                'sigma': 1.0,
                'interpolation': 'bilinear'
            },
            {
                'task': 'crop_reconstruction',
                'crop_ratio': 0.5,
                'crop_mode': 'center',
                'boundary_mode': 'reflect'
            }
        ]
        
        for config in configs:
            result = apply_degradation_operator(data, config)
            
            # 零输入应该产生零输出（或接近零）
            assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), \
                f"零输入没有产生零输出，任务: {config['task']}"
    
    def test_constant_input(self, device):
        """测试常数输入"""
        constant_value = 5.0
        data = torch.full((1, 2, 64, 64), constant_value, device=device)
        
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        result = apply_degradation_operator(data, config)
        
        # 常数输入经过线性操作应该仍然是常数（或接近常数）
        result_std = torch.std(result)
        assert result_std < 0.1, f"常数输入产生了过大的变化: std={result_std}"
        
        # 平均值应该接近原始常数值
        result_mean = torch.mean(result)
        assert abs(result_mean - constant_value) < 0.5, \
            f"常数输入的平均值变化过大: {result_mean} vs {constant_value}"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])