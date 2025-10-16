"""
PDEBench稀疏观测重建系统 - 数据一致性损失DC单元测试

测试数据一致性损失的计算准确性，确保：
1. DC损失计算正确：||H(ŷ) - y||
2. 值域转换正确（z-score域 → 原值域）
3. 梯度计算正确
4. 与观测算子H的一致性
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

from utils.losses import TotalLoss, DataConsistencyLoss
from ops.degradation import apply_degradation_operator


class TestDataConsistencyLoss:
    """数据一致性损失DC的单元测试类"""
    
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
        gt_data = torch.randn(batch_size, channels, height, width, device=device)
        pred_data = gt_data + 0.1 * torch.randn_like(gt_data)  # 添加小噪声
        
        return gt_data, pred_data
    
    @pytest.fixture
    def normalization_stats(self, device):
        """归一化统计量"""
        channels = 3
        mean = torch.tensor([0.5, -0.2, 0.1], device=device).view(1, channels, 1, 1)
        std = torch.tensor([2.0, 1.5, 0.8], device=device).view(1, channels, 1, 1)
        return mean, std
    
    @pytest.fixture
    def dc_configs(self):
        """DC损失配置"""
        return [
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
    
    def test_dc_loss_basic_computation(self, sample_data, dc_configs, device):
        """测试DC损失的基本计算"""
        gt_data, pred_data = sample_data
        
        for config in dc_configs:
            # 生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 创建DC损失函数
            dc_loss_fn = DataConsistencyLoss(config)
            
            # 计算DC损失
            dc_loss = dc_loss_fn(pred_data, observed)
            
            # 检查基本属性
            assert isinstance(dc_loss, torch.Tensor), "DC损失应该是tensor"
            assert dc_loss.dim() == 0, "DC损失应该是标量"
            assert dc_loss >= 0, f"DC损失应该非负: {dc_loss}"
            assert torch.isfinite(dc_loss), "DC损失应该是有限值"
    
    def test_dc_loss_perfect_reconstruction(self, sample_data, dc_configs, device):
        """测试完美重建时的DC损失"""
        gt_data, _ = sample_data
        
        for config in dc_configs:
            # 生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 创建DC损失函数
            dc_loss_fn = DataConsistencyLoss(config)
            
            # 使用GT作为预测（完美重建）
            dc_loss = dc_loss_fn(gt_data, observed)
            
            # 完美重建的DC损失应该接近0
            assert dc_loss < 1e-6, f"完美重建的DC损失过大: {dc_loss}, 任务: {config['task']}"
    
    def test_dc_loss_with_normalization(self, sample_data, normalization_stats, dc_configs, device):
        """测试带归一化的DC损失计算"""
        gt_data, pred_data = sample_data
        mean, std = normalization_stats
        
        # 归一化数据（模拟z-score归一化）
        gt_normalized = (gt_data - mean) / std
        pred_normalized = (pred_data - mean) / std
        
        for config in dc_configs:
            # 在原值域生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 创建带归一化的DC损失函数
            dc_loss_fn = DataConsistencyLoss(config, mean=mean, std=std)
            
            # 计算DC损失（输入是z-score域，但计算在原值域）
            dc_loss = dc_loss_fn(pred_normalized, observed)
            
            # 检查损失有效性
            assert torch.isfinite(dc_loss), "带归一化的DC损失应该是有限值"
            assert dc_loss >= 0, "带归一化的DC损失应该非负"
            
            # 手动验证计算
            # 1. 反归一化预测
            pred_denorm = pred_normalized * std + mean
            # 2. 应用观测算子
            pred_observed = apply_degradation_operator(pred_denorm, config)
            # 3. 计算MSE
            expected_loss = torch.mean((pred_observed - observed) ** 2)
            
            # 应该匹配（在数值精度范围内）
            assert torch.allclose(dc_loss, expected_loss, atol=1e-6), \
                f"DC损失计算不匹配: {dc_loss} vs {expected_loss}"
    
    def test_dc_loss_gradient_flow(self, sample_data, dc_configs, device):
        """测试DC损失的梯度流动"""
        gt_data, _ = sample_data
        
        for config in dc_configs:
            # 生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 创建需要梯度的预测
            pred_data = gt_data.clone().requires_grad_(True)
            
            # 创建DC损失函数
            dc_loss_fn = DataConsistencyLoss(config)
            
            # 计算DC损失
            dc_loss = dc_loss_fn(pred_data, observed)
            
            # 反向传播
            dc_loss.backward()
            
            # 检查梯度
            assert pred_data.grad is not None, "预测数据应该有梯度"
            assert torch.isfinite(pred_data.grad).all(), "梯度应该是有限值"
            assert not torch.isnan(pred_data.grad).any(), "梯度不应该包含NaN"
            
            # 梯度范数应该合理
            grad_norm = torch.norm(pred_data.grad)
            assert grad_norm > 1e-8, f"梯度范数过小: {grad_norm}"
            assert grad_norm < 1e3, f"梯度范数过大: {grad_norm}"
    
    def test_dc_loss_consistency_with_h_operator(self, sample_data, dc_configs, device):
        """测试DC损失与H算子的一致性"""
        gt_data, pred_data = sample_data
        
        for config in dc_configs:
            # 生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 方法1：使用DC损失函数
            dc_loss_fn = DataConsistencyLoss(config)
            dc_loss1 = dc_loss_fn(pred_data, observed)
            
            # 方法2：手动计算
            pred_observed = apply_degradation_operator(pred_data, config)
            dc_loss2 = torch.mean((pred_observed - observed) ** 2)
            
            # 两种方法应该得到相同结果
            assert torch.allclose(dc_loss1, dc_loss2, atol=1e-6), \
                f"DC损失计算不一致: {dc_loss1} vs {dc_loss2}, 任务: {config['task']}"
    
    def test_total_loss_integration(self, sample_data, dc_configs, device):
        """测试DC损失在总损失中的集成"""
        gt_data, pred_data = sample_data
        
        for config in dc_configs:
            # 生成观测数据
            observed = apply_degradation_operator(gt_data, config)
            
            # 创建总损失函数
            total_loss_fn = TotalLoss(
                rec_weight=1.0,
                spec_weight=0.5,
                dc_weight=1.0,
                dc_config=config
            )
            
            # 计算总损失
            total_loss, loss_dict = total_loss_fn(pred_data, gt_data, observed)
            
            # 检查损失字典
            assert 'rec_loss' in loss_dict, "缺少重建损失"
            assert 'spec_loss' in loss_dict, "缺少频谱损失"
            assert 'dc_loss' in loss_dict, "缺少DC损失"
            
            # 检查各项损失
            rec_loss = loss_dict['rec_loss']
            spec_loss = loss_dict['spec_loss']
            dc_loss = loss_dict['dc_loss']
            
            assert torch.isfinite(rec_loss), "重建损失应该是有限值"
            assert torch.isfinite(spec_loss), "频谱损失应该是有限值"
            assert torch.isfinite(dc_loss), "DC损失应该是有限值"
            
            # 检查总损失计算
            expected_total = 1.0 * rec_loss + 0.5 * spec_loss + 1.0 * dc_loss
            assert torch.allclose(total_loss, expected_total, atol=1e-6), \
                f"总损失计算不正确: {total_loss} vs {expected_total}"
    
    def test_dc_loss_batch_consistency(self, dc_configs, device):
        """测试DC损失的批处理一致性"""
        # 创建单个样本
        single_gt = torch.randn(1, 3, 64, 64, device=device)
        single_pred = single_gt + 0.1 * torch.randn_like(single_gt)
        
        # 创建批次数据
        batch_size = 4
        batch_gt = single_gt.repeat(batch_size, 1, 1, 1)
        batch_pred = single_pred.repeat(batch_size, 1, 1, 1)
        
        for config in dc_configs:
            # 单个样本的损失
            single_observed = apply_degradation_operator(single_gt, config)
            dc_loss_fn = DataConsistencyLoss(config)
            single_loss = dc_loss_fn(single_pred, single_observed)
            
            # 批次样本的损失
            batch_observed = apply_degradation_operator(batch_gt, config)
            batch_loss = dc_loss_fn(batch_pred, batch_observed)
            
            # 批次损失应该等于单个损失（因为所有样本相同）
            assert torch.allclose(batch_loss, single_loss, atol=1e-6), \
                f"批次损失不一致: {batch_loss} vs {single_loss}"
    
    def test_dc_loss_numerical_stability(self, device):
        """测试DC损失的数值稳定性"""
        # 测试极小值
        small_data = torch.full((1, 1, 32, 32), 1e-8, device=device)
        small_pred = small_data + 1e-9 * torch.randn_like(small_data)
        
        # 测试极大值
        large_data = torch.full((1, 1, 32, 32), 1e8, device=device)
        large_pred = large_data + 1e7 * torch.randn_like(large_data)
        
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        dc_loss_fn = DataConsistencyLoss(config)
        
        # 测试极小值
        small_observed = apply_degradation_operator(small_data, config)
        small_loss = dc_loss_fn(small_pred, small_observed)
        assert torch.isfinite(small_loss), "极小值情况下DC损失不稳定"
        
        # 测试极大值
        large_observed = apply_degradation_operator(large_data, config)
        large_loss = dc_loss_fn(large_pred, large_observed)
        assert torch.isfinite(large_loss), "极大值情况下DC损失不稳定"
    
    def test_dc_loss_zero_prediction(self, sample_data, dc_configs, device):
        """测试零预测的DC损失"""
        gt_data, _ = sample_data
        zero_pred = torch.zeros_like(gt_data)
        
        for config in dc_configs:
            observed = apply_degradation_operator(gt_data, config)
            dc_loss_fn = DataConsistencyLoss(config)
            
            dc_loss = dc_loss_fn(zero_pred, observed)
            
            # 零预测的DC损失应该等于 ||H(0) - y||^2 = ||0 - y||^2 = ||y||^2
            zero_observed = apply_degradation_operator(zero_pred, config)
            expected_loss = torch.mean((zero_observed - observed) ** 2)
            
            assert torch.allclose(dc_loss, expected_loss, atol=1e-6), \
                f"零预测DC损失不正确: {dc_loss} vs {expected_loss}"


class TestDataConsistencyLossEdgeCases:
    """数据一致性损失边缘情况测试"""
    
    def test_dc_loss_with_different_dtypes(self, device):
        """测试不同数据类型的DC损失"""
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        # 测试float32
        data_f32 = torch.randn(1, 1, 32, 32, dtype=torch.float32, device=device)
        pred_f32 = data_f32 + 0.1 * torch.randn_like(data_f32)
        observed_f32 = apply_degradation_operator(data_f32, config)
        
        dc_loss_fn = DataConsistencyLoss(config)
        loss_f32 = dc_loss_fn(pred_f32, observed_f32)
        
        assert loss_f32.dtype == torch.float32, "float32输入应产生float32输出"
        
        # 测试float64
        data_f64 = data_f32.double()
        pred_f64 = pred_f32.double()
        observed_f64 = observed_f32.double()
        
        loss_f64 = dc_loss_fn(pred_f64, observed_f64)
        assert loss_f64.dtype == torch.float64, "float64输入应产生float64输出"
        
        # 结果应该在精度范围内一致
        assert torch.allclose(loss_f32.double(), loss_f64, atol=1e-6), \
            "不同精度的结果应该一致"
    
    def test_dc_loss_memory_efficiency(self, device):
        """测试DC损失的内存效率"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA进行内存测试")
        
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 创建较大的数据
        large_data = torch.randn(4, 3, 256, 256, device=device)
        large_pred = large_data + 0.1 * torch.randn_like(large_data)
        large_observed = apply_degradation_operator(large_data, config)
        
        dc_loss_fn = DataConsistencyLoss(config)
        
        # 计算损失
        dc_loss = dc_loss_fn(large_pred, large_observed)
        
        # 检查内存使用
        peak_memory = torch.cuda.memory_allocated()
        memory_usage = peak_memory - initial_memory
        
        # 清理
        del large_data, large_pred, large_observed, dc_loss
        torch.cuda.empty_cache()
        
        # 内存使用应该合理（这里只是一个粗略的检查）
        expected_memory = 4 * 3 * 256 * 256 * 4 * 3  # 大约3个tensor的大小
        assert memory_usage < expected_memory, f"内存使用过多: {memory_usage} bytes"
    
    def test_dc_loss_with_nan_input(self, device):
        """测试包含NaN的输入"""
        config = {
            'task': 'super_resolution',
            'scale_factor': 2,
            'kernel_size': 5,
            'sigma': 1.0,
            'interpolation': 'bilinear'
        }
        
        # 创建包含NaN的数据
        data = torch.randn(1, 1, 32, 32, device=device)
        data[0, 0, 0, 0] = float('nan')
        
        observed = torch.randn(1, 1, 16, 16, device=device)
        
        dc_loss_fn = DataConsistencyLoss(config)
        
        # DC损失应该能检测到NaN
        dc_loss = dc_loss_fn(data, observed)
        assert torch.isnan(dc_loss), "包含NaN的输入应该产生NaN损失"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])