"""
综合框架测试套件
验证PDEBench系统的核心功能，包括数据一致性、模型接口、损失函数等
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import tempfile
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import get_model
from ops.degradation import get_observation_operator
from ops.loss import CombinedLoss
from datasets.pdebench_dataset import PDEBenchDataset
from utils.validation import validate_config
from utils.reproducibility import set_seed


class TestDataConsistency:
    """数据一致性测试"""
    
    @pytest.fixture
    def sample_config(self):
        """样本配置"""
        return {
            'data': {
                'dataset_name': '2D-cfd-ns',
                'data_path': 'data/pdebench',
                'image_size': 64,
                'observation_mode': 'SR',
                'sr_scale': 4,
                'sigma': 1.0,
                'task': 'SRx4-DR2D-64'
            },
            'model': {
                'name': 'SwinUNet',
                'in_channels': 3,
                'out_channels': 3,
                'img_size': 64,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24]
            },
            'loss': {
                'reconstruction_weight': 1.0,
                'spectral_weight': 0.5,
                'dc_weight': 1.0,
                'spectral_loss_type': 'fft',
                'spectral_low_freq': 16
            }
        }
    
    def test_observation_operator_consistency(self, sample_config):
        """测试观测算子一致性"""
        # 创建观测算子
        obs_operator = get_observation_operator(sample_config['data'])
        
        # 创建测试数据
        batch_size = 2
        channels = sample_config['model']['in_channels']
        height = width = sample_config['data']['image_size']
        
        # 创建高分辨率输入
        hr_input = torch.randn(batch_size, channels, height, width)
        
        # 生成观测数据
        lr_observed = obs_operator(hr_input)
        
        # 验证输出形状
        expected_lr_size = height // sample_config['data']['sr_scale']
        assert lr_observed.shape == (batch_size, channels, expected_lr_size, expected_lr_size), \
            f"观测数据形状不匹配: {lr_observed.shape} vs 期望 {(batch_size, channels, expected_lr_size, expected_lr_size)}"
        
        # 验证数值范围
        assert not torch.isnan(lr_observed).any(), "观测数据包含NaN值"
        assert not torch.isinf(lr_observed).any(), "观测数据包含Inf值"
    
    def test_data_consistency_pipeline(self, sample_config):
        """测试数据一致性管道"""
        # 创建临时数据集
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建模拟数据文件
            data_path = Path(tmp_dir) / "test_data.npz"
            
            # 生成模拟PDEBench数据
            n_samples = 10
            image_size = sample_config['data']['image_size']
            n_channels = sample_config['model']['in_channels']
            
            # 创建模拟数据
            data = {
                'u': np.random.randn(n_samples, n_channels, image_size, image_size).astype(np.float32),
                'v': np.random.randn(n_samples, n_channels, image_size, image_size).astype(np.float32),
                'p': np.random.randn(n_samples, n_channels, image_size, image_size).astype(np.float32)
            }
            
            np.savez(data_path, **data)
            
            # 创建数据集配置
            dataset_config = sample_config['data'].copy()
            dataset_config['data_path'] = str(data_path)
            dataset_config['keys'] = ['u', 'v', 'p']
            
            # 创建数据集
            dataset = PDEBenchDataset(dataset_config, split='train')
            
            # 测试数据加载
            assert len(dataset) == n_samples, f"数据集长度不匹配: {len(dataset)} vs {n_samples}"
            
            # 测试数据格式
            sample = dataset[0]
            assert 'input' in sample, "样本中缺少input字段"
            assert 'target' in sample, "样本中缺少target字段"
            
            # 验证数据形状
            input_data = sample['input']
            target_data = sample['target']
            
            assert input_data.shape[0] == n_channels, f"输入通道数不匹配: {input_data.shape[0]} vs {n_channels}"
            assert target_data.shape[0] == n_channels, f"目标通道数不匹配: {target_data.shape[0]} vs {n_channels}"


class TestModelInterface:
    """模型接口测试"""
    
    @pytest.fixture
    def model_config(self):
        """模型配置"""
        return {
            'name': 'SwinUNet',
            'in_channels': 3,
            'out_channels': 3,
            'img_size': 64,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24]
        }
    
    def test_model_forward_pass(self, model_config):
        """测试模型前向传播"""
        # 创建模型
        model = get_model(model_config)
        
        # 创建测试输入
        batch_size = 2
        in_channels = model_config['in_channels']
        img_size = model_config['img_size']
        
        test_input = torch.randn(batch_size, in_channels, img_size, img_size)
        
        # 前向传播
        output = model(test_input)
        
        # 验证输出
        assert output is not None, "模型输出为None"
        assert not torch.isnan(output).any(), "模型输出包含NaN值"
        assert not torch.isinf(output).any(), "模型输出包含Inf值"
        
        # 验证输出形状
        expected_shape = (batch_size, model_config['out_channels'], img_size, img_size)
        assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
    
    def test_model_gradient_flow(self, model_config):
        """测试模型梯度流"""
        # 创建模型
        model = get_model(model_config)
        
        # 创建测试输入和目标
        batch_size = 2
        in_channels = model_config['in_channels']
        out_channels = model_config['out_channels']
        img_size = model_config['img_size']
        
        test_input = torch.randn(batch_size, in_channels, img_size, img_size, requires_grad=True)
        test_target = torch.randn(batch_size, out_channels, img_size, img_size)
        
        # 前向传播
        output = model(test_input)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(output, test_target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含NaN"
                assert not torch.isinf(param.grad).any(), f"参数 {name} 的梯度包含Inf"
                assert grad_norm > 0, f"参数 {name} 的梯度为零"
        
        assert has_gradients, "模型没有任何参数有梯度"
    
    def test_model_reproducibility(self, model_config):
        """测试模型可重现性"""
        # 设置随机种子
        set_seed(42)
        
        # 创建模型
        model1 = get_model(model_config)
        
        # 再次设置相同的种子
        set_seed(42)
        model2 = get_model(model_config)
        
        # 创建相同的输入
        test_input = torch.randn(2, model_config['in_channels'], 
                                model_config['img_size'], model_config['img_size'])
        
        # 前向传播
        output1 = model1(test_input)
        output2 = model2(test_input)
        
        # 验证输出一致性
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)


class TestLossFunctions:
    """损失函数测试"""
    
    @pytest.fixture
    def loss_config(self):
        """损失函数配置"""
        return {
            'reconstruction_weight': 1.0,
            'spectral_weight': 0.5,
            'dc_weight': 1.0,
            'spectral_loss_type': 'fft',
            'spectral_low_freq': 16,
            'observation_mode': 'SR',
            'sr_scale': 4
        }
    
    def test_combined_loss_forward(self, loss_config):
        """测试组合损失函数前向传播"""
        # 创建损失函数
        criterion = CombinedLoss(loss_config)
        
        # 创建测试数据
        batch_size = 2
        channels = 3
        height = width = 64
        
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randn(batch_size, channels, height, width)
        
        # 计算损失
        loss_dict = criterion(pred, target)
        
        # 验证损失字典
        assert 'total_loss' in loss_dict, "损失字典缺少total_loss"
        assert 'reconstruction_loss' in loss_dict, "损失字典缺少reconstruction_loss"
        
        # 验证损失值
        assert not torch.isnan(loss_dict['total_loss']), "总损失为NaN"
        assert not torch.isinf(loss_dict['total_loss']), "总损失为Inf"
        assert loss_dict['total_loss'].item() > 0, "总损失为零或负数"
    
    def test_loss_components(self, loss_config):
        """测试损失函数组件"""
        # 创建损失函数
        criterion = CombinedLoss(loss_config)
        
        # 创建测试数据
        batch_size = 2
        channels = 3
        height = width = 64
        
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randn(batch_size, channels, height, width)
        
        # 计算各个组件损失
        rec_loss = criterion.reconstruction_loss(pred, target)
        spec_loss = criterion.spectral_loss(pred, target)
        
        # 验证组件损失
        assert not torch.isnan(rec_loss), "重建损失为NaN"
        assert not torch.isinf(rec_loss), "重建损失为Inf"
        assert rec_loss.item() >= 0, "重建损失为负数"
        
        assert not torch.isnan(spec_loss), "频谱损失为NaN"
        assert not torch.isinf(spec_loss), "频谱损失为Inf"
        assert spec_loss.item() >= 0, "频谱损失为负数"
    
    def test_loss_weighting(self, loss_config):
        """测试损失权重"""
        # 创建不同权重的损失函数
        config1 = loss_config.copy()
        config1['spectral_weight'] = 0.0
        
        config2 = loss_config.copy()
        config2['spectral_weight'] = 1.0
        
        criterion1 = CombinedLoss(config1)
        criterion2 = CombinedLoss(config2)
        
        # 创建测试数据
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        # 计算损失
        loss1 = criterion1(pred, target)
        loss2 = criterion2(pred, target)
        
        # 验证权重影响
        assert loss1['spectral_loss'].item() == 0.0, "权重为0时频谱损失不为0"
        assert loss2['spectral_loss'].item() > 0, "权重为1时频谱损失为0"


class TestReproducibility:
    """可重现性测试"""
    
    def test_seed_setting(self):
        """测试随机种子设置"""
        # 设置种子
        set_seed(42)
        result1 = torch.randn(5, 5)
        
        # 再次设置相同的种子
        set_seed(42)
        result2 = torch.randn(5, 5)
        
        # 验证结果一致性
        torch.testing.assert_close(result1, result2)
    
    def test_numpy_torch_consistency(self):
        """测试NumPy和PyTorch的一致性"""
        # 设置种子
        set_seed(123)
        
        # NumPy随机数
        np_result = np.random.randn(10, 10)
        
        # PyTorch随机数
        torch_result = torch.randn(10, 10).numpy()
        
        # 验证统计特性
        assert abs(np.mean(np_result) - np.mean(torch_result)) < 0.1
        assert abs(np.std(np_result) - np.std(torch_result)) < 0.1


class TestConfiguration:
    """配置测试"""
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = {
            'experiment': {'name': 'test', 'seed': 42},
            'data': {'dataset_name': 'test', 'data_path': 'test'},
            'model': {'name': 'SwinUNet', 'in_channels': 3, 'out_channels': 3},
            'training': {'epochs': 10, 'batch_size': 16}
        }
        
        # 验证应该通过
        assert validate_config(valid_config) is True
        
        # 无效配置（缺少必要字段）
        invalid_config = {
            'experiment': {'name': 'test'},
            # 缺少data、model、training字段
        }
        
        # 验证应该失败
        assert validate_config(invalid_config) is False
    
    def test_config_defaults(self):
        """测试配置默认值"""
        # 最小配置
        minimal_config = {
            'data': {'dataset_name': 'test', 'data_path': 'test'},
            'model': {'name': 'SwinUNet'},
            'training': {'epochs': 10}
        }
        
        # 验证配置
        result = validate_config(minimal_config)
        
        # 应该自动填充默认值
        assert result is True


class TestResourceMonitoring:
    """资源监控测试"""
    
    def test_resource_usage_computation(self):
        """测试资源使用计算"""
        from utils.metrics import compute_resource_usage
        
        # 创建简单模型
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding=1)
        )
        
        # 计算资源使用
        resource_stats = compute_resource_usage(model, input_shape=(1, 3, 64, 64))
        
        # 验证结果
        assert 'total_params' in resource_stats
        assert 'trainable_params' in resource_stats
        assert 'flops_g' in resource_stats
        assert 'memory_mb' in resource_stats
        
        assert resource_stats['total_params'] > 0, "总参数数为零"
        assert resource_stats['trainable_params'] > 0, "可训练参数数为零"
        assert resource_stats['flops_g'] >= 0, "FLOPs为负数"
        assert resource_stats['memory_mb'] > 0, "内存使用为零"


def test_comprehensive_pipeline():
    """综合管道测试"""
    # 创建完整配置
    config = {
        'experiment': {
            'name': 'comprehensive_test',
            'seed': 42,
            'output_dir': 'test_runs'
        },
        'data': {
            'dataset_name': '2D-cfd-ns',
            'data_path': 'test_data',
            'image_size': 64,
            'observation_mode': 'SR',
            'sr_scale': 4,
            'keys': ['u', 'v', 'p'],
            'batch_size': 4
        },
        'model': {
            'name': 'SwinUNet',
            'in_channels': 3,
            'out_channels': 3,
            'img_size': 64,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24]
        },
        'training': {
            'epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'spectral_weight': 0.5,
            'dc_weight': 1.0
        }
    }
    
    # 验证配置
    assert validate_config(config)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    # 创建模型
    model = get_model(config['model'])
    
    # 创建损失函数
    criterion = CombinedLoss(config['loss'])
    
    # 创建观测算子
    obs_operator = get_observation_operator(config['data'])
    
    # 创建测试数据
    batch_size = config['data']['batch_size']
    channels = config['model']['in_channels']
    height = width = config['data']['image_size']
    
    # 高分辨率输入
    hr_input = torch.randn(batch_size, channels, height, width)
    
    # 生成观测数据
    lr_observed = obs_operator(hr_input)
    
    # 模型推理
    with torch.no_grad():
        pred_hr = model(lr_observed)
    
    # 计算损失
    loss_dict = criterion(pred_hr, hr_input)
    
    # 验证结果
    assert pred_hr.shape == hr_input.shape, "输出形状不匹配"
    assert not torch.isnan(loss_dict['total_loss']), "损失为NaN"
    assert loss_dict['total_loss'].item() > 0, "损失为零或负数"
    
    logger.info("✅ 综合管道测试通过")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])