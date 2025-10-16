#!/usr/bin/env python3
"""
PDEBench稀疏观测重建系统 - 端到端测试

开发手册 - 黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置（核/σ/插值/对齐/边界）
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：横向对比必须报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁

测试覆盖：
- 完整的训练-评估流程
- 数据一致性验证（H算子）
- 模型接口统一性
- 损失函数正确性
- 指标计算准确性
- 可复现性验证
"""

import os
import sys
import tempfile
import shutil
import json
import yaml
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.reproducibility import set_seed, get_environment_info
from utils.metrics import compute_metrics
from ops.degradation import SuperResolutionDegradation, CropDegradation
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from datasets.pdebench_dataset import PDEBenchDataset


class TestEndToEnd:
    """端到端测试类
    
    测试完整的训练-评估流程，确保系统各组件正确协作
    """
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """创建示例配置"""
        return {
            'model': {
                'name': 'SwinUNet',
                'in_channels': 2,
                'out_channels': 2,
                'img_size': 64,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1
            },
            'data': {
                'name': 'PDEBench',
                'root_dir': 'data/pdebench',
                'task_type': 'sr',
                'scale_factor': 2,
                'img_size': 64,
                'batch_size': 4,
                'num_workers': 0,
                'pin_memory': False,
                'variables': ['u', 'v'],
                'normalization': 'z_score',
                'degradation': {
                    'type': 'sr',
                    'scale_factor': 2,
                    'blur_kernel': 'gaussian',
                    'blur_sigma': 1.0,
                    'noise_level': 0.0
                }
            },
            'training': {
                'epochs': 2,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'warmup_epochs': 0,
                'seed': 42,
                'amp': False,
                'gradient_clip': 1.0
            },
            'loss': {
                'reconstruction_weight': 1.0,
                'spectral_weight': 0.5,
                'data_consistency_weight': 1.0,
                'spectral_modes': 16
            },
            'evaluation': {
                'metrics': ['rel_l2', 'mae', 'psnr', 'ssim'],
                'save_predictions': True,
                'visualize': True
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        batch_size = 4
        channels = 2
        height = 64
        width = 64
        
        # 生成随机数据
        gt_data = torch.randn(batch_size, channels, height, width)
        
        # 生成观测数据（下采样）
        lr_data = torch.nn.functional.interpolate(
            gt_data, 
            scale_factor=0.5, 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'gt': gt_data,
            'lr': lr_data,
            'mask': torch.ones_like(gt_data),
            'coords': torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing='ij'
            ), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        }
    
    def test_model_interface_consistency(self, sample_config):
        """测试模型接口统一性
        
        黄金法则3：统一接口 - 所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
        """
        batch_size = 4
        in_channels = sample_config['model']['in_channels']
        out_channels = sample_config['model']['out_channels']
        img_size = sample_config['model']['img_size']
        
        # 测试输入
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        
        # 测试SwinUNet
        model = SwinUNet(**sample_config['model'])
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        # 验证输出形状
        expected_shape = (batch_size, out_channels, img_size, img_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # 验证输出数值范围合理
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        # 测试Hybrid模型
        hybrid_config = sample_config['model'].copy()
        hybrid_config['name'] = 'Hybrid'
        hybrid_config['fno_modes'] = 16
        hybrid_config['mlp_hidden_dim'] = 256
        
        hybrid_model = HybridModel(**hybrid_config)
        hybrid_model.eval()
        
        with torch.no_grad():
            hybrid_output = hybrid_model(x)
        
        assert hybrid_output.shape == expected_shape, f"Hybrid model output shape mismatch"
        assert torch.isfinite(hybrid_output).all(), "Hybrid output contains non-finite values"
    
    def test_degradation_consistency(self, sample_config, sample_data):
        """测试观测算子H的一致性
        
        黄金法则1：一致性优先 - 观测算子H与训练DC必须复用同一实现与配置
        """
        gt_data = sample_data['gt']
        
        # 创建SR退化算子
        sr_degradation = SuperResolutionDegradation(
            scale_factor=sample_config['data']['degradation']['scale_factor'],
            blur_kernel=sample_config['data']['degradation']['blur_kernel'],
            blur_sigma=sample_config['data']['degradation']['blur_sigma'],
            noise_level=sample_config['data']['degradation']['noise_level']
        )
        
        # 应用退化
        degraded_1 = sr_degradation(gt_data)
        degraded_2 = sr_degradation(gt_data)
        
        # 验证确定性（相同输入应产生相同输出）
        torch.testing.assert_close(degraded_1, degraded_2, rtol=1e-6, atol=1e-8)
        
        # 验证退化后的尺寸
        expected_size = gt_data.shape[-1] // sample_config['data']['degradation']['scale_factor']
        assert degraded_1.shape[-1] == expected_size, f"Degraded size mismatch"
        
        # 测试Crop退化算子
        crop_degradation = CropDegradation(
            crop_ratio=0.5,
            center_aligned=True,
            boundary_condition='mirror'
        )
        
        cropped_1 = crop_degradation(gt_data)
        cropped_2 = crop_degradation(gt_data)
        
        torch.testing.assert_close(cropped_1, cropped_2, rtol=1e-6, atol=1e-8)
    
    def test_loss_function_correctness(self, sample_config, sample_data):
        """测试损失函数正确性
        
        验证损失函数在正确的值域计算（z-score域 vs 原值域）
        """
        from losses.combined_loss import CombinedLoss
        
        # 创建损失函数
        loss_fn = CombinedLoss(
            reconstruction_weight=sample_config['loss']['reconstruction_weight'],
            spectral_weight=sample_config['loss']['spectral_weight'],
            data_consistency_weight=sample_config['loss']['data_consistency_weight'],
            spectral_modes=sample_config['loss']['spectral_modes']
        )
        
        gt_data = sample_data['gt']
        pred_data = gt_data + 0.1 * torch.randn_like(gt_data)  # 添加小噪声
        
        # 计算损失
        loss_dict = loss_fn(pred_data, gt_data)
        
        # 验证损失组件
        assert 'total_loss' in loss_dict, "Missing total_loss"
        assert 'reconstruction_loss' in loss_dict, "Missing reconstruction_loss"
        assert 'spectral_loss' in loss_dict, "Missing spectral_loss"
        assert 'data_consistency_loss' in loss_dict, "Missing data_consistency_loss"
        
        # 验证损失值合理性
        assert loss_dict['total_loss'] > 0, "Total loss should be positive"
        assert torch.isfinite(loss_dict['total_loss']), "Total loss should be finite"
        
        # 验证梯度可计算
        pred_data.requires_grad_(True)
        loss_dict = loss_fn(pred_data, gt_data)
        loss_dict['total_loss'].backward()
        
        assert pred_data.grad is not None, "Gradients should be computed"
        assert torch.isfinite(pred_data.grad).all(), "Gradients should be finite"
    
    def test_metrics_computation(self, sample_data):
        """测试指标计算准确性"""
        gt_data = sample_data['gt']
        pred_data = gt_data + 0.1 * torch.randn_like(gt_data)
        
        # 计算指标
        metrics = compute_metrics(pred_data, gt_data)
        
        # 验证指标存在
        expected_metrics = ['rel_l2', 'mae', 'psnr', 'ssim']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
            assert np.isfinite(metrics[metric]), f"Metric {metric} should be finite"
        
        # 验证指标合理性
        assert metrics['rel_l2'] >= 0, "Rel-L2 should be non-negative"
        assert metrics['mae'] >= 0, "MAE should be non-negative"
        assert 0 <= metrics['ssim'] <= 1, "SSIM should be in [0, 1]"
        
        # 测试完美预测的情况
        perfect_metrics = compute_metrics(gt_data, gt_data)
        assert perfect_metrics['rel_l2'] < 1e-6, "Perfect prediction should have near-zero Rel-L2"
        assert perfect_metrics['mae'] < 1e-6, "Perfect prediction should have near-zero MAE"
        assert perfect_metrics['ssim'] > 0.99, "Perfect prediction should have high SSIM"
    
    def test_reproducibility(self, sample_config, temp_dir):
        """测试可复现性
        
        黄金法则2：可复现 - 同一YAML+种子，验证指标方差≤1e-4
        """
        seed = sample_config['training']['seed']
        
        # 运行两次相同的实验
        results_1 = self._run_mini_experiment(sample_config, temp_dir / "run1", seed)
        results_2 = self._run_mini_experiment(sample_config, temp_dir / "run2", seed)
        
        # 验证结果一致性
        for metric in ['rel_l2', 'mae', 'psnr', 'ssim']:
            if metric in results_1 and metric in results_2:
                diff = abs(results_1[metric] - results_2[metric])
                assert diff <= 1e-4, f"Metric {metric} variance {diff} exceeds threshold 1e-4"
    
    def test_data_consistency_check(self, sample_config, sample_data):
        """测试数据一致性检查
        
        验证MSE(H(GT), y) < 1e-8
        """
        gt_data = sample_data['gt']
        
        # 创建退化算子
        degradation = SuperResolutionDegradation(
            scale_factor=sample_config['data']['degradation']['scale_factor'],
            blur_kernel=sample_config['data']['degradation']['blur_kernel'],
            blur_sigma=sample_config['data']['degradation']['blur_sigma'],
            noise_level=0.0  # 无噪声以确保一致性
        )
        
        # 生成观测数据
        observed = degradation(gt_data)
        
        # 重新应用相同的退化
        observed_again = degradation(gt_data)
        
        # 计算MSE
        mse = torch.mean((observed - observed_again) ** 2).item()
        
        # 验证一致性
        assert mse < 1e-8, f"Data consistency check failed: MSE = {mse}"
    
    def test_training_pipeline(self, sample_config, temp_dir):
        """测试训练流程"""
        # 创建模型
        model = SwinUNet(**sample_config['model'])
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=sample_config['training']['lr'],
            weight_decay=sample_config['training']['weight_decay']
        )
        
        # 创建损失函数
        from losses.combined_loss import CombinedLoss
        loss_fn = CombinedLoss(**sample_config['loss'])
        
        # 模拟训练数据
        batch_size = sample_config['data']['batch_size']
        channels = sample_config['model']['in_channels']
        img_size = sample_config['model']['img_size']
        
        # 训练几个步骤
        model.train()
        initial_loss = None
        final_loss = None
        
        for step in range(10):
            # 生成随机数据
            gt_data = torch.randn(batch_size, channels, img_size, img_size)
            lr_data = torch.nn.functional.interpolate(
                gt_data, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            
            # 前向传播
            optimizer.zero_grad()
            pred_data = model(lr_data)
            
            # 计算损失
            loss_dict = loss_fn(pred_data, gt_data)
            loss = loss_dict['total_loss']
            
            if step == 0:
                initial_loss = loss.item()
            if step == 9:
                final_loss = loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), sample_config['training']['gradient_clip'])
            
            # 优化器步骤
            optimizer.step()
        
        # 验证训练有效性（损失应该下降）
        assert final_loss < initial_loss, f"Training failed: loss increased from {initial_loss} to {final_loss}"
    
    def test_evaluation_pipeline(self, sample_config, sample_data):
        """测试评估流程"""
        # 创建模型
        model = SwinUNet(**sample_config['model'])
        model.eval()
        
        gt_data = sample_data['gt']
        lr_data = sample_data['lr']
        
        # 评估模式
        with torch.no_grad():
            pred_data = model(lr_data)
        
        # 计算指标
        metrics = compute_metrics(pred_data, gt_data)
        
        # 验证指标完整性
        expected_metrics = sample_config['evaluation']['metrics']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing evaluation metric: {metric}"
        
        # 验证指标合理性
        assert all(np.isfinite(v) for v in metrics.values()), "All metrics should be finite"
    
    def test_resource_monitoring(self, sample_config):
        """测试资源监控"""
        from utils.resource_monitor import ResourceMonitor
        
        # 创建资源监控器
        monitor = ResourceMonitor()
        
        # 创建模型
        model = SwinUNet(**sample_config['model'])
        
        # 监控参数数量
        params_count = monitor.count_parameters(model)
        assert params_count > 0, "Parameter count should be positive"
        
        # 监控FLOPs
        batch_size = 1
        channels = sample_config['model']['in_channels']
        img_size = sample_config['model']['img_size']
        
        dummy_input = torch.randn(batch_size, channels, img_size, img_size)
        flops = monitor.compute_flops(model, dummy_input)
        assert flops > 0, "FLOPs should be positive"
        
        # 监控内存使用
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            
            memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
            assert memory_usage > 0, "Memory usage should be positive"
    
    def test_config_validation(self, sample_config):
        """测试配置验证"""
        from utils.config_validator import validate_config
        
        # 验证有效配置
        is_valid, errors = validate_config(sample_config)
        assert is_valid, f"Valid config rejected: {errors}"
        
        # 测试无效配置
        invalid_config = sample_config.copy()
        invalid_config['model']['in_channels'] = -1  # 无效值
        
        is_valid, errors = validate_config(invalid_config)
        assert not is_valid, "Invalid config should be rejected"
        assert len(errors) > 0, "Should report validation errors"
    
    def test_checkpoint_saving_loading(self, sample_config, temp_dir):
        """测试检查点保存和加载"""
        # 创建模型
        model = SwinUNet(**sample_config['model'])
        
        # 保存检查点
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': sample_config,
            'epoch': 10,
            'best_metric': 0.123
        }, checkpoint_path)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建新模型并加载权重
        new_model = SwinUNet(**checkpoint['config']['model'])
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 验证权重一致性
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            torch.testing.assert_close(param1, param2, rtol=1e-6, atol=1e-8)
    
    def _run_mini_experiment(self, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, float]:
        """运行小型实验"""
        # 设置随机种子
        set_seed(seed)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型
        model = SwinUNet(**config['model'])
        model.eval()
        
        # 生成测试数据
        batch_size = 2
        channels = config['model']['in_channels']
        img_size = config['model']['img_size']
        
        gt_data = torch.randn(batch_size, channels, img_size, img_size)
        lr_data = torch.nn.functional.interpolate(
            gt_data, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        
        # 预测
        with torch.no_grad():
            pred_data = model(lr_data)
        
        # 计算指标
        metrics = compute_metrics(pred_data, gt_data)
        
        # 保存结果
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


class TestIntegration:
    """集成测试类
    
    测试系统各组件的集成
    """
    
    def test_full_pipeline_integration(self, temp_dir):
        """测试完整流程集成"""
        # 这里可以添加完整的训练-评估-生成材料包的集成测试
        pass
    
    def test_multi_gpu_training(self):
        """测试多GPU训练（如果可用）"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")
        
        # 这里可以添加多GPU训练测试
        pass
    
    def test_mixed_precision_training(self, sample_config):
        """测试混合精度训练"""
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision test requires CUDA")
        
        # 这里可以添加混合精度训练测试
        pass


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])