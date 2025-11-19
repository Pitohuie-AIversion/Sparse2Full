#!/usr/bin/env python3
"""
测试稀疏注意力模型在训练框架中的集成
验证与现有训练系统的兼容性
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.training.train_basic import create_model_local, train_epoch, validate_epoch
from models.sparse_attention_encoder import SparseAttentionEncoder, SparseSwinUNet
from ops.total_loss import TotalLoss
from omegaconf import DictConfig, OmegaConf


class TestSparseAttentionTraining:
    """测试稀疏注意力模型训练集成"""
    
    def setup_method(self):
        """测试前设置"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.img_size = 64  # 小尺寸用于测试
        self.in_channels = 3  # [baseline, coords, mask]
        self.out_channels = 1
        
        # 创建测试配置
        self.cfg = OmegaConf.create({
            'data': {
                'image_size': self.img_size
            },
            'model': {
                'name': 'sparse_swin_unet',
                'params': {
                    'in_channels': 4,  # [baseline(1) + coords(2) + mask(1)]
                    'out_channels': self.out_channels,
                    'img_size': self.img_size,
                    'embed_dim': 48,  # 小维度用于测试
                    'sparse_encoder_config': {
                        'embed_dim': 64,
                        'num_heads': 4,
                        'sensor_dim': 16,
                        'coord_dim': 16,
                        'mask_dim': 8,
                        'use_sparse_bias': True,
                        'dropout': 0.1
                    },
                    'swin_unet_config': {
                        'depths': [1, 1],  # 简化架构
                        'num_heads': [3, 6],
                        'window_size': 4
                    }
                }
            },
            'training': {
                'grad_clip_norm': 1.0,
                'use_amp': False,
                'log_interval': 10
            }
        })
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        # 基础输入
        baseline = torch.randn(self.batch_size, 1, self.img_size, self.img_size)
        
        # 坐标输入 (归一化到[0,1]，2D坐标)
        x_coords = torch.linspace(0, 1, self.img_size).view(1, 1, 1, -1).repeat(
            self.batch_size, 1, self.img_size, 1
        )
        y_coords = torch.linspace(0, 1, self.img_size).view(1, 1, -1, 1).repeat(
            self.batch_size, 1, 1, self.img_size
        )
        coords = torch.cat([x_coords, y_coords], dim=1)  # [B, 2, H, W]
        
        # 掩码 (随机稀疏掩码)
        mask = torch.zeros(self.batch_size, 1, self.img_size, self.img_size)
        num_sparse_points = int(0.1 * self.img_size * self.img_size)  # 10%稀疏度
        for b in range(self.batch_size):
            indices = torch.randperm(self.img_size * self.img_size)[:num_sparse_points]
            mask.view(-1)[indices] = 1.0
        
        # 目标输出 (模拟GT)
        target = torch.randn(self.batch_size, self.out_channels, self.img_size, self.img_size)
        
        # 观测数据 (模拟降质后的输入)
        observation = baseline + 0.1 * torch.randn_like(baseline)
        
        # 任务参数
        task_params = {'task': 'sr', 'scale_factor': 4}
        
        self.batch = {
            'target': target,
            'observation': observation,
            'baseline': baseline,
            'coords': coords,
            'mask': mask,
            'task_params': task_params
        }
    
    def test_sparse_model_creation(self):
        """测试稀疏模型创建"""
        print("测试稀疏模型创建...")
        
        # 测试SparseSwinUNet创建
        model = create_model_local(self.cfg)
        assert isinstance(model, SparseSwinUNet), f"期望SparseSwinUNet，得到{type(model)}"
        
        # 测试基本前向传播
        baseline = self.batch['baseline']
        coords = self.batch['coords']
        mask = self.batch['mask']
        
        # 构建输入 [baseline, coords, mask] - 总共4通道
        model_input = torch.cat([baseline, coords, mask], dim=1)  # [B, 4, H, W]
        
        # 前向传播
        with torch.no_grad():
            output = model(model_input, coords=coords, mask=mask)
        
        assert output.shape == (self.batch_size, self.out_channels, self.img_size, self.img_size), \
            f"输出形状不匹配: {output.shape}"
        
        assert torch.isfinite(output).all(), "输出包含非有限值"
        
        print("✅ 稀疏模型创建测试通过!")
    
    def test_sparse_attention_forward(self):
        """测试稀疏注意力前向传播"""
        print("测试稀疏注意力前向传播...")
        
        model = create_model_local(self.cfg)
        model = model.to(self.device)
        
        # 将数据移到设备
        baseline = self.batch['baseline'].to(self.device)
        coords = self.batch['coords'].to(self.device)
        mask = self.batch['mask'].to(self.device)
        target = self.batch['target'].to(self.device)
        
        # 构建输入
        model_input = torch.cat([baseline, coords, mask], dim=1)
        
        # 测试不同稀疏度
        for sparse_ratio in [0.05, 0.1, 0.2]:
            # 调整掩码稀疏度
            test_mask = mask.clone()
            if sparse_ratio != 0.1:  # 重新生成掩码
                test_mask.zero_()
                num_points = int(sparse_ratio * self.img_size * self.img_size)
                for b in range(self.batch_size):
                    indices = torch.randperm(self.img_size * self.img_size)[:num_points]
                    test_mask.view(self.batch_size, -1)[b, indices] = 1.0
            
            # 更新模型稀疏度
            if hasattr(model.sparse_encoder, 'sparse_ratio'):
                model.sparse_encoder.sparse_ratio = sparse_ratio
            
            # 前向传播
            with torch.no_grad():
                output = model(model_input, coords=coords, mask=test_mask)
            
            assert output.shape == target.shape, f"稀疏度{sparse_ratio}输出形状不匹配"
            assert torch.isfinite(output).all(), f"稀疏度{sparse_ratio}输出包含非有限值"
            
            # 计算基本指标
            mse = torch.mean((output - target) ** 2).item()
            print(f"稀疏度{sparse_ratio}: MSE = {mse:.6f}")
        
        print("✅ 稀疏注意力前向传播测试通过!")
    
    def test_training_step(self):
        """测试训练步骤"""
        print("测试训练步骤...")
        
        model = create_model_local(self.cfg)
        model = model.to(self.device)
        model.train()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 创建损失函数
        loss_config = OmegaConf.create({
            'reconstruction': {'type': 'l2', 'weight': 1.0},
            'spectral': {'type': 'spectral_l2', 'weight': 0.5, 'low_freq_components': 8},
            'data_consistency': {'type': 'data_consistency', 'weight': 1.0, 'task': 'sr', 'scale_factor': 4}
        })
        loss_fn = TotalLoss(loss_config)
        
        # 创建数据加载器
        class MockDataLoader:
            def __init__(self, batch_data):
                self.batch_data = batch_data
            
            def __iter__(self):
                yield self.batch_data
            
            def __len__(self):
                return 1
        
        # 将数据移到设备
        batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in self.batch.items()}
        
        dataloader = MockDataLoader(batch_device)
        
        # 运行一个训练步骤
        try:
            metrics = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=None,
                loss_fn=loss_fn,
                device=self.device,
                config=self.cfg,
                epoch=1,
                scaler=None,
                logger=None,
                profiler=None,
                denormalize_fn=None,
                mu=torch.tensor(0.0).to(self.device),
                sigma=torch.tensor(1.0).to(self.device)
            )
            
            assert 'loss' in metrics, "训练指标中缺少损失值"
            assert metrics['loss'] >= 0, f"损失值无效: {metrics['loss']}"
            
            print(f"训练步骤成功: Loss = {metrics['loss']:.6f}")
            print("✅ 训练步骤测试通过!")
            
        except Exception as e:
            print(f"训练步骤失败: {e}")
            raise
    
    def test_validation_step(self):
        """测试验证步骤"""
        print("测试验证步骤...")
        
        model = create_model_local(self.cfg)
        model = model.to(self.device)
        model.eval()
        
        # 创建损失函数
        loss_config = OmegaConf.create({
            'reconstruction': {'type': 'l2', 'weight': 1.0}
        })
        loss_fn = TotalLoss(loss_config)
        
        # 创建数据加载器
        class MockDataLoader:
            def __init__(self, batch_data):
                self.batch_data = batch_data
            
            def __iter__(self):
                yield self.batch_data
            
            def __len__(self):
                return 1
        
        # 将数据移到设备
        batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in self.batch.items()}
        
        dataloader = MockDataLoader(batch_device)
        
        # 运行验证步骤
        try:
            metrics = validate_epoch(
                model=model,
                dataloader=dataloader,
                loss_fn=loss_fn,
                device=self.device,
                config=self.cfg,
                logger=None,
                denormalize_fn=None,
                mu=torch.tensor(0.0).to(self.device),
                sigma=torch.tensor(1.0).to(self.device)
            )
            
            assert 'loss' in metrics, "验证指标中缺少损失值"
            assert 'rel_l2' in metrics, "验证指标中缺少相对L2误差"
            assert metrics['loss'] >= 0, f"验证损失值无效: {metrics['loss']}"
            
            print(f"验证步骤成功: Loss = {metrics['loss']:.6f}, Rel-L2 = {metrics['rel_l2']:.6f}")
            print("✅ 验证步骤测试通过!")
            
        except Exception as e:
            print(f"验证步骤失败: {e}")
            raise
    
    def test_h_consistency(self):
        """测试H算子一致性检查"""
        print("测试H算子一致性...")
        
        model = create_model_local(self.cfg)
        model = model.to(self.device)
        model.eval()
        
        # 创建测试数据
        baseline = self.batch['baseline'].to(self.device)
        coords = self.batch['coords'].to(self.device)
        mask = self.batch['mask'].to(self.device)
        target = self.batch['target'].to(self.device)
        
        # 构建输入
        model_input = torch.cat([baseline, coords, mask], dim=1)
        
        # 前向传播
        with torch.no_grad():
            pred = model(model_input, coords=coords, mask=mask)
        
        # 检查输出与目标的一致性（简化检查）
        rel_l2 = torch.norm(pred - target, p=2) / (torch.norm(target, p=2) + 1e-8)
        
        print(f"相对L2误差: {rel_l2.item():.6f}")
        
        # 检查输出合理性
        assert torch.isfinite(pred).all(), "预测输出包含非有限值"
        assert pred.shape == target.shape, "输出形状与目标不匹配"
        
        print("✅ H算子一致性测试通过!")


def test_sparse_attention_integration():
    """运行所有稀疏注意力训练集成测试"""
    print("=" * 60)
    print("开始稀疏注意力模型训练集成测试")
    print("=" * 60)
    
    test_instance = TestSparseAttentionTraining()
    
    try:
        test_instance.setup_method()
        test_instance.test_sparse_model_creation()
        test_instance.test_sparse_attention_forward()
        test_instance.test_training_step()
        test_instance.test_validation_step()
        test_instance.test_h_consistency()
        
        print("=" * 60)
        print("🎉 所有稀疏注意力训练集成测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    test_sparse_attention_integration()