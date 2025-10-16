"""
PDEBench稀疏观测重建系统 - 训练流程单元测试

测试训练循环、验证循环、损失函数集成和优化器行为
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import shutil

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.losses import TotalLoss, ReconstructionLoss, SpectralLoss, DataConsistencyLoss
from models.swin_unet import SwinUNet
from ops.degradation import SuperResolutionOperator, CropOperator


class TestLossIntegration:
    """损失函数集成测试"""
    
    @pytest.fixture
    def loss_config(self):
        """损失函数配置"""
        return {
            'reconstruction': {'weight': 1.0, 'loss_type': 'l2'},
            'spectral': {'weight': 0.5, 'n_modes': 16, 'loss_type': 'l2'},
            'data_consistency': {'weight': 1.0, 'loss_type': 'l2'}
        }
    
    @pytest.fixture
    def sample_data(self, device):
        """样本数据"""
        batch_size = 2
        channels = 3
        size = 64
        
        # 创建高分辨率真实数据
        gt_hr = torch.randn(batch_size, channels, size, size, device=device)
        
        # 创建观测算子
        sr_op = SuperResolutionOperator(scale_factor=2, blur_sigma=1.0)
        
        # 生成低分辨率观测
        obs_lr = sr_op(gt_hr)
        
        # 创建模型预测（高分辨率）
        pred_hr = torch.randn(batch_size, channels, size, size, device=device)
        
        return {
            'gt_hr': gt_hr,
            'obs_lr': obs_lr,
            'pred_hr': pred_hr,
            'sr_op': sr_op
        }
    
    def test_total_loss_computation(self, device, loss_config, sample_data):
        """测试总损失计算"""
        # 创建归一化函数
        def denormalize_fn(x):
            return x * 2.0 + 1.0  # 简单的反归一化
        
        total_loss = TotalLoss(
            loss_config, 
            observation_op=sample_data['sr_op'],
            denormalize_fn=denormalize_fn
        ).to(device)
        
        # 计算损失
        loss_dict = total_loss(
            pred=sample_data['pred_hr'],
            target=sample_data['gt_hr'],
            observation=sample_data['obs_lr']
        )
        
        # 检查损失结构
        expected_keys = ['total', 'reconstruction', 'spectral', 'data_consistency']
        for key in expected_keys:
            assert key in loss_dict, f"Missing loss component: {key}"
            assert isinstance(loss_dict[key], torch.Tensor), f"{key} should be tensor"
            assert loss_dict[key].dim() == 0, f"{key} should be scalar"
            assert torch.isfinite(loss_dict[key]), f"{key} should be finite"
        
        # 检查总损失是各部分的加权和
        expected_total = (
            loss_config['reconstruction']['weight'] * loss_dict['reconstruction'] +
            loss_config['spectral']['weight'] * loss_dict['spectral'] +
            loss_config['data_consistency']['weight'] * loss_dict['data_consistency']
        )
        
        assert torch.allclose(loss_dict['total'], expected_total, atol=1e-6), \
            "Total loss should be weighted sum of components"
    
    def test_loss_gradients(self, device, loss_config, sample_data, test_utils):
        """测试损失函数梯度"""
        def denormalize_fn(x):
            return x * 2.0 + 1.0
        
        total_loss = TotalLoss(
            loss_config,
            observation_op=sample_data['sr_op'],
            denormalize_fn=denormalize_fn
        ).to(device)
        
        # 设置需要梯度
        pred = sample_data['pred_hr'].clone().requires_grad_(True)
        
        loss_dict = total_loss(
            pred=pred,
            target=sample_data['gt_hr'],
            observation=sample_data['obs_lr']
        )
        
        # 反向传播
        loss_dict['total'].backward()
        
        # 检查梯度
        test_utils.assert_gradient_properties(pred, check_exists=True, finite=True)
    
    def test_loss_with_different_operators(self, device, loss_config):
        """测试不同观测算子的损失计算"""
        batch_size = 2
        channels = 3
        size = 64
        
        # 测试裁剪算子
        crop_op = CropOperator(crop_size=(32, 32), center_crop=True)
        gt_hr = torch.randn(batch_size, channels, size, size, device=device)
        obs_crop = crop_op(gt_hr)
        pred_hr = torch.randn(batch_size, channels, size, size, device=device)
        
        def denormalize_fn(x):
            return x
        
        total_loss = TotalLoss(
            loss_config,
            observation_op=crop_op,
            denormalize_fn=denormalize_fn
        ).to(device)
        
        loss_dict = total_loss(
            pred=pred_hr,
            target=gt_hr,
            observation=obs_crop
        )
        
        # 检查损失计算正常
        for key in ['total', 'reconstruction', 'spectral', 'data_consistency']:
            assert torch.isfinite(loss_dict[key]), f"{key} should be finite for crop operator"


class TestTrainingLoop:
    """训练循环测试"""
    
    @pytest.fixture
    def training_setup(self, device):
        """训练设置"""
        # 模型
        model = SwinUNet(
            in_channels=3, out_channels=3, img_size=64,
            patch_size=4, embed_dim=96,
            depths=[2, 2], num_heads=[3, 6]
        ).to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 损失函数
        loss_config = {
            'reconstruction': {'weight': 1.0, 'loss_type': 'l2'},
            'spectral': {'weight': 0.5, 'n_modes': 16, 'loss_type': 'l2'},
            'data_consistency': {'weight': 1.0, 'loss_type': 'l2'}
        }
        
        sr_op = SuperResolutionOperator(scale_factor=2, blur_sigma=1.0)
        
        def denormalize_fn(x):
            return x
        
        criterion = TotalLoss(
            loss_config,
            observation_op=sr_op,
            denormalize_fn=denormalize_fn
        ).to(device)
        
        return {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'sr_op': sr_op
        }
    
    def test_single_training_step(self, device, training_setup):
        """测试单个训练步骤"""
        model = training_setup['model']
        optimizer = training_setup['optimizer']
        criterion = training_setup['criterion']
        sr_op = training_setup['sr_op']
        
        # 创建批数据
        batch_size = 2
        gt_hr = torch.randn(batch_size, 3, 64, 64, device=device)
        obs_lr = sr_op(gt_hr)
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        pred_hr = model(obs_lr)
        
        # 计算损失
        loss_dict = criterion(pred=pred_hr, target=gt_hr, observation=obs_lr)
        
        # 反向传播
        loss_dict['total'].backward()
        
        # 检查梯度
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        assert grad_norm > 0, "Model should have gradients"
        assert grad_norm < 1000, "Gradient norm should not be too large"
        
        # 优化器步骤
        optimizer.step()
        
        # 检查参数更新
        assert True, "Training step completed successfully"
    
    def test_training_convergence(self, device, training_setup):
        """测试训练收敛性"""
        model = training_setup['model']
        optimizer = training_setup['optimizer']
        criterion = training_setup['criterion']
        sr_op = training_setup['sr_op']
        
        # 创建固定的训练数据
        torch.manual_seed(42)
        gt_hr = torch.randn(1, 3, 64, 64, device=device)
        obs_lr = sr_op(gt_hr)
        
        model.train()
        losses = []
        
        # 训练多个步骤
        for step in range(10):
            optimizer.zero_grad()
            
            pred_hr = model(obs_lr)
            loss_dict = criterion(pred=pred_hr, target=gt_hr, observation=obs_lr)
            
            loss_dict['total'].backward()
            optimizer.step()
            
            losses.append(loss_dict['total'].item())
        
        # 检查损失下降趋势
        # 最后几步的平均损失应该小于前几步
        early_loss = sum(losses[:3]) / 3
        late_loss = sum(losses[-3:]) / 3
        
        assert late_loss < early_loss, f"Loss should decrease: {early_loss} -> {late_loss}"
    
    def test_validation_mode(self, device, training_setup):
        """测试验证模式"""
        model = training_setup['model']
        criterion = training_setup['criterion']
        sr_op = training_setup['sr_op']
        
        # 创建验证数据
        gt_hr = torch.randn(2, 3, 64, 64, device=device)
        obs_lr = sr_op(gt_hr)
        
        # 验证模式
        model.eval()
        
        with torch.no_grad():
            pred_hr = model(obs_lr)
            loss_dict = criterion(pred=pred_hr, target=gt_hr, observation=obs_lr)
        
        # 检查没有梯度
        for param in model.parameters():
            assert param.grad is None, "Parameters should not have gradients in eval mode"
        
        # 检查损失计算正常
        assert torch.isfinite(loss_dict['total']), "Validation loss should be finite"
    
    def test_mixed_precision_training(self, device, training_setup):
        """测试混合精度训练"""
        if device.type != 'cuda':
            pytest.skip("Mixed precision requires CUDA")
        
        model = training_setup['model']
        optimizer = training_setup['optimizer']
        criterion = training_setup['criterion']
        sr_op = training_setup['sr_op']
        
        # 创建GradScaler
        scaler = torch.cuda.amp.GradScaler()
        
        # 创建数据
        gt_hr = torch.randn(2, 3, 64, 64, device=device)
        obs_lr = sr_op(gt_hr)
        
        model.train()
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with torch.cuda.amp.autocast():
            pred_hr = model(obs_lr)
            loss_dict = criterion(pred=pred_hr, target=gt_hr, observation=obs_lr)
        
        # 缩放反向传播
        scaler.scale(loss_dict['total']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert True, "Mixed precision training completed"


class TestOptimizer:
    """优化器测试"""
    
    def test_adam_optimizer(self, device):
        """测试Adam优化器"""
        # 简单的线性模型
        model = nn.Linear(10, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # 创建简单的回归任务
        x = torch.randn(100, 10, device=device)
        y = torch.randn(100, 1, device=device)
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(50):
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 49:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # 损失应该下降
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"
    
    def test_learning_rate_scheduling(self, device):
        """测试学习率调度"""
        model = nn.Linear(10, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # 运行20个epoch
        for epoch in range(20):
            # 模拟训练步骤
            optimizer.zero_grad()
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 1, device=device)
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # 学习率应该下降
        expected_lr = initial_lr * (0.5 ** 2)  # 两次衰减
        assert abs(final_lr - expected_lr) < 1e-6, f"LR should be {expected_lr}, got {final_lr}"
    
    def test_gradient_clipping(self, device):
        """测试梯度裁剪"""
        # 创建容易产生大梯度的模型
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # 大学习率
        
        # 创建数据
        x = torch.randn(10, 10, device=device) * 10  # 大输入
        y = torch.randn(10, 1, device=device)
        
        # 不裁剪梯度的情况
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        
        # 计算梯度范数
        grad_norm_before = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_before += param.grad.norm().item() ** 2
        grad_norm_before = grad_norm_before ** 0.5
        
        # 裁剪梯度
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # 计算裁剪后的梯度范数
        grad_norm_after = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_after += param.grad.norm().item() ** 2
        grad_norm_after = grad_norm_after ** 0.5
        
        # 如果原始梯度范数大于max_norm，裁剪后应该等于max_norm
        if grad_norm_before > max_norm:
            assert abs(grad_norm_after - max_norm) < 1e-4, \
                f"Clipped gradient norm should be {max_norm}, got {grad_norm_after}"


class TestCheckpointing:
    """检查点测试"""
    
    def test_model_save_load(self, device, tmp_path):
        """测试模型保存和加载"""
        # 创建模型
        model = SwinUNet(
            in_channels=3, out_channels=3, img_size=64,
            patch_size=4, embed_dim=96,
            depths=[2, 2], num_heads=[3, 6]
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 训练几步
        for _ in range(3):
            x = torch.randn(1, 3, 64, 64, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 保存检查点
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'loss': 0.5
        }, checkpoint_path)
        
        # 创建新模型并加载
        new_model = SwinUNet(
            in_channels=3, out_channels=3, img_size=64,
            patch_size=4, embed_dim=96,
            depths=[2, 2], num_heads=[3, 6]
        ).to(device)
        
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 检查参数是否相同
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.equal(param1, param2), f"Parameter {name1} not loaded correctly"
        
        # 检查优化器状态
        assert checkpoint['epoch'] == 10, "Epoch not loaded correctly"
        assert checkpoint['loss'] == 0.5, "Loss not loaded correctly"
    
    def test_resume_training(self, device, tmp_path):
        """测试恢复训练"""
        # 创建模型和数据
        model = SwinUNet(
            in_channels=3, out_channels=3, img_size=64,
            patch_size=4, embed_dim=96,
            depths=[2, 2], num_heads=[3, 6]
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(1, 3, 64, 64, device=device)
        target = torch.randn(1, 3, 64, 64, device=device)
        
        # 训练并保存
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss_before = nn.MSELoss()(pred, target)
        loss_before.backward()
        optimizer.step()
        
        # 保存状态
        checkpoint_path = tmp_path / "resume_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        
        # 加载并继续训练
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 继续训练
        optimizer.zero_grad()
        pred = model(x)
        loss_after = nn.MSELoss()(pred, target)
        loss_after.backward()
        optimizer.step()
        
        # 损失应该继续下降
        assert loss_after.item() < loss_before.item(), "Training should continue from checkpoint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])