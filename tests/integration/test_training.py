"""
PDEBench稀疏观测重建系统 - 训练流程集成测试

测试完整训练流程的集成性，包括数据加载、模型训练、损失计算、
检查点保存等关键环节的协同工作。
遵循技术架构文档7.7节TDD准则要求。
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import torch
import torch.nn as nn
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

# 导入被测试模块
from train import main as train_main
from datasets.pdebench import PDEBenchDataModule
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.losses import CombinedLoss
from utils.checkpoint import CheckpointManager
from utils.logger import Logger
from utils.metrics import compute_metrics


class TestTrainingIntegration:
    """训练流程集成测试类"""
    
    @pytest.fixture
    def training_config(self, temp_dir):
        """创建训练配置"""
        config = {
            "data": {
                "root_path": str(temp_dir / "data"),
                "keys": ["u"],
                "task": "SR",
                "sr_scale": 2,  # 使用较小的scale加速测试
                "crop_size": [64, 64],
                "normalize": True,
                "batch_size": 2,
                "num_workers": 0,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "model": {
                "name": "swin_unet",
                "patch_size": 4,
                "window_size": 4,
                "embed_dim": 48,
                "depths": [2, 2],
                "num_heads": [3, 6],
                "in_channels": 4,  # baseline + coords + mask
                "out_channels": 1
            },
            "training": {
                "num_epochs": 3,  # 少量epoch用于测试
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "warmup_steps": 10,
                "gradient_clip_val": 1.0,
                "use_amp": False,  # 关闭AMP简化测试
                "save_every": 1,
                "validate_every": 1,
                "log_every": 1
            },
            "loss": {
                "reconstruction_weight": 1.0,
                "spectral_weight": 0.1,
                "data_consistency_weight": 0.5,
                "gradient_weight": 0.0,
                "spectral_loss": {
                    "low_freq_modes": 8,
                    "use_rfft": True,
                    "normalize": True
                }
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999]
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 10,
                "max_steps": 100
            },
            "experiment": {
                "name": "test_training_integration",
                "save_dir": str(temp_dir / "runs"),
                "seed": 42,
                "deterministic": True
            }
        }
        return OmegaConf.create(config)
    
    @pytest.fixture
    def mock_training_data(self, temp_dir):
        """创建模拟训练数据"""
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        # 创建HDF5数据文件
        h5_path = data_dir / "test_data.h5"
        with h5py.File(h5_path, 'w') as f:
            # 创建20个样本用于训练测试
            for i in range(20):
                case_group = f.create_group(str(i))
                # 创建64x64的数据
                u_data = np.random.randn(64, 64).astype(np.float32)
                case_group.create_dataset("u", data=u_data)
        
        # 创建splits目录和文件
        splits_dir = data_dir / "splits"
        splits_dir.mkdir()
        
        # 训练集：0-13 (14个样本)
        with open(splits_dir / "train.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(14)]))
        
        # 验证集：14-16 (3个样本)
        with open(splits_dir / "val.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(14, 17)]))
        
        # 测试集：17-19 (3个样本)
        with open(splits_dir / "test.txt", "w") as f:
            f.write("\n".join([str(i) for i in range(17, 20)]))
        
        return str(h5_path)
    
    def test_data_module_initialization(self, training_config, mock_training_data):
        """测试数据模块初始化"""
        # 更新配置中的数据路径
        training_config.data.root_path = str(Path(mock_training_data).parent)
        
        data_module = PDEBenchDataModule(training_config.data)
        data_module.setup()
        
        # 验证数据加载器创建
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # 验证数据批次形状
        train_batch = next(iter(train_loader))
        assert "input" in train_batch
        assert "target" in train_batch
        
        input_tensor = train_batch["input"]
        target_tensor = train_batch["target"]
        
        assert input_tensor.shape[0] == training_config.data.batch_size
        assert target_tensor.shape[0] == training_config.data.batch_size
        assert input_tensor.shape[1] == 4  # baseline + coords + mask
        assert target_tensor.shape[1] == 1  # single channel output
    
    def test_model_initialization_and_forward(self, training_config):
        """测试模型初始化和前向传播"""
        model_config = training_config.model
        
        # 测试SwinUNet模型
        model = SwinUNet(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            img_size=training_config.data.crop_size[0],
            patch_size=model_config.patch_size,
            window_size=model_config.window_size,
            embed_dim=model_config.embed_dim,
            depths=model_config.depths,
            num_heads=model_config.num_heads
        )
        
        # 验证模型参数
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # 测试前向传播
        batch_size = 2
        input_tensor = torch.randn(
            batch_size, 
            model_config.in_channels, 
            training_config.data.crop_size[0], 
            training_config.data.crop_size[1]
        )
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (
            batch_size, 
            model_config.out_channels, 
            training_config.data.crop_size[0], 
            training_config.data.crop_size[1]
        )
    
    def test_loss_computation(self, training_config):
        """测试损失函数计算"""
        loss_fn = CombinedLoss(training_config)
        
        # 创建模拟数据
        batch_size = 2
        channels = 1
        height, width = 64, 64
        
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randn(batch_size, channels, height, width)
        
        # 创建观测数据
        observation_data = {
            "baseline": torch.randn(batch_size, channels, height, width),
            "coords": torch.randn(batch_size, 2, height, width),
            "mask": torch.ones(batch_size, 1, height, width),
            "degradation_params": {
                "task": "SR",
                "scale": 2,
                "sigma": 1.0,
                "kernel_size": 5,
                "boundary": "mirror"
            }
        }
        
        # 计算损失
        loss_dict = loss_fn(pred, target, observation_data)
        
        # 验证损失字典结构
        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert "reconstruction_loss" in loss_dict
        assert "spectral_loss" in loss_dict
        assert "data_consistency_loss" in loss_dict
        
        # 验证损失值合理性
        total_loss = loss_dict["total_loss"]
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.item() >= 0
    
    def test_checkpoint_manager(self, training_config, temp_dir):
        """测试检查点管理"""
        save_dir = temp_dir / "checkpoints"
        save_dir.mkdir()
        
        checkpoint_manager = CheckpointManager(
            save_dir=str(save_dir),
            max_keep=3
        )
        
        # 创建模拟模型和优化器
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 保存检查点
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={"val_loss": 0.5, "val_rel_l2": 0.1}
        )
        
        # 验证检查点文件存在
        checkpoint_files = list(save_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        
        # 测试加载检查点
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_files[0])
        assert "model_state_dict" in loaded_checkpoint
        assert "optimizer_state_dict" in loaded_checkpoint
        assert "epoch" in loaded_checkpoint
        assert "metrics" in loaded_checkpoint
    
    @pytest.mark.slow
    def test_short_training_run(self, training_config, mock_training_data, temp_dir):
        """测试短期训练运行"""
        # 设置随机种子确保可重现性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 更新配置
        training_config.data.root_path = str(Path(mock_training_data).parent)
        training_config.experiment.save_dir = str(temp_dir / "runs")
        
        # 创建保存目录
        save_dir = Path(training_config.experiment.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据模块
        data_module = PDEBenchDataModule(training_config.data)
        data_module.setup()
        
        # 初始化模型
        model = SwinUNet(
            in_channels=training_config.model.in_channels,
            out_channels=training_config.model.out_channels,
            img_size=training_config.data.crop_size[0],
            patch_size=training_config.model.patch_size,
            window_size=training_config.model.window_size,
            embed_dim=training_config.model.embed_dim,
            depths=training_config.model.depths,
            num_heads=training_config.model.num_heads
        )
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.training.learning_rate,
            weight_decay=training_config.training.weight_decay
        )
        
        # 初始化损失函数
        loss_fn = CombinedLoss(training_config)
        
        # 初始化检查点管理器
        checkpoint_manager = CheckpointManager(
            save_dir=str(save_dir / "checkpoints"),
            max_keep=2
        )
        
        # 训练循环
        model.train()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(training_config.training.num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                observation_data = batch.get("observation_data", {})
                
                pred = model(input_tensor)
                
                # 计算损失
                loss_dict = loss_fn(pred, target_tensor, observation_data)
                total_loss = loss_dict["total_loss"]
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    training_config.training.gradient_clip_val
                )
                
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                
                # 记录初始损失
                if initial_loss is None:
                    initial_loss = total_loss.item()
            
            # 记录最终损失
            final_loss = np.mean(epoch_losses)
            
            # 验证阶段
            if epoch % training_config.training.validate_every == 0:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch["input"]
                        val_target = val_batch["target"]
                        val_observation = val_batch.get("observation_data", {})
                        
                        val_pred = model(val_input)
                        val_loss_dict = loss_fn(val_pred, val_target, val_observation)
                        val_losses.append(val_loss_dict["total_loss"].item())
                
                val_loss = np.mean(val_losses)
                
                # 保存检查点
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"val_loss": val_loss}
                )
                
                model.train()
        
        # 验证训练效果
        assert initial_loss is not None
        assert final_loss is not None
        
        # 验证损失下降（允许一定波动）
        loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
        assert loss_reduction_ratio > -0.5, f"Loss increased too much: {loss_reduction_ratio}"
        
        # 验证检查点文件生成
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
    
    def test_training_reproducibility(self, training_config, mock_training_data, temp_dir):
        """测试训练可重现性"""
        # 更新配置
        training_config.data.root_path = str(Path(mock_training_data).parent)
        training_config.training.num_epochs = 1  # 单个epoch测试
        
        def run_single_epoch():
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 初始化组件
            data_module = PDEBenchDataModule(training_config.data)
            data_module.setup()
            
            model = SwinUNet(
                in_channels=training_config.model.in_channels,
                out_channels=training_config.model.out_channels,
                img_size=training_config.data.crop_size[0],
                patch_size=training_config.model.patch_size,
                window_size=training_config.model.window_size,
                embed_dim=training_config.model.embed_dim,
                depths=training_config.model.depths,
                num_heads=training_config.model.num_heads
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.training.learning_rate,
                weight_decay=training_config.training.weight_decay
            )
            
            loss_fn = CombinedLoss(training_config)
            
            # 训练一个epoch
            model.train()
            train_loader = data_module.train_dataloader()
            
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_tensor = batch["input"]
                target_tensor = batch["target"]
                observation_data = batch.get("observation_data", {})
                
                pred = model(input_tensor)
                loss_dict = loss_fn(pred, target_tensor, observation_data)
                
                loss_dict["total_loss"].backward()
                optimizer.step()
                
                total_loss += loss_dict["total_loss"].item()
            
            return total_loss
        
        # 运行两次相同的训练
        loss1 = run_single_epoch()
        loss2 = run_single_epoch()
        
        # 验证结果一致性（允许极小的数值误差）
        np.testing.assert_allclose(loss1, loss2, rtol=1e-5, atol=1e-8)
    
    def test_training_with_different_models(self, training_config, mock_training_data):
        """测试不同模型的训练兼容性"""
        # 更新配置
        training_config.data.root_path = str(Path(mock_training_data).parent)
        training_config.training.num_epochs = 1
        
        # 测试模型列表
        models_to_test = [
            ("swin_unet", SwinUNet),
            ("hybrid", HybridModel),
            ("mlp", MLPModel)
        ]
        
        for model_name, model_class in models_to_test:
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 初始化数据模块
            data_module = PDEBenchDataModule(training_config.data)
            data_module.setup()
            
            # 根据模型类型调整参数
            if model_name == "swin_unet":
                model = model_class(
                    in_channels=training_config.model.in_channels,
                    out_channels=training_config.model.out_channels,
                    img_size=training_config.data.crop_size[0],
                    patch_size=training_config.model.patch_size,
                    window_size=training_config.model.window_size,
                    embed_dim=training_config.model.embed_dim,
                    depths=training_config.model.depths,
                    num_heads=training_config.model.num_heads
                )
            elif model_name == "hybrid":
                model = model_class(
                    in_channels=training_config.model.in_channels,
                    out_channels=training_config.model.out_channels,
                    img_size=training_config.data.crop_size[0]
                )
            else:  # mlp
                model = model_class(
                    in_channels=training_config.model.in_channels,
                    out_channels=training_config.model.out_channels,
                    img_size=training_config.data.crop_size[0]
                )
            
            # 初始化优化器和损失函数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.training.learning_rate,
                weight_decay=training_config.training.weight_decay
            )
            
            loss_fn = CombinedLoss(training_config)
            
            # 训练一个batch
            model.train()
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            
            optimizer.zero_grad()
            
            input_tensor = batch["input"]
            target_tensor = batch["target"]
            observation_data = batch.get("observation_data", {})
            
            pred = model(input_tensor)
            loss_dict = loss_fn(pred, target_tensor, observation_data)
            
            # 验证输出形状正确
            assert pred.shape == target_tensor.shape
            
            # 验证损失计算成功
            assert "total_loss" in loss_dict
            assert loss_dict["total_loss"].requires_grad
            
            # 验证反向传播成功
            loss_dict["total_loss"].backward()
            
            # 验证梯度存在
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in model.parameters()
            )
            assert has_gradients, f"No gradients found for {model_name}"
            
            optimizer.step()