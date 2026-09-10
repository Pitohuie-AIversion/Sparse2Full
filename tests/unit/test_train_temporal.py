"""时序训练器 (TemporalTrainer) 架构解耦与接口回归单元测试

验证 `train_temporal.py` 完全对齐 `train.py` 解耦架构规范：
1. 核心解耦子系统（BatchProcessor, CurriculumScheduler, TrainingArtifactManager）装配
2. 数据流、模型创建、优化器与调度器构建
3. 训练与评估生命周期向后兼容性
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from train_temporal import TemporalTrainer


class TestTemporalTrainerArchitecture:
    """TemporalTrainer 解耦架构测试集"""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        return OmegaConf.create({
            "experiment": {
                "name": "test_temporal_exp",
                "device": "cpu",
                "seed": 42,
                "output_dir": str(tmp_path / "runs"),
                "log_every_n_steps": 10,
                "val_check_interval": 1,
                "early_stopping": {"patience": 5}
            },
            "train": {
                "max_epochs": 2,
                "gradient_clip_val": 1.0,
                "optimizer": {
                    "name": "AdamW",
                    "lr": 1e-3,
                    "weight_decay": 1e-4
                },
                "scheduler": {
                    "name": "cosine",
                    "params": {"T_max": 2, "eta_min": 1e-6}
                }
            },
            "model": {
                "name": "SwinTemporalNAR",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "img_size": 32
                }
            },
            "temporal": {
                "T_in": 10,
                "T_out": 5,
                "ar": {"T_in": 10, "T_out": 5}
            },
            "loss": {
                "ar_loss": {"weight": 1.0},
                "spectral_loss": {"weight": 0.1},
                "dc_loss": {"weight": 0.0}
            },
            "data": {
                "data_path": "dummy.h5"
            }
        })

    def test_trainer_initialization_architecture(self, minimal_config, monkeypatch):
        """测试按标准架构初始化核心子系统"""
        # Mock 数据模块以避免真实加载不存在的 dummy.h5
        mock_dm = MagicMock()
        mock_dm.train_dataloader.return_value = [None] * 5
        mock_dm.val_dataloader.return_value = [None] * 2
        mock_dm.test_dataloader.return_value = [None] * 2
        monkeypatch.setattr("train_temporal.TemporalPDEBenchDataModule", lambda cfg: mock_dm)

        # Mock 模型创建
        dummy_model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1))
        monkeypatch.setattr("train_temporal.registry_create_model", lambda name, **kw: dummy_model)

        trainer = TemporalTrainer(minimal_config)

        # 1. 验证核心解耦子系统实例存在
        assert hasattr(trainer, "batch_processor")
        assert hasattr(trainer, "curriculum_scheduler")
        assert hasattr(trainer, "artifact_manager")
        assert trainer.artifact_manager.output_dir.exists()

        # 2. 验证模型与数据对齐
        assert trainer.model is dummy_model
        assert len(trainer.train_loader) == 5

        # 3. 验证优化器与调度器统一由 EngineBuilder 成功装配
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        # 4. 验证检查点持久化代理
        trainer.save_checkpoint(is_best=True)
        assert (trainer.output_dir / "best.ckpt").exists()
        assert (trainer.output_dir / "last.ckpt").exists()

    def test_forward_model_and_metrics_flow(self, minimal_config, monkeypatch):
        """测试 5D/4D 前向传播与轻量评估指标"""
        mock_dm = MagicMock()
        mock_dm.train_dataloader.return_value = []
        mock_dm.val_dataloader.return_value = []
        mock_dm.test_dataloader.return_value = []
        monkeypatch.setattr("train_temporal.TemporalPDEBenchDataModule", lambda cfg: mock_dm)

        # 构造简单卷积模型以测试 5D 前向与回退
        class DummyARModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                if x.ndim == 5:
                    B, T, C, H, W = x.shape
                    out = self.net(x.reshape(B * T, C, H, W))
                    return out.reshape(B, T, C, H, W)
                return self.net(x)

        dummy_model = DummyARModel()
        monkeypatch.setattr("train_temporal.registry_create_model", lambda name, **kw: dummy_model)

        trainer = TemporalTrainer(minimal_config)

        # 测试 5D 前向
        x5d = torch.randn(2, 5, 1, 32, 32)
        out5d = trainer._forward_model(x5d)
        assert out5d.shape == (2, 5, 1, 32, 32)

        # 测试指标计算 (尺寸不匹配自动对齐)
        pred = torch.randn(2, 5, 1, 32, 32)
        target = torch.randn(2, 5, 1, 64, 64)
        rel_l2, mae = trainer._compute_light_metrics(pred, target)
        assert isinstance(rel_l2, float)
        assert isinstance(mae, float)
        assert rel_l2 >= 0.0
        assert mae >= 0.0
