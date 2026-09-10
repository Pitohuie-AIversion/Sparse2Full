"""EngineBuilder 专项单元测试

覆盖优化器、调度器、Warmup 与 AMP GradScaler 的构建与安全检测。
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from training.engine_builder import EngineBuilder


class TestEngineBuilder:
    """测试 EngineBuilder 各组件构建"""

    def test_build_optimizer_types(self):
        """测试不同优化器类型构建"""
        model = nn.Linear(10, 2)

        # AdamW
        cfg_adamw = OmegaConf.create({
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 1e-4, "weight_decay": 1e-3}
            }
        })
        opt_adamw = EngineBuilder.build_optimizer(model, cfg_adamw)
        assert isinstance(opt_adamw, torch.optim.AdamW)
        assert opt_adamw.defaults["lr"] == 1e-4
        assert opt_adamw.defaults["weight_decay"] == 1e-3

        # Adam
        cfg_adam = OmegaConf.create({
            "optimizer": {
                "name": "Adam",
                "params": {"lr": 2e-4}
            }
        })
        opt_adam = EngineBuilder.build_optimizer(model, cfg_adam)
        assert isinstance(opt_adam, torch.optim.Adam)

        # SGD
        cfg_sgd = OmegaConf.create({
            "optimizer": {
                "name": "SGD",
                "params": {"lr": 1e-2, "momentum": 0.9}
            }
        })
        opt_sgd = EngineBuilder.build_optimizer(model, cfg_sgd)
        assert isinstance(opt_sgd, torch.optim.SGD)
        assert opt_sgd.defaults["momentum"] == 0.9

    def test_build_scheduler_cosine_and_warmup(self):
        """测试 CosineAnnealingLR 与 Warmup 组合"""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        fake_loader = [None] * 20  # 20 batches per epoch

        cfg = OmegaConf.create({
            "epochs": 10,
            "scheduler": {
                "name": "cosine_warmup",
                "params": {
                    "warmup_epochs": 2,
                    "warmup_start_factor": 0.05,
                    "eta_min": 1e-6
                }
            }
        })

        sched, warmup = EngineBuilder.build_scheduler(optimizer, cfg, fake_loader)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert sched.T_max == 10 * 20  # 200 steps
        assert sched.eta_min == 1e-6

        assert isinstance(warmup, torch.optim.lr_scheduler.LinearLR)
        assert warmup.total_iters == 2
        assert warmup.start_factor == 0.05

    def test_build_scheduler_plateau_and_step(self):
        """测试 ReduceLROnPlateau 与 StepLR"""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        cfg_step = OmegaConf.create({
            "scheduler": {
                "name": "step",
                "params": {"step_size": 15, "gamma": 0.5}
            }
        })
        sched_step, _ = EngineBuilder.build_scheduler(optimizer, cfg_step)
        assert isinstance(sched_step, torch.optim.lr_scheduler.StepLR)

        cfg_plateau = OmegaConf.create({
            "scheduler": {
                "name": "plateau",
                "params": {"mode": "min", "factor": 0.2, "patience": 5}
            }
        })
        sched_plateau, _ = EngineBuilder.build_scheduler(optimizer, cfg_plateau)
        assert isinstance(sched_plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_build_amp_scaler_safe_disabling_for_fno(self):
        """测试 AMP 在面对复数运算模型（如 FNO2D）时自动禁用"""
        # 标准模型：启用 AMP
        cfg_amp = OmegaConf.create({"use_amp": True})
        scaler_normal, enabled_normal = EngineBuilder.build_amp_scaler(cfg_amp, model_name="SwinUNet")
        assert enabled_normal is True
        assert scaler_normal is not None

        # FNO 模型：应自动检测并禁用
        scaler_fno, enabled_fno = EngineBuilder.build_amp_scaler(cfg_amp, model_name="FNO2D")
        assert enabled_fno is False
        assert scaler_fno is None

        scaler_ufno, enabled_ufno = EngineBuilder.build_amp_scaler(cfg_amp, model_name="UFNO_UNet")
        assert enabled_ufno is False
        assert scaler_ufno is None
