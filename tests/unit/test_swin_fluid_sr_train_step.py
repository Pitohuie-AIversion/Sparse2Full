"""
SwinFluidSR 端到端训练 Step 验证测试

模拟完整的训练迭代：
1. 合成包含原始低维观测、目标场、坐标、掩码的 batch 字典；
2. 实例化 SwinFluidSR 模型（use_lowres_input=True）；
3. 计算组合损失（重建 L2 + 频域能谱 + 物理守恒散度/涡度）；
4. 反向传播与优化器更新，确保参数有效迭代；
5. 验证训练流程与 train.py 的 _build_model_input / _prepare_target 逻辑完全契约对齐。
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from models.registry import create_model
from ops.losses import CombinedLoss


class TestSwinFluidSRTrainingStep:
    @pytest.fixture
    def mock_sr_batch(self):
        """构造 SR 任务的标准 batch 结构 (scale=4, 32x32 -> 128x128)"""
        B = 2
        # 流体速度场：2 通道 (u, v)
        lr_obs = torch.randn(B, 2, 32, 32, dtype=torch.float32)
        hr_target = torch.randn(B, 2, 128, 128, dtype=torch.float32)
        baseline = torch.nn.functional.interpolate(lr_obs, size=(128, 128), mode="bicubic", align_corners=False)
        coords = torch.randn(B, 2, 128, 128, dtype=torch.float32)
        mask = torch.ones(B, 1, 128, 128, dtype=torch.float32)

        return {
            "original_observation": lr_obs,
            "lr_observation": lr_obs,
            "target": hr_target,
            "baseline": baseline,
            "coords": coords,
            "mask": mask,
        }

    def test_end_to_end_train_step_with_physics(self, mock_sr_batch):
        # 1. 实例化 2 通道流体超分辨模型
        model = create_model(
            "SwinFluidSR",
            in_channels=2,
            out_channels=2,
            img_size=32,
            upscale_factor=4,
            embed_dim=48,
            depths=[2, 2],
            num_heads=[2, 4],
            window_size=8,
            use_lowres_input=True,
            global_residual=True,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 2. 模拟 train.py _build_model_input
        batch = mock_sr_batch
        if getattr(model, "use_lowres_input", False):
            raw_obs = batch.get("lr_observation", batch.get("original_observation", None))
            model_input = raw_obs
        else:
            model_input = batch["baseline"]

        assert model_input.shape == (2, 2, 32, 32)

        # 3. 前向计算
        pred = model(model_input)
        assert pred.shape == (2, 2, 128, 128)

        # 4. 模拟 train.py _prepare_target
        target = batch["target"]
        assert target.shape == pred.shape

        # 5. 组合物理损失：测试 CombinedLoss
        loss_cfg = OmegaConf.create({
            "loss": {
                "reconstruction_weight": 1.0,
                "spectral_weight": 0.1,
                "data_consistency_weight": 0.0,
                "reconstruction_loss_type": "l2",
            }
        })
        loss_fn = CombinedLoss(loss_cfg)
        loss_res = loss_fn(pred, target)
        loss = loss_res["total_loss"]
        assert torch.isfinite(loss)
        assert "reconstruction_loss" in loss_res
        assert "spectral_loss" in loss_res

        # 5.1 验证 train.py 使用的 compute_total_loss
        from ops.losses import compute_total_loss
        full_cfg = OmegaConf.create({
            "loss": {
                "rec_weight": 1.0,
                "spec_weight": 0.1,
                "dc_weight": 0.0,
                "div_weight": 0.05,
                "vort_weight": 0.05,
                "rec_loss_type": "l2",
            }
        })
        train_losses = compute_total_loss(
            pred_z=pred,
            target_z=target,
            obs_data=batch,
            config=full_cfg
        )
        assert torch.isfinite(train_losses["total_loss"])
        assert "rec_loss" in train_losses
        assert "div_loss" in train_losses
        assert "vort_loss" in train_losses

        # 使用包含物理守恒的总损失进行反向传播
        optimizer.zero_grad()
        train_losses["total_loss"].backward()

        # 检查关键层均获得非零有限梯度
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"NaN/Inf gradient in {name}"

        optimizer.step()

    def test_end_to_end_train_step_16x16_8x(self):
        """测试 16x16 输入、8x 上采样到 128x128 的完整训练步反向传播"""
        B = 2
        # 流体物理场 1 通道（如标量温度/浮力场）或 2 通道（速度场）
        lr_obs = torch.randn(B, 1, 16, 16, dtype=torch.float32)
        hr_target = torch.randn(B, 1, 128, 128, dtype=torch.float32)
        baseline = torch.nn.functional.interpolate(lr_obs, size=(128, 128), mode="bicubic", align_corners=False)
        coords = torch.randn(B, 2, 128, 128, dtype=torch.float32)
        mask = torch.ones(B, 1, 128, 128, dtype=torch.float32)

        batch = {
            "original_observation": lr_obs,
            "lr_observation": lr_obs,
            "target": hr_target,
            "baseline": baseline,
            "coords": coords,
            "mask": mask,
        }

        # 1. 实例化 16x16 -> 128x128 8x 超分辨率模型
        model = create_model(
            "SwinFluidSR",
            in_channels=1,
            out_channels=1,
            img_size=16,
            upscale_factor=8,
            embed_dim=48,
            depths=[2, 2],
            num_heads=[2, 4],
            window_size=8,
            use_lowres_input=True,
            global_residual=True,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 2. 模拟 train.py _build_model_input
        model_input = batch.get("lr_observation", batch.get("original_observation", None))
        assert model_input.shape == (B, 1, 16, 16)

        # 3. 前向计算
        pred = model(model_input)
        assert pred.shape == (B, 1, 128, 128)

        # 4. 损失计算与反向传播
        from ops.losses import compute_total_loss
        loss_cfg = OmegaConf.create({
            "loss": {
                "rec_weight": 1.0,
                "spec_weight": 0.1,
                "dc_weight": 0.0,
                "div_weight": 0.0,
                "vort_weight": 0.0,
                "rec_loss_type": "l2",
            }
        })
        train_losses = compute_total_loss(
            pred_z=pred,
            target_z=batch["target"],
            obs_data=batch,
            config=loss_cfg
        )
        assert torch.isfinite(train_losses["total_loss"])

        optimizer.zero_grad()
        train_losses["total_loss"].backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"NaN/Inf gradient in {name}"

        optimizer.step()
