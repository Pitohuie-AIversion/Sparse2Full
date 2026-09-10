"""BatchProcessor 专项单元测试

覆盖输入张量构建、时序切片、坐标掩码追加、通道对齐与 Fail-Fast 守恒校验。
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from training.batch_processor import BatchProcessor


class DummyModel(nn.Module):
    def __init__(self, in_channels: int = 4, use_lowres: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.use_lowres_input = use_lowres
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestBatchProcessor:
    """测试 BatchProcessor 核心逻辑"""

    def test_build_model_input_standard_coords_mask(self):
        """测试标准模式下 baseline + coords + mask 的拼接"""
        cfg = OmegaConf.create({
            "model": {"in_channels": 4},
            "training": {"allow_channel_mismatch": True}
        })
        processor = BatchProcessor(cfg)
        model = DummyModel(in_channels=4, use_lowres=False)

        B, H, W = 2, 64, 64
        batch = {
            "baseline": torch.randn(B, 1, H, W),
            "coords": torch.randn(B, 2, H, W),
            "mask": torch.ones(B, 1, H, W)
        }

        inp = processor.build_model_input(batch, model)
        assert inp.shape == (B, 4, H, W)
        assert torch.allclose(inp[:, 0:1], batch["baseline"])
        assert torch.allclose(inp[:, 1:3], batch["coords"])
        assert torch.allclose(inp[:, 3:4], batch["mask"])

    def test_build_model_input_direct_lowres(self):
        """测试使用 direct low-res 输入（如 32x32）"""
        cfg = OmegaConf.create({
            "model": {"in_channels": 1},
            "training": {"allow_channel_mismatch": True}
        })
        processor = BatchProcessor(cfg)
        model = DummyModel(in_channels=1, use_lowres=True)

        B = 2
        batch = {
            "baseline": torch.randn(B, 1, 128, 128),  # 插值后的高分辨率
            "lr_observation": torch.randn(B, 1, 32, 32),  # 原始低分辨率
        }

        inp = processor.build_model_input(batch, model)
        assert inp.shape == (B, 1, 32, 32)
        assert torch.allclose(inp, batch["lr_observation"])

    def test_build_model_input_channel_padding_and_trimming(self):
        """测试通道自动填充与裁剪"""
        cfg = OmegaConf.create({
            "model": {"in_channels": 4},
            "training": {"allow_channel_mismatch": True}
        })
        processor = BatchProcessor(cfg)
        model = DummyModel(in_channels=4)

        # 1. 实际 2 通道，期望 4 通道 -> 自动补零
        batch_fewer = {"baseline": torch.randn(2, 2, 32, 32)}
        inp_padded = processor.build_model_input(batch_fewer, model)
        assert inp_padded.shape == (2, 4, 32, 32)
        assert torch.all(inp_padded[:, 2:] == 0.0)

        # 2. 实际 6 通道，期望 4 通道 -> 自动截断
        batch_more = {"baseline": torch.randn(2, 6, 32, 32)}
        inp_trimmed = processor.build_model_input(batch_more, model)
        assert inp_trimmed.shape == (2, 4, 32, 32)

    def test_build_model_input_fail_fast_on_channel_mismatch(self):
        """测试在禁止通道不匹配时严格拦截抛出 ValueError"""
        cfg = OmegaConf.create({
            "model": {"in_channels": 4},
            "training": {"allow_channel_mismatch": False}
        })
        processor = BatchProcessor(cfg)
        model = DummyModel(in_channels=4)

        batch_fewer = {"baseline": torch.randn(2, 2, 32, 32)}
        with pytest.raises(ValueError, match="Model input channel mismatch"):
            processor.build_model_input(batch_fewer, model)

    def test_prepare_target_temporal_slice(self):
        """测试目标张量时序切片"""
        processor = BatchProcessor()
        # [B, T, C, H, W]
        target_seq = torch.randn(2, 5, 1, 64, 64)
        pred_shape = (2, 1, 64, 64)

        target = processor.prepare_target(target_seq, pred_shape)
        assert target.shape == (2, 1, 64, 64)
        assert torch.allclose(target, target_seq[:, -1])

    def test_prepare_target_fail_fast_resampling(self):
        """测试空间尺寸不匹配时 Fail-Fast 严格拦截（杜绝静默重采样真值）"""
        cfg = OmegaConf.create({
            "training": {"allow_target_resampling": False}
        })
        processor = BatchProcessor(cfg)

        target = torch.randn(2, 1, 128, 128)
        pred_shape = (2, 1, 64, 64)  # 预测尺寸不符

        with pytest.raises(ValueError, match="Physical field spatial dimension mismatch"):
            processor.prepare_target(target, pred_shape)

    def test_prepare_target_allow_resampling_flag(self):
        """测试当显式开启 allow_target_resampling 时的双线性插值行为"""
        cfg = OmegaConf.create({
            "training": {"allow_target_resampling": True}
        })
        processor = BatchProcessor(cfg)

        target = torch.randn(2, 1, 64, 64)
        pred_shape = (2, 1, 128, 128)

        res = processor.prepare_target(target, pred_shape)
        assert res.shape == (2, 1, 128, 128)
