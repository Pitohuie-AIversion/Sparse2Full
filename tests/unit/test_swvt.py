"""SWVT (Video Swin Transformer) 单元测试

验证：
1. 模型注册与工厂函数实例化
2. 5D 全场张量输入与单步/多步前向输出形状
3. 梯度反向传播有效性
4. 教师强制比率与 AR 滚动推演逻辑
"""

import pytest
import torch
from models.registry import create_model, resolve_model_name
from models.temporal.components.video_swin import VideoSwinPredictor


class TestSWVT:
    """SWVT 模型测试用例"""

    def test_registry_resolution(self):
        """测试模型注册表别名解析"""
        for name in ["SWVT", "VideoSwin", "video_swin", "swvt", "VideoSwinPredictor"]:
            resolved = resolve_model_name(name)
            assert resolved == "SWVT", f"Alias {name} resolved to {resolved}"

    def test_model_instantiation(self):
        """测试模型实例化与基础属性"""
        model = create_model(
            "SWVT",
            in_channels=1,
            out_channels=1,
            hidden_dim=48,
            num_layers=2,
            window_size=(2, 8, 8)
        )
        assert isinstance(model, VideoSwinPredictor)
        info = model.get_model_info()
        assert info["model_type"] == "SWVT"
        assert info["in_channels"] == 1
        assert info["out_channels"] == 1
        assert info["total_parameters"] > 0

    def test_forward_single_step(self):
        """测试单步时空推演输出形状 (T_out=1)"""
        model = VideoSwinPredictor(
            in_channels=1,
            out_channels=1,
            hidden_dim=32,
            num_layers=2,
            window_size=(2, 8, 8)
        )
        # 输入 5D: [B, T_in, C, H, W]
        x = torch.randn(2, 1, 1, 32, 32)
        out = model(x, T_out=1)
        assert out.shape == (2, 1, 1, 32, 32)

    def test_forward_multi_step_rollout(self):
        """测试多步自回归滚动推演 (T_in=2, T_out=4)"""
        model = VideoSwinPredictor(
            in_channels=2,
            out_channels=2,
            hidden_dim=32,
            num_layers=2,
            window_size=(2, 8, 8)
        )
        x = torch.randn(2, 2, 2, 32, 32)
        teacher = torch.randn(2, 4, 2, 32, 32)
        out = model(x, T_out=4, teacher_seq=teacher, teacher_forcing_ratio=0.5)
        assert out.shape == (2, 4, 2, 32, 32)

    def test_gradient_flow(self):
        """测试梯度反向传播完整性"""
        model = VideoSwinPredictor(
            in_channels=1,
            out_channels=1,
            hidden_dim=32,
            num_layers=2,
            window_size=(2, 8, 8)
        )
        x = torch.randn(2, 1, 1, 32, 32)
        out = model(x, T_out=2)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient is NaN"

    def test_forward_with_checkpointing(self):
        """测试开启激活检查点（Activation Checkpointing）时的反向传播有效性"""
        model = VideoSwinPredictor(
            in_channels=1,
            out_channels=1,
            hidden_dim=32,
            num_layers=2,
            window_size=(2, 8, 8),
            use_checkpoint=True
        )
        model.train()
        x = torch.randn(2, 2, 1, 32, 32)
        out = model(x, T_out=3)
        assert out.shape == (2, 3, 1, 32, 32)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no grad with checkpointing"

    def test_unequal_channels_with_teacher_forcing(self):
        """测试输入输出通道不同时，教师强制反馈映射机制正确性"""
        model = VideoSwinPredictor(
            in_channels=1,
            out_channels=2,
            hidden_dim=32,
            num_layers=2,
            window_size=(2, 8, 8)
        )
        model.train()
        x = torch.randn(2, 2, 1, 32, 32)
        teacher = torch.randn(2, 3, 2, 32, 32)
        out = model(x, T_out=3, teacher_seq=teacher, teacher_forcing_ratio=1.0)
        assert out.shape == (2, 3, 2, 32, 32)
        loss = out.sum()
        loss.backward()

