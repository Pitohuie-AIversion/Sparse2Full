import pytest
import torch
from models.spatial.cnn_attn_lite import CNNAttnLite
from models.spatial.conv_gate_lite import ConvGateLite
from models.spatial.conv_unet_lite import ConvUNetLite


class TestLiteModels:
    """Unit tests for lightweight spatial backbones."""

    def test_cnn_attn_lite_forward_and_backward(self):
        model = CNNAttnLite(in_channels=2, out_channels=1, img_size=32, embed_dim=16, depth=2)
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_cnn_attn_lite_nan_defense(self):
        model = CNNAttnLite(in_channels=1, out_channels=1, img_size=16, embed_dim=16, depth=1)
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()

    def test_conv_gate_lite_forward_and_backward(self):
        model = ConvGateLite(in_channels=2, out_channels=2, img_size=32, embed_dim=16, depth=2)
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 2, 32, 32)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_conv_gate_lite_nan_defense(self):
        model = ConvGateLite(in_channels=1, out_channels=1, img_size=16, embed_dim=16, depth=1)
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()

    def test_conv_unet_lite_forward_and_backward(self):
        model = ConvUNetLite(in_channels=3, out_channels=1, img_size=32, embed_dim=16)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_conv_unet_lite_nan_defense(self):
        model = ConvUNetLite(in_channels=1, out_channels=1, img_size=16, embed_dim=16)
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()
