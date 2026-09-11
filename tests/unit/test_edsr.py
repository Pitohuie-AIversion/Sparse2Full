import pytest
import torch
from models.spatial.edsr import EDSR


class TestEDSR:
    """Unit tests for EDSR model."""

    def test_same_resolution_forward(self):
        model = EDSR(in_channels=1, out_channels=1, upscale=1, n_feats=32, n_resblocks=4)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_multichannel_forward(self):
        model = EDSR(in_channels=3, out_channels=2, upscale=1, n_feats=32, n_resblocks=2)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 32, 32)

    def test_upscale_forward(self):
        # Test SRx2
        model_x2 = EDSR(in_channels=1, out_channels=1, upscale=2, n_feats=32, n_resblocks=2)
        x = torch.randn(1, 1, 32, 32)
        out_x2 = model_x2(x)
        assert out_x2.shape == (1, 1, 64, 64)

        # Test SRx4
        model_x4 = EDSR(in_channels=1, out_channels=1, upscale=4, n_feats=32, n_resblocks=2)
        out_x4 = model_x4(x)
        assert out_x4.shape == (1, 1, 128, 128)

    def test_gradient_checkpointing_and_backward(self):
        model = EDSR(in_channels=2, out_channels=2, upscale=1, n_feats=32, n_resblocks=8)
        model.train()
        model.set_gradient_checkpointing(True)
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_nan_inf_defense(self):
        model = EDSR(in_channels=1, out_channels=1, upscale=1, n_feats=16, n_resblocks=2)
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        x[0, 0, 0, 1] = float("inf")
        out = model(x)
        assert torch.isfinite(out).all()
