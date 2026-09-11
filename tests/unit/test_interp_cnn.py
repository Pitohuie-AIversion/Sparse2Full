import pytest
import torch
from models.spatial.interp_cnn import BicubicCNN, RBFCNN


class TestInterpCNN:
    """Unit tests for interpolation-guided CNN baselines."""

    def test_bicubic_cnn_forward(self):
        model = BicubicCNN(in_channels=2, out_channels=2, img_size=64)
        x = torch.randn(2, 2, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 64, 64)

    def test_bicubic_cnn_channel_padding(self):
        model = BicubicCNN(in_channels=2, out_channels=2, img_size=32)
        x = torch.randn(2, 1, 32, 32)  # fewer channels than out_channels
        out = model(x)
        assert out.shape == (2, 2, 32, 32)

    def test_rbf_cnn_forward(self):
        model = RBFCNN(in_channels=2, out_channels=1, img_size=64)
        x = torch.randn(2, 2, 32, 32)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_rbf_cnn_all_masked_fallback(self):
        model = RBFCNN(in_channels=2, out_channels=1, img_size=32)
        x = torch.randn(1, 2, 32, 32)
        x[:, -1, :, :] = 0.0  # all mask is 0
        out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_gradient_flow(self):
        model = BicubicCNN(in_channels=1, out_channels=1, img_size=32)
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
