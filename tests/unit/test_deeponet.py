import pytest
import torch
from models.spatial.deeponet import DeepONet


class TestDeepONet:
    """Unit tests for DeepONet (2D neural operator) model."""

    def test_forward_shape(self):
        model = DeepONet(
            in_channels=1,
            out_channels=1,
            img_size=64,
            latent_dim=64,
            branch_channels=(32, 64),
            trunk_hidden=(64, 64),
        )
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_multichannel(self):
        model = DeepONet(
            in_channels=3,
            out_channels=2,
            img_size=32,
            latent_dim=64,
            branch_channels=(32, 64),
            trunk_hidden=(64, 64),
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 32, 32)

    def test_gradient_flow(self):
        model = DeepONet(
            in_channels=1,
            out_channels=1,
            img_size=32,
            latent_dim=32,
            branch_channels=(16, 32),
            trunk_hidden=(32, 32),
        )
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_without_fourier_features(self):
        model = DeepONet(
            in_channels=1,
            out_channels=1,
            img_size=32,
            latent_dim=32,
            branch_channels=(16, 32),
            trunk_hidden=(32, 32),
            use_fourier_features=False,
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_nan_inf_defense(self):
        model = DeepONet(
            in_channels=1,
            out_channels=1,
            img_size=32,
            latent_dim=32,
            branch_channels=(16, 32),
            trunk_hidden=(32, 32),
        )
        x = torch.randn(1, 1, 32, 32)
        x[0, 0, 0, 0] = float("nan")
        x[0, 0, 0, 1] = float("inf")
        out = model(x)
        assert torch.isfinite(out).all()
