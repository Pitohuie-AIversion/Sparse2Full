import pytest
import torch
from models.spatial.rcan import RCAN
from models.spatial.rdn import RDN
from models.spatial.nafnet import NAFNet
from models.spatial.swinir import SwinIR


class TestRestorationModels:
    """Unit tests for spatial restoration models (RCAN, RDN, NAFNet, SwinIR)."""

    def test_rcan_forward_and_backward(self):
        model = RCAN(
            in_channels=2,
            out_channels=1,
            n_feats=16,
            n_groups=2,
            n_blocks=2,
            upscale=1,
        )
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_rcan_upscale(self):
        model = RCAN(
            in_channels=1,
            out_channels=1,
            n_feats=16,
            n_groups=2,
            n_blocks=2,
            upscale=2,
        )
        x = torch.randn(1, 1, 16, 16)
        out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_rcan_nan_defense(self):
        model = RCAN(in_channels=1, out_channels=1, n_feats=16, n_groups=1, n_blocks=2)
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()

    def test_rdn_forward_and_backward(self):
        model = RDN(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            growth_rate=8,
            num_blocks=2,
            num_layers=2,
            scale=1,
        )
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 2, 32, 32)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_rdn_upscale(self):
        model = RDN(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            growth_rate=8,
            num_blocks=2,
            num_layers=2,
            scale=2,
        )
        x = torch.randn(1, 1, 16, 16)
        out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_rdn_nan_defense(self):
        model = RDN(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            growth_rate=8,
            num_blocks=1,
            num_layers=2,
        )
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()

    def test_nafnet_forward_and_backward(self):
        model = NAFNet(
            in_channels=2,
            out_channels=1,
            width=16,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            middle_blk_num=1,
        )
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_nafnet_nan_defense(self):
        model = NAFNet(
            in_channels=1,
            out_channels=1,
            width=16,
            enc_blk_nums=[1],
            dec_blk_nums=[1],
            middle_blk_num=1,
        )
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()

    def test_swinir_forward_and_backward(self):
        model = SwinIR(
            in_channels=2,
            out_channels=1,
            img_size=32,
            embed_dim=24,
            depths=[2, 2],
            num_heads=[3, 3],
            window_size=8,
        )
        x = torch.randn(2, 2, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_swinir_nan_defense(self):
        model = SwinIR(
            in_channels=1,
            out_channels=1,
            img_size=16,
            embed_dim=16,
            depths=[1],
            num_heads=[2],
            window_size=8,
        )
        x = torch.randn(1, 1, 16, 16)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.isfinite(out).all()
