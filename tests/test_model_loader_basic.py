import torch
from omegaconf import OmegaConf
from tools.training.model_loader import create_model_with_loader

def test_unet_loader_forward():
    cfg = OmegaConf.create({"data": {"img_size": 64, "channels": 1}})
    m = create_model_with_loader("unet", cfg, in_channels=1, out_channels=1, img_size=64)
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == (1, 1, 64, 64)

def test_fno2d_loader_forward():
    cfg = OmegaConf.create({"data": {"img_size": 64, "channels": 1}})
    m = create_model_with_loader("fno2d", cfg, in_channels=1, out_channels=1, img_size=64)
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == (1, 1, 64, 64)

def test_mlpmixer_loader_forward():
    cfg = OmegaConf.create({"data": {"img_size": 64, "channels": 1}})
    m = create_model_with_loader("mlpmixer", cfg, in_channels=1, out_channels=1, img_size=64, patch_size=16)
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == (1, 1, 64, 64)