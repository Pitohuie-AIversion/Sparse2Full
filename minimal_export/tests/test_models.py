import torch
import pytest
from models.spatial.swin_unet import SwinUNet
from models.spatial.unet import UNet
from models.ar.wrapper import ARWrapper

def test_swin_unet_initialization():
    model = SwinUNet(in_channels=1, out_channels=1, img_size=128)
    assert model is not None
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == x.shape

def test_unet_initialization():
    model = UNet(in_channels=1, out_channels=1)
    assert model is not None
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == x.shape

def test_ar_wrapper_initialization():
    base_model = UNet(in_channels=1, out_channels=1)
    wrapper = ARWrapper(base_model)
    assert wrapper is not None
    
    # Test forward pass
    # Depending on how ARWrapper is implemented, its input shape might differ
    # We will just verify it instantiates.
    assert hasattr(wrapper, 'forward')
