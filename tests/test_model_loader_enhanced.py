import torch
from omegaconf import OmegaConf
from tools.training.model_loader_enhanced import create_enhanced_model, test_enhanced_model as enhanced_forward_test

def test_swinunet_enhanced_forward():
    cfg = OmegaConf.create({"data": {"img_size": 64, "channels": 1}})
    m = create_enhanced_model("swin_unet", cfg, in_channels=1, out_channels=1, img_size=64)
    assert enhanced_forward_test("swin_unet", cfg, in_channels=1, out_channels=1, img_size=64)

def test_fallback_to_original_loader_unet():
    cfg = OmegaConf.create({"data": {"img_size": 64, "channels": 1}})
    m = create_enhanced_model("unet", cfg, in_channels=1, out_channels=1, img_size=64)
    x = torch.randn(1, 1, 64, 64)
    y = m(x)
    assert y.shape == (1, 1, 64, 64)

def test_enhanced_model_supports_listconfig_img_size():
    cfg = OmegaConf.create({"model": {"img_size": [64, 64], "in_channels": 1}})
    assert enhanced_forward_test("fno2d", cfg, in_channels=1, out_channels=1, img_size=[64, 64])
