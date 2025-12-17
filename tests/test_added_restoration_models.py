import torch
from models.spatial.factory import create_model

def _forward_ok(name):
    m = create_model(name, in_channels=1, out_channels=1, img_size=128)
    x = torch.randn(2,1,128,128)
    y = m(x)
    assert y.shape == (2,1,128,128)

def test_swinir_lite():
    _forward_ok('SwinIRLite')

def test_nafnet_lite():
    _forward_ok('NAFNetLite')

def test_uformer_lite():
    _forward_ok('UformerLite')

