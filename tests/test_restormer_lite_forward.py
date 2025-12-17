import torch
from models.spatial.factory import create_model

def test_restormer_lite_forward_shape():
    model = create_model('RestormerLite', in_channels=1, out_channels=1, img_size=128, embed_dim=48, depth=2)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)

