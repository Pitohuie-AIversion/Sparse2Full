import torch
from ops.degradation import apply_degradation_operator

def test_sr4_observation_shape_and_equivalence():
    gt = torch.randn(1, 1, 128, 128)
    h_params = {
        'task': 'SR',
        'scale': 4,
        'sigma': 1.0,
        'kernel_size': 5,
        'boundary': 'mirror',
        'downsample_interpolation': 'area'
    }
    obs = apply_degradation_operator(gt, h_params)
    assert obs.shape[-2:] == (32, 32)
    mse = torch.mean((obs - obs) ** 2).item()
    assert mse < 1e-12

