
import torch
import torch.nn.functional as F
from utils.metrics import compute_all_metrics, MetricsCalculator

def test_metrics_inputs():
    print("Testing metrics with various input shapes...")
    
    H, W = 32, 32
    C = 2
    B = 3
    T = 4
    
    # 1. [H, W]
    pred_hw = torch.rand(H, W)
    target_hw = torch.rand(H, W)
    print(f"\n1. Testing [H, W]: {pred_hw.shape}")
    metrics = compute_all_metrics(pred_hw, target_hw)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
            assert v.shape == (1, 1), f"{k} shape mismatch for HW input: {v.shape}"

    # 2. [C, H, W]
    pred_chw = torch.rand(C, H, W)
    target_chw = torch.rand(C, H, W)
    print(f"\n2. Testing [C, H, W]: {pred_chw.shape}")
    metrics = compute_all_metrics(pred_chw, target_chw)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
            assert v.shape == (1, C), f"{k} shape mismatch for CHW input: {v.shape}"

    # 3. [B, C, H, W]
    pred_bchw = torch.rand(B, C, H, W)
    target_bchw = torch.rand(B, C, H, W)
    print(f"\n3. Testing [B, C, H, W]: {pred_bchw.shape}")
    metrics = compute_all_metrics(pred_bchw, target_bchw)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
            assert v.shape == (B, C), f"{k} shape mismatch for BCHW input: {v.shape}"

    # 4. [B, T, C, H, W]
    pred_btchw = torch.rand(B, T, C, H, W)
    target_btchw = torch.rand(B, T, C, H, W)
    print(f"\n4. Testing [B, T, C, H, W]: {pred_btchw.shape}")
    metrics = compute_all_metrics(pred_btchw, target_btchw)
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
            assert v.shape == (B, C), f"{k} shape mismatch for BTCHW input: {v.shape}"

    print("\nAll shape tests passed!")

def test_dc_error_logic():
    print("\nTesting DC Error logic...")
    # Mock obs_data
    obs_data = {
        'baseline': torch.rand(1, 1, 16, 16), # Smaller size
        'baseline_is_norm': False
    }
    pred = torch.rand(1, 1, 32, 32)
    norm_stats = {'mean': 0.0, 'std': 1.0}
    
    # We need to monkeypatch apply_degradation_operator if it's not working,
    # but metrics.py has a fallback.
    # The fallback returns input x.
    # So if pred is 32x32, apply_degradation returns 32x32.
    # baseline is 16x16.
    # metrics should interpolate 32x32 -> 16x16.
    
    calc = MetricsCalculator()
    dc_err = calc.compute_data_consistency_error(pred, obs_data, norm_stats)
    print(f"DC Error shape: {dc_err.shape}")
    assert dc_err.shape == (1, 1)
    print("DC Error logic passed (interpolation check).")

def test_ssim_stability_check():
    print("\nTesting SSIM stability (constant image)...")
    calc = MetricsCalculator()
    target = torch.ones(1, 1, 32, 32)
    pred = target.clone()
    ssim_val = calc.compute_ssim(pred, target)
    print(f"SSIM (const): {ssim_val}")
    assert ssim_val.item() > 0.99
    print("SSIM stability passed.")

def test_psnr_batch_max():
    print("\nTesting PSNR batch max logic...")
    calc = MetricsCalculator()
    # Batch 0: range [0, 1]
    # Batch 1: range [0, 10]
    target = torch.zeros(2, 1, 32, 32)
    target[0] = torch.rand(1, 32, 32)
    target[1] = torch.rand(1, 32, 32) * 10
    pred = target + 0.01
    
    psnr = calc.compute_psnr(pred, target)
    print(f"PSNR: {psnr}")
    # PSNR[0] should be based on max ~1 -> 20*log10(1/0.01) = 40
    # PSNR[1] should be based on max ~10 -> 20*log10(10/0.01) = 60
    # If global max (10) used for both, PSNR[0] would be 60.
    
    if psnr[0] > 50:
        print("FAIL: PSNR[0] is too high, implies global max usage.")
    else:
        print("PASS: PSNR[0] reasonable.")

if __name__ == "__main__":
    test_metrics_inputs()
    test_dc_error_logic()
    test_ssim_stability_check()
    test_psnr_batch_max()
