import json
import os

runs = [
    ("Consistent (1.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Consistent-Sigma1.0-model_UNet-s2025-20260115/test_results.json"),
    ("Mismatch (2.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Mismatch-Sigma2.0-model_UNet-s2025-20260115/test_results.json"),
    ("Mismatch (3.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Mismatch-Sigma3.0-model_UNet-s2025-20260115/test_results.json")
]

for name, path in runs:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            metrics = data.get('final_test_metrics', {})
            rel_l2 = metrics.get('rel_l2', 'N/A')
            psnr = metrics.get('psnr', 'N/A')
            ssim = metrics.get('ssim', 'N/A')
            herr = metrics.get('dc_error', 'N/A')
            print(f"{name}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, Herr={herr:.4f}")
    else:
        print(f"Not found: {path}")
