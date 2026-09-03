import json
import os

runs = [
    ("Consistent (1.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Consistent-Sigma1.0-model_UNet-s2025-20260115/eval/summary_stats.json"),
    ("Mismatch (2.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Mismatch-Sigma2.0-model_UNet-s2025-20260115/eval/summary_stats.json"),
    ("Mismatch (3.0)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Mismatch-Sigma3.0-model_UNet-s2025-20260115/eval/summary_stats.json")
]

for name, path in runs:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            rel_l2 = data.get('rel_l2_mean', 'N/A')
            psnr = data.get('psnr_mean', 'N/A')
            ssim = data.get('ssim_mean', 'N/A')
            herr = data.get('dc_error_mean', 'N/A')
            print(f"{name}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, Herr={herr:.4f}")
    else:
        print(f"Not found: {path}")
