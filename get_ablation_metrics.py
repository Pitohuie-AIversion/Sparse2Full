import json
import os

runs = [
    ("EDSR A0 (RecOnly)", "runs_drd_paper/Ablation-A0-RecOnly-model_EDSR-s2025-20260110/test_results.json"),
    ("EDSR A2 (RecSpec)", "runs_drd_paper/Ablation-A2-RecSpec-model_EDSR-s2025-20260110/test_results.json"),
    ("EDSR A3 (Full)", "runs_drd_paper/Ablation-A3-Full-model_EDSR-s2025-20260110/test_results.json"),
    ("UNet NoSpec", "runs_drd_paper/AR-DR2D-UNet-SRx4-Ablation-NoSpec-model_UNet-s2025-20260115/test_results.json"),
    ("UNet Consistent (Full)", "runs_drd_paper/AR-DR2D-UNet-SRx4-Consistent-Sigma1.0-model_UNet-s2025-20260115/test_results.json"),
    ("EDSR NoSpec", "runs_drd_paper/AR-DR2D-EDSR-SRx4-NoSpec-model_EDSR-s2025-20260115/test_results.json"),
    ("EDSR Consistent (Full)", "runs_drd_paper/AR-DR2D-EDSR-SRx4-Consistent-Sigma1.0-model_EDSR-s2025-20260114/test_results.json")
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
            frmse_low = metrics.get('frmse_low', 'N/A')
            print(f"{name}: Rel-L2={rel_l2}, PSNR={psnr}, SSIM={ssim}, Herr={herr}, fRMSE-Low={frmse_low}")
    else:
        print(f"Not found: {path}")
