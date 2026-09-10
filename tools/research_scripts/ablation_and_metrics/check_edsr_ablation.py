import json
import os

runs = [
    ("EDSR (A0 RecOnly)", "runs_drd_paper/Ablation-A0-RecOnly-model_EDSR-s2025-20260110/eval/summary_stats.json"),
    ("EDSR (A3 Full)", "runs_drd_paper/Ablation-A3-Full-model_EDSR-s2025-20260110/eval/summary_stats.json"),
]

for name, path in runs:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
            if 'final_test_metrics' in data:
                metrics = data['final_test_metrics']
                rel_l2 = metrics.get('rel_l2', 'N/A')
                psnr = metrics.get('psnr', 'N/A')
                ssim = metrics.get('ssim', 'N/A')
                herr = metrics.get('dc_error', 'N/A')
                frmse_low = metrics.get('frmse_low', 'N/A')
            else:
                rel_l2 = data.get('rel_l2_mean', 'N/A')
                psnr = data.get('psnr_mean', 'N/A')
                ssim = data.get('ssim_mean', 'N/A')
                herr = data.get('dc_error_mean', 'N/A')
                frmse_low = data.get('frmse_low_mean', 'N/A')
                
            print(f"{name}: Rel-L2={rel_l2}, PSNR={psnr}, SSIM={ssim}, fRMSE-Low={frmse_low}, Herr={herr}")
    else:
        print(f"Not found: {name}")
