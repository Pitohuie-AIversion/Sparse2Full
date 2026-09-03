import json
import os

runs = [
    ("UNet A0 (MSE Only)", "runs_3loss_ablation_unet_100ep/A0_Baseline_Repro_v4/test_results.json"),
    ("UNet A2 (RecDC)", "runs_3loss_ablation_unet_100ep/A2_RecDC_Repro_v4/test_results.json"),
    ("UNet A3 (Full)", "runs_3loss_ablation_unet_100ep/A3_Full_Repro_v4/test_results.json"),
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
            print(f"{name}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, Herr={herr:.4f}, fRMSE-Low={frmse_low:.2f}")
    else:
        print(f"Not found: {path}")
