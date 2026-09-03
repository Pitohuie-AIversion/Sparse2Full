import os
import glob
import json

runs = [
    # EDSR
    ("EDSR A0 (MSE Only)", "runs_drd_paper/Ablation-A0-RecOnly-model_EDSR-s2025-20260110/test_results.json"),
    ("EDSR A2 (RecSpec)", "runs_drd_paper/Ablation-A2-RecSpec-model_EDSR-s2025-20260110/test_results.json"),
    ("EDSR A3 (Full)", "runs_drd_paper/Ablation-A3-Full-model_EDSR-s2025-20260110/test_results.json"),
    # UNet
    ("UNet A0 (MSE Only)", "runs_3loss_ablation_unet_100ep/A0_Baseline_Repro_v4/test_results.json"),
    ("UNet A2 (RecSpec)", "runs_3loss_ablation_unet_100ep/A2_RecSpec_Repro_v4/test_results.json"),
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
        # Check if there's an eval/summary_stats.json
        alt_path = path.replace("test_results.json", "eval/summary_stats.json")
        if os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                data = json.load(f)
                rel_l2 = data.get('rel_l2_mean', 'N/A')
                psnr = data.get('psnr_mean', 'N/A')
                ssim = data.get('ssim_mean', 'N/A')
                herr = data.get('dc_error_mean', 'N/A')
                print(f"[From summary_stats] {name}: Rel-L2={rel_l2:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, Herr={herr:.4f}")
        else:
            print(f"Not found: {path} (or {alt_path})")
