import json
import os

runs = [
    ("A0_Baseline_Repro_v2", "runs_3loss_ablation_unet/A0_Baseline_Repro_v2"),
    ("A2_RecSpec_Repro_v2", "runs_3loss_ablation_unet/A2_RecSpec_Repro_v2"),
    ("A2_RecDC_Repro_v2", "runs_3loss_ablation_unet/A2_RecDC_Repro_v2"),
    ("A3_Full_Repro_v2", "runs_3loss_ablation_unet/A3_Full_Repro_v2")
]

print("| Model (UNet) | Rel-L2 | MAE | PSNR | SSIM | fRMSE-low | bRMSE | cRMSE | ||H(ŷ)-y|| |")
print("|---|---|---|---|---|---|---|---|---|")

for name, path in runs:
    res_path = os.path.join(path, "test_results.json")
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            data = json.load(f)["final_test_metrics"]
            print(f"| {name} | {data['rel_l2']:.4f} | {data['mae']:.4f} | {data['psnr']:.2f} | {data['ssim']:.4f} | {data['frmse_low']:.2f} | {data['brmse']:.4f} | {data['crmse']:.4f} | {data['dc_error']:.4f} |")
    else:
        print(f"| {name} | Missing |")
