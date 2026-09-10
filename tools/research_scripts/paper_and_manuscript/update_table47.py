import json

runs = {
    "UNet_MSE": "runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/test_results.json",
    "UNet_DC": "runs_drd_paper/AR-DR2D-UNet-SRx4-Ablation-NoSpec-model_UNet-s2025-20260115/test_results.json",
    "UNet_Full": "runs_drd_paper/AR-DR2D-UNet-SRx4-Consistent-Sigma1.0-model_UNet-s2025-20260115/test_results.json",
    "EDSR_MSE": "runs_drd_paper/AR-DR2D-EDSR-SRx4-NoSpec-model_EDSR-s2025-20260115/test_results.json",
    "EDSR_Full": "runs_drd_paper/AR-DR2D-EDSR-SRx4-Consistent-Sigma1.0-model_EDSR-s2025-20260114/test_results.json"
}

metrics_map = {}
for name, path in runs.items():
    with open(path, 'r') as f:
        data = json.load(f)['final_test_metrics']
        metrics_map[name] = {
            'rel_l2': data['rel_l2'],
            'psnr': data['psnr'],
            'ssim': data['ssim'],
            'frmse_low': data['frmse_low'],
            'herr': data['dc_error']
        }

def format_row(m):
    return f"{m['rel_l2']:.4f} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['frmse_low']:.2f} | {m['herr']:.4f}"

print("UNet MSE Only: | " + format_row(metrics_map['UNet_MSE']))
print("UNet + L_dc:   | " + format_row(metrics_map['UNet_DC']))
print("UNet + Full:   | " + format_row(metrics_map['UNet_Full']))
print("EDSR MSE Only: | " + format_row(metrics_map['EDSR_MSE']))
print("EDSR + Full:   | " + format_row(metrics_map['EDSR_Full']))

gain_rel_l2 = (metrics_map['UNet_Full']['rel_l2'] - metrics_map['UNet_MSE']['rel_l2']) / metrics_map['UNet_MSE']['rel_l2'] * 100
gain_psnr = metrics_map['UNet_Full']['psnr'] - metrics_map['UNet_MSE']['psnr']
gain_ssim = (metrics_map['UNet_Full']['ssim'] - metrics_map['UNet_MSE']['ssim']) / metrics_map['UNet_MSE']['ssim'] * 100
gain_frmse = (metrics_map['UNet_Full']['frmse_low'] - metrics_map['UNet_MSE']['frmse_low']) / metrics_map['UNet_MSE']['frmse_low'] * 100
gain_herr = (metrics_map['UNet_Full']['herr'] - metrics_map['UNet_MSE']['herr']) / metrics_map['UNet_MSE']['herr'] * 100

print(f"Gain: {gain_rel_l2:.1f}% | +{gain_psnr:.1f}dB | +{gain_ssim:.1f}% | {gain_frmse:.1f}% | {gain_herr:.1f}%")
