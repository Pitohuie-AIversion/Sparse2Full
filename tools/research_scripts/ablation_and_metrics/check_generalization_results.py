import json
import os
import re

def get_metrics(report_path):
    metrics = {}
    if not os.path.exists(report_path):
        return metrics
    with open(report_path, 'r') as f:
        content = f.read()
        
    rel_l2 = re.search(r'<td><strong>REL_L2</strong></td>\s*<td>([\d.]+)</td>', content)
    psnr = re.search(r'<td><strong>PSNR</strong></td>\s*<td>([\d.]+)</td>', content)
    dc_error = re.search(r'<td><strong>DC_ERROR</strong></td>\s*<td>([\d.]+)</td>', content)
    ssim = re.search(r'<td><strong>SSIM</strong></td>\s*<td>([\d.]+)</td>', content)
    
    if rel_l2: metrics['Rel_L2'] = float(rel_l2.group(1))
    if psnr: metrics['PSNR'] = float(psnr.group(1))
    if dc_error: metrics['DC_Error'] = float(dc_error.group(1))
    if ssim: metrics['SSIM'] = float(ssim.group(1))
    
    return metrics

experiments = {
    'Zero-shot SWE pretrained on DRD': 'runs/AR-SW-10M-edsr/test_visualizations/test_report.html',
    'Zero-shot DRD pretrained on SWE': 'runs_drd_paper/AR-DR2D-Stage1-EDSR-SRx4-model_EDSR-s2025-20260116/test_visualizations/test_report.html',
    'Few-shot SWE pretrained on DRD': 'runs/expA_fewshot_drd/test_visualizations/test_report.html',
    'Few-shot DRD pretrained on SWE': 'runs/expA_fewshot_swe/test_visualizations/test_report.html',
}

print("=== Experiment A: Cross-Equation Generalization Results ===")
print(f"{'Experiment':<35} | {'Rel_L2':<8} | {'PSNR':<8} | {'SSIM':<8} | {'DC_Error':<8}")
print("-" * 75)

for name, path in experiments.items():
    m = get_metrics(path)
    if m:
        print(f"{name:<35} | {m.get('Rel_L2', 'N/A'):<8.4f} | {m.get('PSNR', 'N/A'):<8.4f} | {m.get('SSIM', 'N/A'):<8.4f} | {m.get('DC_Error', 'N/A'):<8.4f}")
    else:
        print(f"{name:<35} | Running...")
