import json
import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
files = (
    glob.glob(str(PROJECT_ROOT / "**" / "*test_results.json"), recursive=True)
    + glob.glob(str(PROJECT_ROOT / "**" / "*summary_stats.json"), recursive=True)
)

for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'final_test_metrics' in data:
                m = data['final_test_metrics']
                l2 = m.get('rel_l2', 0)
                psnr = m.get('psnr', 0)
                if abs(l2 - 0.1780) < 0.001 or abs(psnr - 36.29) < 0.01:
                    print(f"FOUND in {f}: Rel-L2: {l2:.4f}, PSNR: {psnr:.2f}, SSIM: {m.get('ssim'):.4f}, H_err: {m.get('dc_error'):.4f}")
            elif 'rel_l2' in data and 'psnr' in data:
                l2 = data['rel_l2'].get('mean', 0)
                psnr = data['psnr'].get('mean', 0)
                if abs(l2 - 0.1780) < 0.001 or abs(psnr - 36.29) < 0.01:
                    print(f"FOUND in {f}: Rel-L2: {l2:.4f}, PSNR: {psnr:.2f}")
    except Exception as e:
        pass
