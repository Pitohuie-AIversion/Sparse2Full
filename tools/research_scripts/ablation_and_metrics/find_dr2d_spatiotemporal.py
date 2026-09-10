import json
import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
files = glob.glob(str(PROJECT_ROOT / "**" / "*test_results.json"), recursive=True)
for f in files:
    if "DR2D" not in f and "drd" not in f.lower():
        continue
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'final_test_metrics' in data:
                m = data['final_test_metrics']
                print(f"{f}")
                print(f"  Rel-L2: {m.get('rel_l2'):.4f}, PSNR: {m.get('psnr'):.2f}, SSIM: {m.get('ssim'):.4f}, H_err: {m.get('dc_error'):.4f}")
    except Exception as e:
        pass
