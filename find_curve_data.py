import os
import json
import glob
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
files = glob.glob(str(PROJECT_ROOT / "**" / "*.json"), recursive=True)
for f in files:
    if "runs" not in f and "paper_package" not in f:
        continue
    if "time" in f.lower() or "step" in f.lower() or "rollout" in f.lower() or "curve" in f.lower():
        try:
            with open(f) as fp:
                data = json.load(fp)
                if isinstance(data, dict) and any(k for k in data.keys() if "time" in k or "step" in k or isinstance(data[k], list)):
                    print(f"Potential time-series data in: {f}")
        except:
            pass
