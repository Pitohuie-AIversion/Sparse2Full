import json
import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_drd_paper"

files = glob.glob(str(RUNS_DIR / "**" / "*.jsonl"), recursive=True)

for f in files:
    if "Stage2" in f or "Stage3" in f or "SwinUNet-E2E" in f:
        print(f"Found jsonl: {f}")

print("\nChecking metric files inside eval directories:")
eval_files = glob.glob(str(RUNS_DIR / "**" / "eval" / "*.json"), recursive=True)
for f in eval_files:
    if "Stage2" in f or "Stage3" in f or "SwinUNet-E2E" in f:
        try:
            with open(f) as fp:
                data = json.load(fp)
                print(f"\n--- {f} ---")
                print("Keys available:", list(data.keys()))
        except Exception as e:
            pass
