import json
import os
import glob

files = glob.glob("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/**/*.jsonl", recursive=True)

for f in files:
    if "Stage2" in f or "Stage3" in f or "SwinUNet-E2E" in f:
        print(f"Found jsonl: {f}")

print("\nChecking metric files inside eval directories:")
eval_files = glob.glob("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/**/eval/*.json", recursive=True)
for f in eval_files:
    if "Stage2" in f or "Stage3" in f or "SwinUNet-E2E" in f:
        try:
            with open(f) as fp:
                data = json.load(fp)
                print(f"\n--- {f} ---")
                print("Keys available:", list(data.keys()))
        except Exception as e:
            pass

