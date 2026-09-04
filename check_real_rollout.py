import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_drd_paper"

files = [
    "AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/test_results.json",
    "AR-DR2D-Stage3-VideoSwin-SRx4-JointFineTune-model_unknown-s2025-20260226/test_results.json",
    "AR-DR2D-SwinUNet-E2E-NoPretrain-SRx4-model_SwinUNet-s2025-20260115/test_results.json"
]

print("Checking for time_metrics or metrics_by_time in the actual test_results.json files...")

for f in files:
    path = RUNS_DIR / f
    if os.path.exists(path):
        try:
            with open(path) as fp:
                data = json.load(fp)
                print(f"\n--- {f} ---")
                print("Keys available:", list(data.keys()))
                for k in ["time_metrics", "metrics_by_time", "time_step_metrics"]:
                    if k in data:
                        print(f"  Found '{k}':", data[k])
                
                # Check inside final_test_metrics or similar if it's nested
                if 'final_test_metrics' in data:
                    print("  Keys in final_test_metrics:", list(data['final_test_metrics'].keys()))
        except Exception as e:
            print(f"Error reading {path}: {e}")
    else:
        print(f"\nFile not found: {path}")
