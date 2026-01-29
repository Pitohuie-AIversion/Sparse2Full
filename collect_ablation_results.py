import json
import os
from pathlib import Path

def read_metrics(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('final_test_metrics', {})
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

base_dir = Path("runs_3loss_ablation")
experiments = {
    "A0_Base": base_dir / "A0_Base" / "test_results.json",
    "A2_RecSpec": base_dir / "A2_RecSpec" / "test_results.json",
    "A3_Full": base_dir / "A3_Full" / "test_results.json"
}

print(f"{'Experiment':<12} | {'Rel L2':<10} | {'MAE':<10} | {'RMSE':<10}")
print("-" * 50)

for name, path in experiments.items():
    metrics = read_metrics(path)
    if metrics:
        # Try both 'test_rel_l2' and 'rel_l2'
        rel_l2 = metrics.get('test_rel_l2', metrics.get('rel_l2', float('nan')))
        mae = metrics.get('test_mae', metrics.get('mae', float('nan')))
        rmse = metrics.get('test_rmse', metrics.get('rmse', float('nan')))
        print(f"{name:<12} | {rel_l2:.6f}   | {mae:.6f}   | {rmse:.6f}")
    else:
        print(f"{name:<12} | Not Found / Error")
