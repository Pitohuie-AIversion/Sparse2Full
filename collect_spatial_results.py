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

base_dir = Path("runs")
print(f"{'Experiment':<60} | {'Rel L2':<10} | {'MAE':<10} | {'RMSE':<10}")
print("-" * 100)

# 遍历所有子目录
for root, dirs, files in os.walk(base_dir):
    if "test_results.json" in files:
        path = Path(root) / "test_results.json"
        metrics = read_metrics(path)
        
        if metrics:
            rel_l2 = metrics.get('test_rel_l2', metrics.get('rel_l2', float('nan')))
            mae = metrics.get('test_mae', metrics.get('mae', float('nan')))
            rmse = metrics.get('test_rmse', metrics.get('rmse', float('nan')))
            
            exp_name = os.path.basename(root)
            # Filter dry_run and smoke test to reduce noise, but print others
            if "dry_run" in exp_name or "SmokeTest" in exp_name:
                continue
                
            print(f"{exp_name:<60} | {rel_l2:.6f}   | {mae:.6f}   | {rmse:.6f}")
