#!/usr/bin/env python3
"""
Crop Capability Scan Script
Systematically evaluates model performance under varying crop sizes (sparsity levels).
Range: [48, 32, 16, 8, 4, 1]
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import pandas as pd

# Configuration
SIZES = [48, 32, 16, 8, 4, 1]
BASE_CONFIG = "thesis_paper/configs/ar_paper_crop_edsr_spatial_only_refined.yaml"
PYTHON_EXE = sys.executable
SCRIPT_PATH = "tools/training/train_real_data_ar.py"

def run_experiment(size):
    exp_name = f"AR-DR2D-Crop-Scan-Size{size}"
    size_str = f"[{size},{size}]"
    
    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: Crop Size {size}x{size}")
    print(f"   Name: {exp_name}")
    print(f"{'='*50}\n")
    
    # Use DDP with 2 GPUs
    cmd = [
        PYTHON_EXE, "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port=29505",
        SCRIPT_PATH,
        "--config", BASE_CONFIG,
        f"experiment.name={exp_name}",
        f"data.observation.crop.size={size_str}",
        f"training.degradation.crop.size={size_str}",
        "data.dataloader.batch_size=192",
        "training.batch_size=192",
        "training.epochs=100"  # Reduce epochs for faster scanning
    ]
    
    # Set environment variables to fix MKL error
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    try:
        # Check if result already exists
        output_dir = Path(f"runs_drd_paper/{exp_name}")
        result_file = output_dir / "test_results.json"
        
        if result_file.exists():
            print(f"✅ Experiment {exp_name} already completed. Skipping.")
            return True
            
        # Run training
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {exp_name} failed with error: {e}")
        return False

def collect_results():
    results = []
    for size in SIZES:
        exp_name = f"AR-DR2D-Crop-Scan-Size{size}"
        result_file = Path(f"runs_drd_paper/{exp_name}/test_results.json")
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('final_test_metrics', {})
                    
                    # Try to get H_err (might be named 'test_h_err' or similar)
                    # Based on previous outputs, H_err key might vary, let's look for common ones
                    h_err = metrics.get('test_h_err_mean', metrics.get('h_err', 0.0))
                    
                    entry = {
                        'Size': size,
                        'Area_Pct': (size*size)/(128*128)*100,
                        'Rel_L2': metrics.get('test_rel_l2_mean', metrics.get('rel_l2', 0.0)),
                        'PSNR': metrics.get('test_psnr_mean', metrics.get('psnr', 0.0)),
                        'SSIM': metrics.get('test_ssim_mean', metrics.get('ssim', 0.0)),
                        'H_Err': h_err
                    }
                    results.append(entry)
            except Exception as e:
                print(f"⚠️ Failed to parse results for {exp_name}: {e}")
    
    return results

def main():
    # 1. Execute Scan
    for size in SIZES:
        success = run_experiment(size)
        if not success:
            print(f"⚠️ Stopping scan due to failure at size {size}")
            break
            
    # 2. Generate Report
    print("\n📊 Generating Summary Report...")
    results = collect_results()
    
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    df = df.sort_values('Size', ascending=False)
    
    print("\n=== Crop Capability Scan Results ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save to file
    report_path = "runs_drd_paper/crop_scan_summary.md"
    with open(report_path, "w") as f:
        f.write("# Crop Capability Scan Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()
