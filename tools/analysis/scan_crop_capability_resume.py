#!/usr/bin/env python3
"""
Crop Capability Scan Script (Resume Version)
Systematically evaluates model performance under varying crop sizes (sparsity levels).
Range: [48, 32, 16, 8, 4, 1]
Auto-detects previous runs and resumes from checkpoints if available.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import pandas as pd
import glob

# Configuration
SIZES = [48, 32, 16, 8, 4, 1]
BASE_CONFIG = "thesis_paper/configs/ar_paper_crop_edsr_spatial_only_refined.yaml"
PYTHON_EXE = sys.executable
SCRIPT_PATH = "tools/training/train_real_data_ar.py"
RUNS_DIR = Path("runs_drd_paper")

def find_latest_run(size):
    """Find the most recent run directory for a given crop size."""
    pattern = f"AR-DR2D-Crop-Scan-Size{size}-*"
    # Search in runs_drd_paper
    candidates = list(RUNS_DIR.glob(pattern))
    
    if not candidates:
        return None
        
    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def check_run_status(run_dir):
    """
    Check the status of a run directory.
    Returns: 'completed', 'interrupted', 'failed'
    """
    if not run_dir:
        return 'missing'
        
    # Check for completion
    if (run_dir / "test_results.json").exists():
        return 'completed'
        
    # Check for checkpoint
    if (run_dir / "last.ckpt").exists():
        return 'interrupted'
        
    return 'failed'

def run_experiment(size):
    size_str = str(size)  # Use scalar string for Hydra to avoid list type issues
    base_exp_name = f"AR-DR2D-Crop-Scan-Size{size}"
    
    print(f"\n{'='*50}")
    print(f"🔍 Checking status for Crop Size {size}x{size}...")
    
    latest_run = find_latest_run(size)
    status = check_run_status(latest_run)
    
    cmd = [
        PYTHON_EXE, "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port=29508",
        SCRIPT_PATH,
        "--config", BASE_CONFIG,
    ]
    
    # Common overrides
    overrides = [
        f"data.observation.crop.size={size_str}",
        f"training.degradation.crop.size={size_str}",
        f"training.degradation.crop_size={size_str}", # Explicit flat key to avoid nesting issues
        "data.dataloader.batch_size=32",  # Reduced from 192 to avoid OOM
        "training.batch_size=32",         # Reduced from 192 to avoid OOM
        "training.epochs=100",
        "visualization.max_images_per_sample=20"  # Increase limit to allow all viz types
    ]
    
    if status == 'completed':
        print(f"✅ Size {size} already completed in {latest_run.name}. FORCING RE-VISUALIZATION...")
        # Check for checkpoint to resume from (prefer best.ckpt for completed runs)
        ckpt_path = latest_run / "best.ckpt"
        if not ckpt_path.exists():
            ckpt_path = latest_run / "last.ckpt"
            
        if ckpt_path.exists():
            print(f"🚀 Resuming from {ckpt_path} to re-run test phase...")
            cmd.extend(["--resume", str(ckpt_path)])
            cmd.append(f"experiment.name={base_exp_name}")
            cmd.append("--test-only")  # Correctly use argparse flag
            cmd.extend(overrides)
        else:
            print("❌ No checkpoint found to resume from, skipping.")
            return True
        
    elif status == 'interrupted':
        # Check if best.ckpt exists -> treat as completed but missing results (so run test-only)
        best_ckpt = latest_run / "best.ckpt"
        last_ckpt = latest_run / "last.ckpt"
        
        if best_ckpt.exists():
            print(f"✅ Found best.ckpt at {latest_run.name}, treating as completed (running test-only)...")
            cmd.extend(["--resume", str(best_ckpt)])
            cmd.append("--test-only")
        else:
            print(f"🔄 Found interrupted run at {latest_run.name}, resuming training...")
            cmd.extend(["--resume", str(last_ckpt)])
        
        cmd.append(f"experiment.name={base_exp_name}")
        cmd.extend(overrides)
        
    else: # missing or failed
        print(f"🆕 Starting NEW experiment for Size {size}...")
        cmd.append(f"experiment.name={base_exp_name}")
        cmd.extend(overrides)
    
    # Set environment variables
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with error: {e}")
        return False

def collect_results():
    results = []
    for size in SIZES:
        latest_run = find_latest_run(size)
        if not latest_run:
            continue
            
        result_file = latest_run / "test_results.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('final_test_metrics', {})
                    
                    # Try to get H_err (might be named 'test_h_err' or similar)
                    h_err = metrics.get('test_h_err_mean', metrics.get('h_err', 0.0))
                    
                    entry = {
                        'Size': size,
                        'Area_Pct': (size*size)/(128*128)*100,
                        'Rel_L2': metrics.get('test_rel_l2_mean', metrics.get('rel_l2', 0.0)),
                        'PSNR': metrics.get('test_psnr_mean', metrics.get('psnr', 0.0)),
                        'SSIM': metrics.get('test_ssim_mean', metrics.get('ssim', 0.0)),
                        'H_Err': h_err,
                        'Path': latest_run.name
                    }
                    
                    # Try to get Params and FLOPs from model_resources.json
                    res_file = latest_run / "model_resources.json"
                    if res_file.exists():
                        try:
                            with open(res_file, 'r') as rf:
                                res_data = json.load(rf)
                                entry['Params(M)'] = res_data.get('params', 0) / 1e6
                                entry['FLOPs(G)'] = res_data.get('flops_g', 0.0)
                        except Exception:
                            entry['Params(M)'] = 0.0
                            entry['FLOPs(G)'] = 0.0
                    else:
                        entry['Params(M)'] = 0.0
                        entry['FLOPs(G)'] = 0.0

                    results.append(entry)
            except Exception as e:
                print(f"⚠️ Failed to parse results for {latest_run.name}: {e}")
    
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
    report_path = RUNS_DIR / "crop_scan_summary_resume.md"
    with open(report_path, "w") as f:
        f.write("# Crop Capability Scan Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()
