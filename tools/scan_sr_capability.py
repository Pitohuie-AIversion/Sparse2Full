#!/usr/bin/env python3
"""
SR Capability Scan Script
Systematically evaluates model performance under varying input resolutions (Super-Resolution scales).
Range: Input sizes [32, 16, 8, 4, 2, 1] (Scales x4 to x128)
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import pandas as pd

# Configuration
# Input sizes corresponding to scales: 4, 8, 16, 32, 64, 128 (assuming 128x128 original)
# SIZES = [32, 16, 8, 4, 2, 1]

# Define scan levels: (input_size, scale_factor)
# Original size is 128x128
scan_levels = [
    # (32, 4),    # 32x32 -> 128x128 (Already Done)
    # (16, 8),    # 16x16 -> 128x128 (Already Done)
    # (8, 16),    # 8x8   -> 128x128 (Already Done)
    (4, 32),    # 4x4   -> 128x128
    (2, 64),    # 2x2   -> 128x128
    (1, 128),   # 1x1   -> 128x128
]
SIZES = [4, 2, 1] # 32, 16, 8 already done
BASE_CONFIG = "thesis_paper/configs/ar_paper_crop_edsr_spatial_only_refined.yaml"
PYTHON_EXE = sys.executable
SCRIPT_PATH = "tools/training/train_real_data_ar.py"
OUTPUT_ROOT = "runs_drd_paper/sr_scan_batch"  # Root directory for all scan experiments

def run_experiment(size):
    # Calculate scale factor
    original_size = 128
    scale = original_size // size
    
    exp_name = f"AR-DR2D-SR-Scan-Input{size}"
    output_dir = Path(OUTPUT_ROOT) / exp_name
    
    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: SR Input Size {size}x{size} (Scale x{scale})")
    print(f"   Name: {exp_name}")
    print(f"   Output Dir: {output_dir}")
    print(f"{'='*50}\n")
    
    # Use Dual GPU with torchrun for proper DDP
    cmd = [
        PYTHON_EXE,
        "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port=29500",
        SCRIPT_PATH,
        "--config", BASE_CONFIG,
        f"experiment.name={exp_name}",
        f"experiment.output_dir={output_dir}",
        "device.devices=2", # Let Lightning know we want 2 devices (it will align with DDP)
        "training.strategy=ddp", # Force DDP strategy
        
        # Switch Task to SR
        "data.observation.mode=sr",
        f"data.observation.scale={scale}",
        
        # Configure Degradation
        "training.degradation.mode=sr",
        f"training.degradation.scale_factor={scale}", # Explicitly set scale_factor for trainer manual parsing
        f"training.degradation.scale={scale}",        # Set scale as well for ops compatibility
        
        # Configure Model (EDSR needs upscale factor)
        f"model.upscale={scale}",
        
        # Adjust batch size for memory safety and utilization (L40 has 48GB)
        "data.dataloader.batch_size=256", # 256 per GPU = 512 total. Tiny inputs should fit easily.
        "training.batch_size=256",
        "training.epochs=100",  # consistent with crop scan
        "training.smoke_test=false", # Disable smoke test to avoid input shape mismatch (SR expects LR input, smoke test might pass GT)
        
        # Visualization
        "testing.save_visualizations=true",
        "testing.num_visualization_samples=5"
    ]
    
    # Set environment variables to fix MKL error
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    try:
        # Check if result already exists
        # output_dir is already defined above
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
    original_size = 128
    
    # Check all sizes, including completed ones
    ALL_SIZES = [32, 16, 8, 4, 2, 1]
    
    for size in ALL_SIZES:
        exp_name = f"AR-DR2D-SR-Scan-Input{size}"
        result_file = Path(OUTPUT_ROOT) / exp_name / "test_results.json"
        
        scale = original_size // size
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('final_test_metrics', {})
                    
                    # Try to get H_err (might be named 'test_h_err' or similar)
                    h_err = metrics.get('test_h_err_mean', metrics.get('h_err', 0.0))
                    
                    entry = {
                        'Input_Size': size,
                        'Scale': scale,
                        'Rel_L2': metrics.get('test_rel_l2_mean', metrics.get('rel_l2', 0.0)),
                        'PSNR': metrics.get('test_psnr_mean', metrics.get('psnr', 0.0)),
                        'SSIM': metrics.get('test_ssim_mean', metrics.get('ssim', 0.0)),
                        'H_Err': h_err
                    }
                    
                    # 3. Collect Resources
                    # Try model_resources.json
                    res_file = Path(OUTPUT_ROOT) / exp_name / "model_resources.json"
                    if res_file.exists():
                        with open(res_file, 'r') as f:
                            res_data = json.load(f)
                            entry['Params(M)'] = res_data.get('params', 0) / 1e6
                            entry['FLOPs(G)'] = res_data.get('flops_g', 0.0)
                            entry['Latency(ms)'] = res_data.get('inference_latency_ms_mean', 0.0)
                    else:
                        # Fallback for Params from model_info.json
                        info_file = Path(OUTPUT_ROOT) / exp_name / "model_info.json"
                        if info_file.exists():
                             with open(info_file, 'r') as f:
                                info_data = json.load(f)
                                entry['Params(M)'] = info_data.get('total_params', 0) / 1e6
                        else:
                            entry['Params(M)'] = 0.0
                            
                        entry['FLOPs(G)'] = 0.0
                        entry['Latency(ms)'] = 0.0

                    # Try resource_summary.json
                    summ_file = Path(OUTPUT_ROOT) / exp_name / "resource_summary.json"
                    if summ_file.exists():
                        with open(summ_file, 'r') as f:
                            summ_data = json.load(f)
                            entry['GPU_Mem(GB)'] = summ_data.get('max_gpu_peak_allocated_gb', 0.0)
                    else:
                        entry['GPU_Mem(GB)'] = 0.0
                        
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
    # Sort by Input Size descending (easier to harder)
    df = df.sort_values('Input_Size', ascending=False)
    
    print("\n=== SR Capability Scan Results ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    
    # Save to file
    report_path = Path(OUTPUT_ROOT) / "sr_scan_summary.md"
    # Ensure root dir exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# SR Capability Scan Results\n\n")
        f.write(f"Original Size: 128x128\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()
