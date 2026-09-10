#!/usr/bin/env python3
"""
Crop Capability Scan Script - UNet Edition
Systematically evaluates UNet model performance under varying crop sizes.
Range: [32, 48, 64, 80, 96, 112]
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import pandas as pd

# Configuration
SIZES = [112, 96, 80, 64, 48, 32, 24, 16, 8, 4, 2, 1]
BASE_CONFIG = "thesis_paper/configs/ar_paper_aligned_crop.yaml"
PYTHON_EXE = sys.executable
SCRIPT_PATH = "tools/training/train_real_data_ar.py"
OUTPUT_ROOT = "runs/UNet_Crop_Scan"

import random

def get_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_experiment(size):
    exp_name = f"AR-DR2D-Crop-Scan-Size{size}-UNet"
    size_str = f"[{size},{size}]"
    
    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: Crop Size {size}x{size} (UNet)")
    print(f"   Name: {exp_name}")
    print(f"{'='*50}\n")
    
    port = get_free_port()
    
    # Construct command
    cmd = [
        PYTHON_EXE, "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={port}",
        SCRIPT_PATH,
        "--config", BASE_CONFIG,
        "--model", "unet",
        f"experiment.name={exp_name}",
        f"experiment.output_dir={OUTPUT_ROOT}/{exp_name}",
        f"data.dataloader.batch_size=512", # Increase batch size to utilize GPU (total 1024)
        f"training.epochs=100",
        f"data.dataloader.num_workers=8", # Increase workers per GPU
        f"data.observation.crop_size={size_str}",
        f"data.observation.crop_mode=center",
        f"model.img_size=128",
        # Enable training crop (Masking) logic explicitly
        f"training.crop.enabled=true",
        f"training.crop.size={size_str}", # Use 'size' instead of 'crop_size' for _apply_random_masking
        f"training.crop.mode=center",
        f"training.crop.patches_per_image=1", # Explicitly set patches_per_image
        # Ensure training uses the same crop size for consistency check if needed
        # (Though usually observation handles it, explicitly setting it avoids ambiguity)
        f"training.degradation.crop_size={size_str}" 
    ]
    
    # Set environment variables
    env = os.environ.copy()
    
    try:
        # Check if result already exists
        output_dir = Path(f"{OUTPUT_ROOT}/{exp_name}")
        # Assuming metrics.jsonl or similar is generated. 
        # The script usually saves 'test_results.json' or 'metrics.jsonl'.
        # Let's check for 'training.log' as a basic check, or better yet, check if it finished.
        # But for now, we'll just run it. If it supports resuming, it might handle it.
        # To be safe, we can check for a final checkpoint or a results file.
        
        # Run training
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {exp_name} failed with error: {e}")
        return False

def collect_results():
    results = []
    for size in SIZES:
        exp_name = f"AR-DR2D-Crop-Scan-Size{size}-UNet"
        # The training script usually outputs to {OUTPUT_ROOT}/{exp_name}
        # We need to find where the metrics are saved.
        # Usually it's in metrics.jsonl or test_results.json.
        # Based on project rules, it might be metrics.jsonl.
        
        output_dir = Path(f"{OUTPUT_ROOT}/{exp_name}")
        
        # Try to find metrics file
        metrics_file = output_dir / "metrics.jsonl"
        if not metrics_file.exists():
             metrics_file = output_dir / "test_results.json"
        
        if metrics_file.exists():
            try:
                # Read the last line of metrics.jsonl or load json
                if metrics_file.suffix == '.jsonl':
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            data = json.loads(lines[-1])
                            # Extract metrics
                            # Assuming standard keys
                            rel_l2 = data.get('test/rel_l2', data.get('test_rel_l2', 0.0))
                            psnr = data.get('test/psnr', data.get('test_psnr', 0.0))
                            ssim = data.get('test/ssim', data.get('test_ssim', 0.0))
                            
                            entry = {
                                'Model': 'UNet',
                                'Size': size,
                                'Area_Pct': (size*size)/(128*128)*100,
                                'Rel_L2': rel_l2,
                                'PSNR': psnr,
                                'SSIM': ssim
                            }
                            results.append(entry)
                elif metrics_file.suffix == '.json':
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        # Adapt based on json structure
                        metrics = data.get('final_test_metrics', data)
                        rel_l2 = metrics.get('test_rel_l2_mean', metrics.get('rel_l2', 0.0))
                        psnr = metrics.get('test_psnr_mean', metrics.get('psnr', 0.0))
                        ssim = metrics.get('test_ssim_mean', metrics.get('ssim', 0.0))
                        
                        entry = {
                            'Model': 'UNet',
                            'Size': size,
                            'Area_Pct': (size*size)/(128*128)*100,
                            'Rel_L2': rel_l2,
                            'PSNR': psnr,
                            'SSIM': ssim
                        }
                        results.append(entry)

            except Exception as e:
                print(f"⚠️ Failed to parse results for {exp_name}: {e}")
        else:
            print(f"⚠️ No metrics file found for {exp_name}")
    
    return results

def main():
    # 1. Execute Scan
    for size in SIZES:
        success = run_experiment(size)
        if not success:
            print(f"⚠️ Stopping scan due to failure at size {size}")
            # We might want to continue or break. 
            # If one fails, maybe others will fail too.
            # But let's try to continue for now? 
            # Actually, usually break is safer.
            break
            
    # 2. Generate Report
    print("\n📊 Generating Summary Report...")
    results = collect_results()
    
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    df = df.sort_values('Size', ascending=False)
    
    print("\n=== Crop Capability Scan Results (UNet) ===")
    # Check if to_markdown is available (requires tabulate)
    try:
        print(df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(df.to_string(index=False))
    
    # Save to file
    report_path = f"{OUTPUT_ROOT}/crop_scan_unet_summary.md"
    # Ensure parent dir exists
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Crop Capability Scan Results (UNet)\n\n")
        try:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            f.write(df.to_string(index=False))
    
    print(f"\n✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()
