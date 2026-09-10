#!/usr/bin/env python3
"""
Regenerate Visualizations for Crop Scan
Runs test-only mode on existing crop scan experiments to generate updated visualizations (including t=20).
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
SIZES = [48, 32, 16, 8, 4, 1]
RUNS_DIR = Path("runs_drd_paper")
PYTHON_EXE = sys.executable
SCRIPT_PATH = "tools/training/train_real_data_ar.py"

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

def regenerate_viz(size):
    run_dir = find_latest_run(size)
    if not run_dir:
        print(f"⚠️ No run found for size {size}")
        return False
        
    config_path = run_dir / "config_merged.yaml"
    ckpt_path = run_dir / "best.ckpt"
    
    if not config_path.exists() or not ckpt_path.exists():
        print(f"⚠️ Missing config or checkpoint for {run_dir.name}")
        return False
        
    print(f"\n{'='*50}")
    print(f"🎨 Regenerating Viz for Size {size}x{size} in {run_dir.name}")
    
    cmd = [
        PYTHON_EXE, SCRIPT_PATH,
        "--config", str(config_path),
        "--mode", "test",
        "--ckpt", str(ckpt_path),
        "--test-only" # Explicit flag
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Viz regeneration complete for {run_dir.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Viz regeneration failed for {run_dir.name}: {e}")
        return False

def main():
    print("🚀 Starting Visualization Regeneration...")
    for size in SIZES:
        regenerate_viz(size)
    print("\n🏁 All Done.")

if __name__ == "__main__":
    main()
