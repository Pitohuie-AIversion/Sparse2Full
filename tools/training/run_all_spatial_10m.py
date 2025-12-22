#!/usr/bin/env python3
"""
Script to run training for classified spatial models with ~10M parameters on Shallow Water dataset.
Uses 2 GPUs via DDP.
Supports RESUMING from interrupted runs.
Uses dynamic ports to avoid collisions.
"""

import sys
import os
import subprocess
import time
import glob
import socket
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Define Model Categories based on factory.py and __init__.py
MODEL_CATEGORIES = {
    "CNN": [
        "UNet",
        "UNetPlusPlus",
        "FNO2d",
        "UFNOUNet"
    ],
    "Transformer": [
        "SegFormer",
        "UNetFormer",
        "SegFormerUNetFormer",
        "ViT",
        "SwinT",
        "Transformer",
        "RestormerLite",
        "SwinIRLite",
        "NAFNetLite",
        "UformerLite"
    ],
    "MLP": [
        "MLP",
        "MLPMixer",
        "LIIF"
    ],
    "Hybrid": [
        "SwinUNet",
        "Hybrid"
    ],
    "Sparse": [
        "SparseSwinUNet"
    ]
}

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def find_existing_run(model_name):
    """
    Finds the latest run directory for a given model.
    Returns (status, path)
    status: 'completed', 'interrupted', 'not_found'
    path: path to directory (if found) or checkpoint (if interrupted)
    """
    runs_dir = project_root / "runs"
    # Search for directories matching the pattern
    # Pattern: AR-ShallowWater-10M-{model_name}*
    # Note: We need to be careful not to match partial names (e.g. UNet vs UNetFormer)
    # So we look for AR-ShallowWater-10M-{model_name}-* 
    # But wait, the experiment name is AR-ShallowWater-10M-{model_name}
    # The trainer appends -s{seed}-{date} or similar.
    # So we search for folders starting with AR-ShallowWater-10M-{model_name}-
    
    candidates = []
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            if not d.is_dir():
                continue
            # Check if it matches the specific model
            # We need to ensure we don't match UNetFormer when looking for UNet
            prefix = f"AR-ShallowWater-10M-{model_name}-"
            if d.name.startswith(prefix):
                 candidates.append(d)
    
    if not candidates:
        return 'not_found', None
    
    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    latest_run = candidates[0]
    
    # Check for completion
    # test_results.json indicates full completion (including testing)
    if (latest_run / "test_results.json").exists():
        return 'completed', latest_run
    
    # Check for checkpoint to resume
    # Check in root first (common behavior)
    ckpt_path = latest_run / "last.ckpt"
    if ckpt_path.exists():
        return 'interrupted', ckpt_path
        
    # Check in checkpoints/ subdir (alternative behavior)
    ckpt_path = latest_run / "checkpoints" / "last.ckpt"
    if ckpt_path.exists():
        return 'interrupted', ckpt_path
        
    # If no checkpoint, we might have just started and crashed, or no checkpoints saved yet.
    # In this case, we treat it as 'not_found' (restart) to be safe, 
    # or 'failed' if we want to debug. 
    # We'll treat it as 'not_found' (restart)
    return 'not_found', None

def main():
    print(f"Project Root: {project_root}")
    print(f"Configuring batch run for {sum(len(v) for v in MODEL_CATEGORIES.values())} models across {len(MODEL_CATEGORIES)} categories.")
    
    # Configuration
    config_path = "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml"
    abs_config_path = project_root / config_path
    
    if not abs_config_path.exists():
        print(f"Error: Config file not found at {abs_config_path}")
        return

    # Load base config
    try:
        base_cfg = OmegaConf.load(abs_config_path)
    except Exception as e:
        print(f"Error loading base config: {e}")
        return

    # Ensure temp dir exists
    tmp_dir = project_root / "runs" / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run loop
    total_models = sum(len(v) for v in MODEL_CATEGORIES.values())
    current_idx = 0
    
    for category, models in MODEL_CATEGORIES.items():
        print(f"\n{'='*80}")
        print(f"📂 Category: {category}")
        print(f"{'='*80}")
        
        for model_name in models:
            current_idx += 1
            print(f"\n[{current_idx}/{total_models}] Checking model: {model_name} ({category})")
            
            status, path = find_existing_run(model_name)
            
            resume_arg = None
            
            if status == 'completed':
                print(f"✅ Model {model_name} already completed. Skipping.")
                continue
            elif status == 'interrupted':
                print(f"⚠️ Model {model_name} interrupted. Resuming from {path}")
                resume_arg = str(path)
            else:
                print(f"🆕 Model {model_name} starting from scratch.")
            
            print(f"{'-'*60}")
            
            # Prepare overrides
            overrides = {
                "model": {"name": model_name},
                "model_budget": {
                    "target_params_m": 10.0,
                    "auto_tune": True
                },
                "experiment": {
                    "name": f"AR-ShallowWater-10M-{model_name}"
                },
                "ar": {"enabled": False},
                "training": {"torch_compile": False}
            }
            
            # Merge config
            cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
            
            # Save temp config
            tmp_cfg_path = tmp_dir / f"batch_run_{model_name}.yaml"
            with open(tmp_cfg_path, 'w') as f:
                OmegaConf.save(cfg, f)
            
            print(f"Generated temp config: {tmp_cfg_path}")

            # Get free port
            master_port = get_free_port()

            # Construct command
            # using torchrun for DDP
            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                f"--master_port={master_port}", 
                str(project_root / "tools/training/train_real_data_ar.py"),
                "--config",
                str(tmp_cfg_path)
            ]
            
            if resume_arg:
                cmd.extend(["--resume", resume_arg])
            
            print(f"Command: {' '.join(cmd)}")
            
            try:
                # Run and wait for completion
                # set cwd to project root to ensure relative paths work
                subprocess.run(cmd, cwd=project_root, check=True)
                print(f"\n✅ Successfully finished {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error running {model_name}: {e}")
                # We continue to the next model even if one fails
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user. Stopping batch run.")
                return
                
            # Small pause to ensure resources are released
            time.sleep(5)

if __name__ == "__main__":
    main()
