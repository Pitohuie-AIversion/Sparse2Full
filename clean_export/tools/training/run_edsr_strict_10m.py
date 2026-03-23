#!/usr/bin/env python3
"""
Dedicated script to run EDSR with strict ~10M parameter configuration.
Manual overrides: n_feats=128, n_resblocks=32.
"""

import sys
import os
import subprocess
import time
import socket
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    print(f"Project Root: {project_root}")
    
    # Configuration
    config_path = "thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml"
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

    # Define strict output directory
    output_dir_name = 'runs_drd_paper'
    base_output_dir = project_root / output_dir_name
    tmp_dir = base_output_dir / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_name = "edsr"
    exp_prefix = "AR-DR2D-EDSR-SRx4-10M-Strict"
    model_output_dir = base_output_dir / exp_prefix

    print(f"🚀 Preparing strict 10M run for {model_name}...")
    
    # Overrides for ~10M Params
    # Calculation: 
    # Body = 32 * ( (128*128*9)*2 ) = 32 * 294912 = 9,437,184
    # Head + Tail approx 0.3M
    # Total ~ 9.7M
    overrides = {
        "model": {
            "name": "EDSR",
            "n_feats": 64,
            "n_resblocks": 32,
            "res_scale": 0.1
        },
        "model_budget": {
            "target_params_m": 10.0,
            "auto_tune": False, # Disable auto-tune to enforce strict config
            "strict_mode": False
        },
        "experiment": {
            "name": exp_prefix,
            "output_dir": str(model_output_dir.relative_to(project_root) if model_output_dir.is_relative_to(project_root) else model_output_dir)
        },
        "ar": {"enabled": False},
        "training": {
            "torch_compile": False,
            "oom_recovery": {"enabled": False},
            "smoke_test": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 24,  # 2 * 2(GPUs) * 24 = 96
            "gradient_checkpointing": True
        },
        "device": {
            "devices": 2,
            "memory_management": {
                "gradient_checkpointing": True
            }
        }
    }
    
    # Merge config
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
    
    # Save temp config
    tmp_cfg_path = tmp_dir / f"strict_run_{model_name}.yaml"
    with open(tmp_cfg_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    print(f"Generated temp config: {tmp_cfg_path}")

    # Construct command - DDP with torchrun
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=" + str(get_free_port()),
        str(project_root / "tools/training/train_real_data_ar.py"),
        "--config",
        str(tmp_cfg_path)
    ]
    
    # Use both GPUs
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # env["CUDA_VISIBLE_DEVICES"] = "0,1" # Let system decide or user control via env, but torchrun handles it

    print(f"Command: {' '.join(cmd)} (DDP on 2 GPUs)")
    
    try:
        # Run
        subprocess.run(cmd, cwd=project_root, check=True, env=env)
        print(f"\n✅ Successfully finished {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {model_name}: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")

if __name__ == "__main__":
    main()
