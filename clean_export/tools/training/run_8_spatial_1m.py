#!/usr/bin/env python3
"""
Script to run training for classified spatial models with ~1M parameters on Diffusion-Reaction 2D dataset.
Uses dual GPU execution (DDP) via torchrun and specific configuration.
Supports RESUMING from interrupted runs.
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

from tools.training.model_loader import list_models

# Define Target Models
MODEL_CATEGORIES = {
    "Target": ["nafnet", "UformerLite", "uno", "ConvUNetLite", "UNet", "stablefno2d", "RestormerLite"]
}

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    print(f"Project Root: {project_root}")
    print(f"Configuring batch run for {sum(len(v) for v in MODEL_CATEGORIES.values())} models across {len(MODEL_CATEGORIES)} categories.")
    
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

    # Determine Base Output Directory
    # We enforce 'drd_paper_1m' as the root for this experiment suite
    output_dir_name = 'drd_paper_1m'
        
    base_output_dir = project_root / output_dir_name
    print(f"Base Output Directory: {base_output_dir}")

    # Ensure temp dir exists
    tmp_dir = base_output_dir / "tmp_configs"
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
            
            # Update prefix for experiment check
            # Format: AR-DR2D-{model_name}-SRx4-1M-100ep
            exp_prefix = f"AR-DR2D-{model_name}-SRx4-1M-100ep"
            
            # Helper to find existing run with new prefix
            def find_run_custom(name_prefix):
                runs_dir = base_output_dir
                candidates = []
                if runs_dir.exists():
                    for d in runs_dir.iterdir():
                        if not d.is_dir(): continue
                        # Exact match for strict folder naming
                        if d.name == name_prefix:
                             candidates.append(d)
                
                if not candidates: return 'not_found', None
                # If multiple (shouldn't be with strict naming), take latest
                candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest = candidates[0]
                
                if (latest / "test_results.json").exists(): return 'completed', latest
                ckpt = latest / "last.ckpt"
                if ckpt.exists(): return 'interrupted', ckpt
                ckpt = latest / "checkpoints" / "last.ckpt"
                if ckpt.exists(): return 'interrupted', ckpt
                return 'not_found', None

            status, path = find_run_custom(exp_prefix)
            
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
            # Construct specific output dir for this model
            model_output_dir = base_output_dir / exp_prefix
            
            overrides = {
                "model": {"name": model_name},
                "model_budget": {
                    "target_params_m": 1.0,
                    "auto_tune": True,
                    "strict_mode": False
                },
                "experiment": {
                    "name": exp_prefix,
                    # Ensure output_dir is passed correctly as the FULL path for this specific run
                    "output_dir": str(model_output_dir.relative_to(project_root) if model_output_dir.is_relative_to(project_root) else model_output_dir)
                },
                "ar": {"enabled": False},
                "training": {
                    "epochs": 100,
                    "torch_compile": False,
                    "oom_recovery": {"enabled": True},
                    "smoke_test": False
                },
                "device": {"devices": 2}
            }
            
            # Merge config
            cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
            
            # Save temp config
            tmp_cfg_path = tmp_dir / f"batch_run_{model_name}.yaml"
            with open(tmp_cfg_path, 'w') as f:
                OmegaConf.save(cfg, f)
            
            print(f"Generated temp config: {tmp_cfg_path}")

            # Construct command - Use torchrun for dual GPU DDP
            port = get_free_port()
            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "--master_port", str(port),
                str(project_root / "tools/training/train_real_data_ar.py"),
                "--config",
                str(tmp_cfg_path)
            ]
            
            if resume_arg:
                cmd.extend(["--resume", resume_arg])
            
            # Prepare environment with CUDA_VISIBLE_DEVICES=0,1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0,1"

            print(f"Command: {' '.join(cmd)} (on CUDA:0,1)")
            
            try:
                # Run and wait for completion
                # set cwd to project root to ensure relative paths work
                subprocess.run(cmd, cwd=project_root, check=True, env=env)
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
