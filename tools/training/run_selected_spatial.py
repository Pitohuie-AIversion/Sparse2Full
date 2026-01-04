#!/usr/bin/env python3
"""
Script to run training for SELECTED strong baseline models with ~10M parameters.
This is a filtered subset of models (Tier 1) for the first round of experiments.
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

# Define Model Categories - Tier 1 Selection
MODEL_CATEGORIES = {
    "Transformer_SOTA": [
        "SwinUNet",       # 论文核心，Hybrid代表
        "Restormer",      # 图像恢复强基线
        "NAFNet",         # 极简高效强基线
        "UformerLite",    # Transformer U-Net代表
    ],
    "CNN_Baseline": [
        "ConvUNetLite",   # 现代化CNN基线
        "UNet",           # 经典基线
    ],
    "Operator_Learning": [
        "FNO2d",          # 物理驱动标准基线
        "UNO",            # U-Net结构算子
    ],
    # "Potential_Strong": [
    #     "SegFormer",    # 语义分割迁移来的强基线
    #     "SwinIRLite",   # SwinIR的轻量版
    # ]
}

def get_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    print(f"Project Root: {project_root}")
    print(f"Configuring batch run for SELECTED {sum(len(v) for v in MODEL_CATEGORIES.values())} strong baselines.")
    
    # Configuration
    config_path = "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml"
    abs_config_path = project_root / config_path
    
    if not abs_config_path.exists():
        print(f"Error: Config file not found at {abs_config_path}")
        # Fallback to the one user mentioned if the above doesn't exist
        fallback_path = "thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml"
        abs_fallback = project_root / fallback_path
        if abs_fallback.exists():
            print(f"Falling back to {fallback_path}")
            abs_config_path = abs_fallback
        else:
            return

    # Load base config
    try:
        base_cfg = OmegaConf.load(abs_config_path)
    except Exception as e:
        print(f"Error loading base config: {e}")
        return

    # Determine Output Directory from config
    base_output_dir = project_root / getattr(base_cfg.experiment, 'output_dir', 'runs')
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
            exp_prefix = f"Spatial-Tier1-10M-{model_name}"
            
            # Helper to find existing run
            def find_run_custom(name_prefix):
                runs_dir = base_output_dir
                candidates = []
                if runs_dir.exists():
                    for d in runs_dir.iterdir():
                        if not d.is_dir(): continue
                        if d.name.startswith(f"{name_prefix}-"):
                            candidates.append(d)
                if not candidates: return 'not_found', None
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
            overrides = {
                "model": {"name": model_name},
                "model_budget": {
                    "target_params_m": 10.0,
                    "auto_tune": True,
                    "strict_mode": True
                },
                "experiment": {
                    "name": exp_prefix,
                    "description": "Tier 1 Spatial Baseline Selection"
                },
                "ar": {"enabled": False}, # Ensure Spatial only
                "training": {
                    "torch_compile": False,
                    "oom_recovery": {"enabled": True},
                },
                # Ensure we use the correct dataset for spatial task
                "data": {
                    "T_in": 1,
                    "T_out": 1
                },
                "device": {"devices": 2}
            }
            
            # Merge config
            cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
            
            # Save temp config
            tmp_cfg_path = tmp_dir / f"batch_run_tier1_{model_name}.yaml"
            with open(tmp_cfg_path, 'w') as f:
                OmegaConf.save(cfg, f)
            
            print(f"Generated temp config: {tmp_cfg_path}")

            # Construct command
            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "--master_port=" + str(get_free_port()),
                str(project_root / "tools/training/train_real_data_ar.py"),
                "--config",
                str(tmp_cfg_path)
            ]
            
            if resume_arg:
                cmd.extend(["--resume", resume_arg])
            
            print(f"Command: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, cwd=project_root, check=True)
                print(f"\n✅ Successfully finished {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error running {model_name}: {e}")
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user. Stopping batch run.")
                return
                
            time.sleep(5)

if __name__ == "__main__":
    main()
