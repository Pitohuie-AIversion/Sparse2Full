#!/usr/bin/env python3
"""
Script to run training for classified spatial models with ~10M parameters on Shallow Water dataset.
Uses 2 GPUs via DDP.
"""

import sys
import os
import subprocess
import time
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
            print(f"\n[{current_idx}/{total_models}] Starting training for model: {model_name} ({category})")
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

            # Construct command
            # using torchrun for DDP
            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "--master_port=29500",  # Default port
                str(project_root / "tools/training/train_real_data_ar.py"),
                "--config",
                str(tmp_cfg_path)
            ]
            
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
