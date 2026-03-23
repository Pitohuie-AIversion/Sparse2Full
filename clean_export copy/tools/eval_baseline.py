#!/usr/bin/env python3
"""
Baseline Evaluation Script for SR (Bilinear/Bicubic)
Executes interpolation-based super-resolution on the validation set and reports metrics.
"""

import os
import sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
from ops.degradation import apply_degradation_operator
from utils.metrics import compute_all_metrics, StatisticalAnalyzer
from utils.ar_visualizer import ARTrainingVisualizer

def evaluate_baseline(config_path, mode_override=None):
    print(f"🚀 Loading configuration from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Ensure device is set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Using device: {device}")

    # Setup DataModule
    print("📂 Setting up DataModule...")
    # Fix potential config structure issues for DataModule
    if not hasattr(config, 'hardware'):
        config.hardware = OmegaConf.create({'num_workers': 4, 'pin_memory': True})
    if not hasattr(config, 'training'):
         config.training = OmegaConf.create({'batch_size': 32})
    if not hasattr(config, 'testing'):
         config.testing = OmegaConf.create({'batch_size': 32})

    data_module = RealDiffusionReactionDataModule(config)
    data_module.setup(stage='fit')
    val_loader = data_module.val_dataloader()

    # Analyzer for metrics
    analyzer = StatisticalAnalyzer()
    
    # Interpolation mode for baseline (upsampling)
    upsample_mode = "bilinear" 
    if mode_override:
        upsample_mode = mode_override
    elif "bicubic" in str(config_path).lower():
        upsample_mode = "bicubic"
    
    print(f"📈 Baseline Upsample Mode: {upsample_mode}")
    
    # Visualizer
    viz_dir = Path("runs_baseline") / f"viz_{upsample_mode}"
    visualizer = ARTrainingVisualizer(str(viz_dir))

    # Degradation parameters from config
    obs_config = config.data.observation
    degradation_params = {
        "task": obs_config.get("mode", "SR"),
        "scale": obs_config.get("sr", {}).get("scale_factor", 4),
        "sigma": obs_config.get("sr", {}).get("blur_sigma", 1.0),
        "kernel_size": obs_config.get("sr", {}).get("blur_kernel_size", 5),
        "boundary": obs_config.get("sr", {}).get("boundary_mode", "mirror"),
        "downsample_mode": obs_config.get("sr", {}).get("downsample_mode", "area"),
    }
    
    print(f"📉 Degradation Params: {degradation_params}")
    
    print("🔄 Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Get HR target [B, T, C, H, W] -> take first frame or all frames?
            # Config says T_in=1, T_out=1.
            # Dataset returns 'target_sequence' [B, T_out, C, H, W]
            # We usually evaluate on the target sequence.
            
            targets = batch['target_sequence'].to(device) # [B, T, C, H, W]
            
            # Iterate over time steps or treat B*T as batch
            B, T, C, H, W = targets.shape
            
            # Flatten B and T for processing
            targets_flat = targets.view(-1, C, H, W)
            
            # 1. Apply Degradation (HR -> LR)
            # We need to construct params for apply_degradation_operator
            # It expects a dictionary.
            lr_imgs = apply_degradation_operator(targets_flat, degradation_params)
            
            # 2. Apply Baseline Upsampling (LR -> SR)
            # F.interpolate requires [N, C, H, W]
            sr_imgs = F.interpolate(
                lr_imgs, 
                size=(H, W), 
                mode=upsample_mode, 
                align_corners=False
            )
            
            # 3. Compute Metrics
            # obs_data for DC error calculation
            # We need to provide the 'y' (observation) which is lr_imgs
            obs_data = degradation_params.copy()
            obs_data['y'] = lr_imgs
            
            # Get normalization stats if available
            norm_stats = data_module.get_normalization_stats()
            if norm_stats:
                # Move to device
                norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}

            batch_metrics = compute_all_metrics(
                pred=sr_imgs,
                target=targets_flat,
                obs_data=obs_data,
                norm_stats=norm_stats,
                include_freq_metrics=True
            )
            
            analyzer.add_result(batch_metrics)
            
            # Visualization using ARTrainingVisualizer
            if batch_idx == 0:
                # Visualize first batch samples
                # We iterate through samples in the batch to generate individual visualizations if needed,
                # or pass the whole batch if the visualizer handles it (it usually takes [B, ...] or [T, ...])
                # ARTrainingVisualizer.visualize_obs_gt_pred_error takes [T, C, H, W] or [B, C, H, W]
                # It visualizes ONE sample (the first one in the tensor).
                
                # Let's visualize a few samples from the first batch
                num_viz = min(5, B)
                for i in range(num_viz):
                    # Prepare single sample tensors: [1, C, H, W] (treated as T=1 sequence) or [C, H, W]
                    # targets_flat is [B, C, H, W]
                    tgt_sample = targets_flat[i].unsqueeze(0) # [1, C, H, W]
                    sr_sample = sr_imgs[i].unsqueeze(0)       # [1, C, H, W]
                    lr_sample = lr_imgs[i].unsqueeze(0)       # [1, C, H, W]
                    
                    visualizer.visualize_obs_gt_pred_error(
                        target_seq=tgt_sample,
                        pred_seq=sr_sample,
                        observation_seq=lr_sample,
                        save_name=f"sample_{i:03d}_obs_gt_pred_error",
                        norm_stats=norm_stats,
                        timestep_idx=0
                    )
                    
                    # Also create temporal analysis (though T=1, it still shows energy/spectrum)
                    visualizer.create_temporal_analysis(
                        pred_seq=sr_sample,
                        target_seq=tgt_sample,
                        save_name=f"sample_{i:03d}_temporal_analysis",
                        norm_stats=norm_stats
                    )


    # Report results
    print("\n📊 Evaluation Results:")
    stats = analyzer.compute_statistics()
    
    # Print formatted table
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 46)
    for name, vals in stats.items():
        print(f"{name:<20} | {vals['mean']:.6f} | {vals['std']:.6f}")
    
    # Save to file
    output_dir = Path("runs_baseline")
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"baseline_{upsample_mode}_metrics.txt"
    with open(result_file, "w") as f:
        f.write(f"Config: {config_path}\n")
        f.write(f"Upsample Mode: {upsample_mode}\n")
        f.write(f"Degradation: {degradation_params}\n")
        f.write("-" * 46 + "\n")
        f.write(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}\n")
        f.write("-" * 46 + "\n")
        for name, vals in stats.items():
            f.write(f"{name:<20} | {vals['mean']:.6f} | {vals['std']:.6f}\n")
    
    print(f"\n✅ Results saved to {result_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config file")
    parser.add_argument("--mode", default=None, help="Interpolation mode (bilinear/bicubic)")
    args = parser.parse_args()
    
    evaluate_baseline(args.config_path, args.mode)
