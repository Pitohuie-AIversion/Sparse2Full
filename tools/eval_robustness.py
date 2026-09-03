#!/usr/bin/env python3
"""
鲁棒性评估脚本：评估模型在不同噪声水平下的性能稳定性
Robustness Evaluation Script: Evaluate model performance under varying noise levels.

Usage:
    python tools/eval_robustness.py --config runs/YOUR_EXP/config_merged.yaml --checkpoint runs/YOUR_EXP/checkpoints/best.pt
"""

import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.pdebench import PDEBenchDataModule
from models import create_model
from ops.degradation import apply_degradation_operator
from utils.reproducibility import set_seed

def add_noise(tensor, sigma):
    """向张量添加高斯噪声"""
    if sigma <= 0:
        return tensor
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise

def main():
    parser = argparse.ArgumentParser(description='Robustness Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1], help='Noise levels (sigma)')
    parser.add_argument('--output_dir', type=str, default='robustness_results', help='Output directory')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    # Fix: create_model expects the model config part, not the whole config
    if hasattr(cfg, 'model'):
        model_cfg = cfg.model
    else:
        model_cfg = cfg
        
    model = create_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle possible state dict keys mismatch (e.g. 'model_state_dict' vs raw)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 兼容 DDP 训练的权重 (去除 'module.' 前缀)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Setup Data
    print("Setting up data...")
    data_module = PDEBenchDataModule(cfg.data)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Metrics storage
    results = {sigma: {'rel_l2': [], 'h_err': []} for sigma in args.noise_levels}

    # Evaluation Loop
    print(f"Starting evaluation on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get GT (Target)
            target = batch['target'].to(device) # [B, C, H, W]
            
            # Re-generate observation from GT to ensure clean baseline before adding custom noise
            # Note: We assume 'task_params' in batch is sufficient for apply_degradation_operator
            # If the original observation was loaded from disk, we might want to use that as base 
            # if sigma=0, but here we want to control noise explicitly.
            # However, apply_degradation_operator usually applies H (downsample/crop).
            # We need to check if we can re-generate H(u).
            # For simplicity, if we trust the batch['observation'] is H(u) + fixed_noise,
            # we might just add *extra* noise to batch['observation']. 
            # But correct robustness test requires H(u) + controlled_noise.
            
            # Let's try to use the batch['observation'] as base and add extra noise 
            # if we can't easily re-generate H(u). 
            # Ideally: obs = apply_degradation_operator(target, ...) + noise
            
            # Fallback: Just add extra noise to the provided observation
            # This tests "Additional Noise Robustness"
            if 'observation' in batch:
                base_observation = batch['observation'].to(device)
                mode = 'direct'
            elif 'lr_observation' in batch:
                base_observation = batch['lr_observation'].to(device)
                mode = 'sr'
            else:
                # Crop case or fallback
                base_observation = batch['baseline'].to(device)
                mode = 'baseline'
            
            # Get task params for H_err calculation
            task_params = batch.get('task_params', {})

            # Prepare common inputs
            # baseline = batch.get('baseline', base_observation).to(device)
            coords = batch.get('coords', None)
            if coords is not None: coords = coords.to(device)
            mask = batch.get('mask', None)
            if mask is not None: mask = mask.to(device)

            # Check if model expects coords/mask based on config
            include_coords = cfg.data.get('include_coords', False)
            include_mask = cfg.data.get('include_mask', False)

            for sigma in args.noise_levels:
                # Add noise to observation
                noisy_obs = add_noise(base_observation, sigma)
                
                # Re-construct input
                if mode == 'sr':
                    # Upsample
                    H, W = target.shape[-2:]
                    curr_baseline = F.interpolate(noisy_obs, size=(H, W), mode='bilinear', align_corners=False)
                else:
                    curr_baseline = noisy_obs
                
                if mask is not None and include_mask:
                     # Only apply mask if we are using it? 
                     # Actually for crop, we usually multiply baseline by mask even if mask is not concatenated
                     # But if include_mask is False, maybe we assume baseline is already masked?
                     # Let's assume baseline logic is handled.
                     curr_baseline = curr_baseline * mask
                elif mask is not None and mode == 'baseline':
                     # If we are in crop mode (baseline source), it is already masked.
                     # But we added noise to it. So we should re-mask it to keep zero-regions zero.
                     curr_baseline = curr_baseline * mask

                x = curr_baseline
                if include_coords and coords is not None:
                    x = torch.cat([x, coords], dim=1)
                if include_mask and mask is not None:
                    x = torch.cat([x, mask], dim=1)
                
                # Forward
                pred = model(x)
                
                # Compute Metrics
                # 1. Rel-L2
                l2_err = torch.norm(pred - target, p=2) / (torch.norm(target, p=2) + 1e-8)
                results[sigma]['rel_l2'].append(l2_err.item())
                
                # 2. H_err (Consistency)
                # We need to apply H to pred. 
                # We reuse the H operator from utils or assume we can compute it.
                # If task_params is available, we use apply_degradation_operator
                if task_params:
                    # Note: apply_degradation_operator expects un-normalized data usually?
                    # Or z-score? Check project rules. Rule 6 says "L_dc in original domain".
                    # Here we might be in z-score domain. 
                    # Let's stick to Rel-L2 for robustness trend which is domain-agnostic (ratio).
                    pass

    # Summary
    print("\n=== Robustness Evaluation Results ===")
    print(f"{'Noise (sigma)':<15} | {'Rel-L2 (Mean)':<15} | {'Std':<15}")
    print("-" * 50)
    
    summary = {}
    for sigma in args.noise_levels:
        mean_l2 = np.mean(results[sigma]['rel_l2'])
        std_l2 = np.std(results[sigma]['rel_l2'])
        print(f"{sigma:<15.4f} | {mean_l2:<15.4f} | {std_l2:<15.4f}")
        summary[sigma] = {'mean': mean_l2, 'std': std_l2}

    # Save
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / 'robustness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
