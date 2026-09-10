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
    data_module.setup(stage='test')
    
    # 因为用户想要0059等训练集样本，所以我们在训练集上也跑一下或者直接遍历所有Loader
    loaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader()
    }

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
        for split_name, dataloader in loaders.items():
            print(f"--- Processing {split_name} loader ---")
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
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
                if norm_stats is not None:
                    # Move to device
                    norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}
                else:
                    print("Warning: norm_stats is None")
    
                batch_metrics = compute_all_metrics(
                    pred=sr_imgs,
                    target=targets_flat,
                    obs_data=obs_data,
                    norm_stats=norm_stats,
                    include_freq_metrics=True
                )
                
                # 仅在 val 集上记录指标
                if split_name == "val":
                    analyzer.add_result(batch_metrics)
                
                # Visualization using ARTrainingVisualizer
                # 我们直接在循环里把所有想要的 sample_id 的结果强制保存出来
                target_sample_keys = ['0800', '0813', '0859', '0959', '0059', '0013'] # 尝试一些测试集的ID
                
                if 'sample_key' in batch:
                    sample_keys = batch['sample_key']
                    # sample_keys might be tensor or list of str
                    if isinstance(sample_keys, torch.Tensor):
                        sample_keys_str = [f"{int(k):04d}" for k in sample_keys.cpu().numpy()]
                    else:
                        sample_keys_str = [str(k).zfill(4) for k in sample_keys]
                    
                    # We should check all elements in the batch
                    for batch_idx_local, key in enumerate(sample_keys_str):
                        if key in target_sample_keys:
                            # 对于每个匹配的样本，我们无论它的 start_time 是多少，
                            # 只要是第一次遇到，就保存可视化，或者加上 start_time 的后缀避免覆盖
                            start_time = int(batch['start_time'][batch_idx_local].item())
                            
                            tgt_seq = targets[batch_idx_local].unsqueeze(0) # [1, T, C, H, W]
                            sr_seq = sr_imgs.view(B, T, C, H, W)[batch_idx_local].unsqueeze(0)
                            H_lr, W_lr = lr_imgs.shape[-2], lr_imgs.shape[-1]
                            lr_seq = lr_imgs.view(B, T, C, H_lr, W_lr)[batch_idx_local].unsqueeze(0)
                            
                            # Just visualize t=0 for simplicity
                            # Use plot_obs_gt_pred_err_horizontal if you want exactly the same look as train_real_data_ar.py
                            # Because train_real_data_ar.py might use this horizontal version.
                            visualizer.plot_obs_gt_pred_err_horizontal(
                                obs=lr_seq[:, 0], # Take first time step
                                gt=tgt_seq[:, 0],
                                pred=sr_seq[:, 0],
                                save_path=str(visualizer.vis_dir / "predictions" / f"sample_{key}_st{start_time}_horizontal_t0.png")
                            )
                            
                            visualizer.visualize_obs_gt_pred_error(
                                target_seq=tgt_seq,
                                pred_seq=sr_seq,
                                observation_seq=lr_seq,
                                save_name=f"sample_{key}_st{start_time}_obs_gt_pred_error",
                                norm_stats=norm_stats,
                                timestep_idx=0
                            )
                            print(f"🎯 成功生成目标样本 {key} (start_time={start_time}) 的可视化！")
            
            # 同时保留原来的可视化前几个样本的逻辑（可选，为了兼容性保留）
            if batch_idx == 0:
                num_viz = min(5, B)
                for i in range(num_viz):
                    # 获取真实的 sample_idx (从 batch 中)
                    # batch['sample_idx'] 包含了这个 batch 中每个样本在数据集中的索引
                    real_sample_idx = batch['sample_idx'][i].item() if 'sample_idx' in batch else i
                    
                    # Prepare single sample tensors: [1, C, H, W] (treated as T=1 sequence) or [C, H, W]
                    # targets_flat is [B, C, H, W]
                    tgt_sample = targets_flat[i].unsqueeze(0) # [1, C, H, W]
                    sr_sample = sr_imgs[i].unsqueeze(0)       # [1, C, H, W]
                    lr_sample = lr_imgs[i].unsqueeze(0)       # [1, C, H, W]
                    
                    visualizer.visualize_obs_gt_pred_error(
                        target_seq=tgt_sample,
                        pred_seq=sr_sample,
                        observation_seq=lr_sample,
                        save_name=f"sample_{real_sample_idx:04d}_obs_gt_pred_error",
                        norm_stats=norm_stats,
                        timestep_idx=0
                    )
                    
                    # Also create temporal analysis (though T=1, it still shows energy/spectrum)
                    visualizer.create_temporal_analysis(
                        pred_seq=sr_sample,
                        target_seq=tgt_sample,
                        save_name=f"sample_{real_sample_idx:04d}_temporal_analysis",
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
