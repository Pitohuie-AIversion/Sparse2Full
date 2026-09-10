
import os
import sys

# 添加项目根目录到 sys.path 的最前面
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added to sys.path: {project_root}")

import torch
import numpy as np
import json
from pathlib import Path
from omegaconf import OmegaConf
from scipy import stats
import logging

try:
    import datasets
    # print(f"datasets module location: {datasets.__file__}")
except ImportError:
    print("datasets module not found initially")

from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel
from datasets.pdebench import PDEBenchDataModule
from utils.reproducibility import set_seed

def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    
    # 解析配置
    if 'sequential' in cfg:
        spatial_cfg = cfg.sequential.spatial
        temporal_cfg = cfg.sequential.temporal
    else:
        # 尝试从 model 部分读取（如果结构不同）
        if 'model' in cfg and 'spatial' in cfg.model:
             spatial_cfg = cfg.model.spatial
             temporal_cfg = cfg.model.temporal
        else:
             raise ValueError("Config does not contain 'sequential' or 'model.spatial' section")
        
    data_cfg = cfg.data
    
    # 转换为 dict (OmegaConf -> dict)
    spatial_cfg = OmegaConf.to_container(spatial_cfg, resolve=True)
    temporal_cfg = OmegaConf.to_container(temporal_cfg, resolve=True)
    data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    
    # 实例化模型
    model = SequentialSpatiotemporalModel(
        spatial_config=spatial_cfg,
        temporal_config=temporal_cfg,
        data_config=data_cfg,
        device=str(device)
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 处理可能的 module. 前缀（如果用了 DDP）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 加载权重，允许部分不匹配（例如 aux heads）
    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
            
    model.to(device)
    model.eval()
    return model, cfg

def compute_rel_l2(pred, target):
    # [B, C, H, W]
    # 逐样本计算
    diff = pred - target
    # Flatten spatial dims
    diff = diff.view(diff.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    numerator = torch.norm(diff, p=2, dim=1)
    denominator = torch.norm(target_flat, p=2, dim=1) + 1e-8
    
    return (numerator / denominator).cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 路径配置
    project_root = Path(__file__).resolve().parents[1]
    e2e_dir = str(project_root / "runs_drd_paper/AR-DR2D-E2E-StrictStride10-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260122")
    twostage_dir = str(project_root / "runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116")
    
    e2e_ckpt = os.path.join(e2e_dir, "best.ckpt")
    e2e_config = os.path.join(e2e_dir, "config_merged.yaml")
    
    twostage_ckpt = os.path.join(twostage_dir, "best.ckpt")
    twostage_config = os.path.join(twostage_dir, "config_merged.yaml")
    
    print("Loading E2E Model...")
    model_e2e, cfg_e2e = load_model(e2e_config, e2e_ckpt, device)
    
    print("Loading Two-Stage Model...")
    model_twostage, cfg_twostage = load_model(twostage_config, twostage_ckpt, device)
    
    # 准备数据
    # 使用 E2E 的 data config，因为它们应该是可比的（同一个测试集）
    # 只需要验证 cfg_e2e.data 和 cfg_twostage.data 是一致的
    print("Setting up DataModule...")
    data_module = PDEBenchDataModule(cfg_e2e.data)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    rel_l2_e2e = []
    rel_l2_twostage = []
    
    print("Starting evaluation...")
    num_samples = 0
    max_samples = 500  # 限制样本数以加快速度，500足够做统计检验
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if num_samples >= max_samples:
                break
                
            # 准备输入
            target = batch['target'].to(device)
            
            # 尝试获取 observation
            if 'observation' in batch:
                observation = batch['observation'].to(device)
            elif 'lr_observation' in batch:
                observation = batch['lr_observation'].to(device)
            elif 'baseline' in batch:
                observation = batch['baseline'].to(device)
            else:
                raise KeyError(f"No observation/baseline found in batch. Keys: {batch.keys()}")
            
            baseline = batch.get('baseline', observation).to(device)
            coords = batch.get('coords')
            mask = batch.get('mask')
            
            # 构建输入
            model_input = baseline
            # EDSR 模型配置为 in_channels=1，不接受 coords 和 mask
            # if coords is not None:
            #     coords = coords.to(device)
            #     model_input = torch.cat([model_input, coords], dim=1)
            # if mask is not None:
            #     mask = mask.to(device)
            #     model_input = torch.cat([model_input, mask], dim=1)
                
            # E2E 推理
            output_e2e = model_e2e(model_input)
            if isinstance(output_e2e, dict):
                pred_e2e = output_e2e['final_pred']
            else:
                pred_e2e = output_e2e
                
            if batch_idx == 0:
                print(f"Sample 0 Stats:")
                print(f"  Target Range: [{target.min():.4f}, {target.max():.4f}], Mean: {target.mean():.4f}")
                print(f"  Pred E2E Range: [{pred_e2e.min():.4f}, {pred_e2e.max():.4f}], Mean: {pred_e2e.mean():.4f}")
                
            l2_e2e = compute_rel_l2(pred_e2e, target)
            rel_l2_e2e.extend(l2_e2e)
            
            # Two-Stage 推理
            output_twostage = model_twostage(model_input)
            if isinstance(output_twostage, dict):
                pred_twostage = output_twostage['final_pred']
            else:
                pred_twostage = output_twostage
            l2_twostage = compute_rel_l2(pred_twostage, target)
            rel_l2_twostage.extend(l2_twostage)
            
            num_samples += target.size(0)
            if batch_idx % 10 == 0:
                print(f"Processed {num_samples} samples...")

    # 统计检验
    rel_l2_e2e = np.array(rel_l2_e2e)
    rel_l2_twostage = np.array(rel_l2_twostage)
    
    # Paired T-Test
    t_stat, p_value = stats.ttest_rel(rel_l2_twostage, rel_l2_e2e) # Baseline - Ours
    
    # Cohen's d
    diff = rel_l2_twostage - rel_l2_e2e # Positive means Ours is better (lower error)
    cohens_d = np.mean(diff) / np.std(diff)
    
    print("\n" + "="*50)
    print("Statistical Analysis Results")
    print("="*50)
    print(f"Sample size: {len(rel_l2_e2e)}")
    print(f"E2E Mean Rel-L2: {np.mean(rel_l2_e2e):.6f} (std: {np.std(rel_l2_e2e):.6f})")
    print(f"Two-Stage Mean Rel-L2: {np.mean(rel_l2_twostage):.6f} (std: {np.std(rel_l2_twostage):.6f})")
    print(f"Improvement: {(np.mean(rel_l2_twostage) - np.mean(rel_l2_e2e)) / np.mean(rel_l2_twostage) * 100:.2f}%")
    print("-" * 30)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  df: {len(rel_l2_e2e) - 1}")
    print(f"Effect Size:")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print("="*50)
    
    # 保存结果到 json
    results = {
        "e2e_mean": float(np.mean(rel_l2_e2e)),
        "e2e_std": float(np.std(rel_l2_e2e)),
        "twostage_mean": float(np.mean(rel_l2_twostage)),
        "twostage_std": float(np.std(rel_l2_twostage)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": len(rel_l2_e2e) - 1,
        "cohens_d": float(cohens_d),
        "improvement_percent": float((np.mean(rel_l2_twostage) - np.mean(rel_l2_e2e)) / np.mean(rel_l2_twostage) * 100)
    }
    
    with open("comparison_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        print("Results saved to comparison_stats.json")

if __name__ == "__main__":
    main()
