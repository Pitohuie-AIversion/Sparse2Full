#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel
from utils.metrics import compute_metrics

def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dm = RealDiffusionReactionDataModule(cfg)
    dm.setup(stage='fit')
    try:
        val_loader = dm.val_dataloader()
    except Exception:
        val_loader = dm.train_dataloader()

    model = SequentialSpatiotemporalModel(
        spatial_config=cfg.sequential.spatial,
        temporal_config=cfg.sequential.temporal,
        data_config=cfg.data,
        device=str(device)
    ).to(device)
    model.eval()

    batch = next(iter(val_loader))
    x = batch['input_sequence'].to(device)
    y = batch['target_sequence'].to(device)
    T_out = y.shape[1]

    with torch.no_grad():
        pred_step = model.rollout_inference(x, T_out, step_by_step=True)
        pred_one = model.rollout_inference(x, T_out, step_by_step=False)

    # 序列->最后一步指标
    H, W = y.shape[-2], y.shape[-1]
    m_step = compute_metrics(pred_step[:, -1], y[:, -1], image_size=(H, W), include_freq_metrics=False)
    m_one = compute_metrics(pred_one[:, -1], y[:, -1], image_size=(H, W), include_freq_metrics=False)

    print({
        'step_by_step': m_step,
        'one_shot': m_one
    })

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
