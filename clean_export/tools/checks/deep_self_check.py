import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer
from ops.losses import compute_ar_total_loss

def _denorm_seq(z, stats, keys=None):
    B, T, C, H, W = z.shape
    if keys is None:
        m = stats.get('data_mean', stats.get('u_mean', torch.tensor(0.0)))
        s = stats.get('data_std', stats.get('u_std', torch.tensor(1.0)))
        mean_t = torch.as_tensor(m, device=z.device).reshape(1, 1, 1, 1)
        std_t = torch.as_tensor(s, device=z.device).reshape(1, 1, 1, 1)
        return z * std_t + mean_t
    means = []
    stds = []
    for k in keys:
        means.append(stats.get(f"{k}_mean", torch.tensor(0.0, device=z.device)))
        stds.append(stats.get(f"{k}_std", torch.tensor(1.0, device=z.device)))
    mean_t = torch.stack([torch.as_tensor(m, device=z.device) for m in means]).reshape(1, C, 1, 1)
    std_t = torch.stack([torch.as_tensor(s, device=z.device) for s in stds]).reshape(1, C, 1, 1)
    zf = z.reshape(B*T, C, H, W)
    of = zf * std_t + mean_t
    return of.reshape(B, T, C, H, W)

def run(config_path: str):
    trainer = RealDataARTrainer(config_path=config_path)
    model = trainer.get_model()
    model.eval()
    torch.use_deterministic_algorithms(True)
    val_iter = iter(trainer.train_loader)
    batch = next(val_iter)
    input_seq = batch['input_sequence'].to(trainer.device)
    target_seq = batch['target_sequence'].to(trainer.device)
    current_T_out = trainer.get_current_T_out(0)
    with torch.no_grad():
        if hasattr(model, 'spatial_forward') and hasattr(model, 'temporal_forward'):
            out_tf = model(input_seq, target_seq)
            pred_seq_tf = out_tf['final_pred']
            out_nar = model(input_seq, None)
            pred_seq_nar = out_nar['final_pred']
        else:
            pred_seq_tf = model(input_seq, current_T_out, target_seq)
            pred_seq_nar = model(input_seq, current_T_out)
    keys = None
    try:
        mk = getattr(trainer.config.data, 'keys', None)
        if isinstance(mk, (list, tuple)):
            keys = list(mk)
    except Exception:
        keys = None
    trainer.ensure_norm_stats()
    gt_orig = _denorm_seq(target_seq, trainer.norm_stats, keys)
    B, T, C, H, W = gt_orig.shape
    obs_flat = trainer.observation_op(gt_orig.reshape(B*T, C, H, W)) if trainer.observation_op is not None else None
    observation_seq = None
    if obs_flat is not None:
        oh, ow = obs_flat.shape[-2:]
        observation_seq = obs_flat.reshape(B, T, C, oh, ow)
    # 解析 H 参数（优先使用配置），避免训练器未成功填充 h_params 的情况
    h_params = trainer.h_params
    try:
        obs_cfg = getattr(getattr(trainer.config, 'data', {}), 'observation', {})
        mode_raw = obs_cfg.get('mode', None)
        mode = str(mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw).lower() if mode_raw is not None else None
        if mode == 'sr':
            sr_sub = obs_cfg.get('sr', {}) if isinstance(obs_cfg.get('sr', {}), dict) else {}
            scale = obs_cfg.get('scale_factor', sr_sub.get('scale_factor', 2))
            sigma = obs_cfg.get('blur_sigma', sr_sub.get('blur_sigma', 1.0))
            kernel_size = obs_cfg.get('blur_kernel_size', sr_sub.get('blur_kernel_size', 5))
            boundary = obs_cfg.get('boundary_mode', sr_sub.get('boundary_mode', 'mirror'))
            downsample = obs_cfg.get('downsample_mode', sr_sub.get('downsample_mode', 'area'))
            h_params = {
                'task': 'SR',
                'scale': int(scale),
                'sigma': float(sigma),
                'kernel_size': int(kernel_size),
                'boundary': str(boundary),
                'downsample_interpolation': str(downsample)
            }
        elif mode == 'crop':
            crop_sub = obs_cfg.get('crop', {}) if isinstance(obs_cfg.get('crop', {}), dict) else {}
            crop_size = obs_cfg.get('crop_size', crop_sub.get('crop_size', None))
            crop_box = obs_cfg.get('crop_box', crop_sub.get('crop_box', None))
            boundary = obs_cfg.get('boundary_mode', crop_sub.get('boundary_mode', 'mirror'))
            h_params = {
                'task': 'Crop',
                'crop_size': crop_size,
                'crop_box': crop_box,
                'boundary': str(boundary)
            }
    except Exception:
        pass
    obs_data = {'observation_seq': observation_seq, 'baseline_seq': input_seq, 'h_params': h_params}
    cfg2 = OmegaConf.create(OmegaConf.to_container(trainer.config, resolve=True))
    if not hasattr(cfg2, 'loss'):
        cfg2.loss = OmegaConf.create({})
    if not hasattr(cfg2.loss, 'spectral'):
        cfg2.loss.spectral = OmegaConf.create({})
    if not hasattr(cfg2.loss, 'data_consistency'):
        cfg2.loss.data_consistency = OmegaConf.create({})
    cfg2.loss.spectral.weight = 1.0
    cfg2.loss.data_consistency.weight = 1.0
    losses_tf = compute_ar_total_loss(pred_seq_tf, target_seq, obs_data, trainer.norm_stats, cfg2)
    losses_nar = compute_ar_total_loss(pred_seq_nar, target_seq, obs_data, trainer.norm_stats, cfg2)
    dc_equiv_mse = None
    if observation_seq is not None and h_params is not None:
        from ops.degradation import apply_degradation_operator
        h_gt = apply_degradation_operator(gt_orig.reshape(B*T, C, H, W), h_params)
        y = observation_seq.reshape(B*T, C, observation_seq.shape[-2], observation_seq.shape[-1])
        dc_equiv_mse = torch.mean((h_gt - y) ** 2).item()
    det_diff = None
    with torch.no_grad():
        if hasattr(model, 'spatial_forward') and hasattr(model, 'temporal_forward'):
            out1 = model(input_seq, None)['final_pred']
            out2 = model(input_seq, None)['final_pred']
        else:
            out1 = model(input_seq, current_T_out)
            out2 = model(input_seq, current_T_out)
        det_diff = torch.max(torch.abs(out1 - out2)).item()
    has_nan = bool(torch.isnan(pred_seq_tf).any().item() or torch.isnan(pred_seq_nar).any().item())
    has_inf = bool(torch.isinf(pred_seq_tf).any().item() or torch.isinf(pred_seq_nar).any().item())
    res = {
        'dc_equivalence_mse': float(dc_equiv_mse) if dc_equiv_mse is not None else None,
        'determinism_max_abs_diff': float(det_diff) if det_diff is not None else None,
        'losses_tf_total': float(losses_tf['total_loss'].item()),
        'losses_nar_total': float(losses_nar['total_loss'].item()),
        'losses_tf_dc': float(losses_tf['dc_loss'].item()),
        'losses_tf_spectral': float(losses_tf['spectral_loss'].item()),
        'losses_nar_dc': float(losses_nar['dc_loss'].item()),
        'losses_nar_spectral': float(losses_nar['spectral_loss'].item()),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'T_out': int(current_T_out)
    }
    outdir = trainer.output_dir if hasattr(trainer, 'output_dir') else Path('runs')
    outpath = Path(outdir) / 'deep_self_check.json'
    with open(outpath, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    run(args.config)
