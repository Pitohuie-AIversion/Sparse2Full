import os
import sys
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from omegaconf import OmegaConf

# 保障本地项目优先于第三方同名包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from datasets.pdebench import PDEBenchDataModule
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("datasets.pdebench", str(ROOT / "datasets" / "pdebench.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    PDEBenchDataModule = mod.PDEBenchDataModule

from models.spatial.swin_t_with_encoder import SwinTWithEncoder
from utils.checkpoint import load_checkpoint
from utils.metrics import MetricsCalculator
from utils.visualization import ARVisualizer


def main(run_dir: str,
         encoder_out_channels: int = 4,
         post_conv3x3: bool = True,
         device: Optional[str] = None,
         max_samples: int = 64,
         override_window_size: Optional[int] = None) -> None:
    run_path = Path(run_dir)
    cfg_path = run_path / 'config_merged.yaml'
    ckpt_path = run_path / 'best.ckpt'
    assert cfg_path.exists(), f"config_merged.yaml not found in {run_dir}"
    assert ckpt_path.exists(), f"best.ckpt not found in {run_dir}"

    config = OmegaConf.load(str(cfg_path))
    out_dir = run_path / 'eval_postconv_head'
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device or (config.experiment.get('device', 'cuda')
                                  if torch.cuda.is_available() else 'cpu'))

    dm = PDEBenchDataModule(config.data)
    dm.setup()
    test_loader = dm.test_dataloader()
    norm_stats = dm.get_norm_stats()

    ws = override_window_size if override_window_size is not None else config.model.get('window_size', 8)
    def build_model(window_size: int):
        return SwinTWithEncoder(
            in_channels=config.model.get('in_channels', 4),
            out_channels=config.model.get('out_channels', 1),
            img_size=config.model.get('img_size', 128),
            encoder_out_channels=encoder_out_channels,
            embed_dim=config.model.get('embed_dim', 96),
            depths=tuple(config.model.get('depths', [2,2,6,2])),
            num_heads=tuple(config.model.get('num_heads', [3,6,12,24])),
            patch_size=config.model.get('patch_size', 4),
            window_size=window_size,
            drop_rate=config.model.get('drop_rate', 0.0),
            attn_drop_rate=config.model.get('attn_drop_rate', 0.0),
            drop_path_rate=config.model.get('drop_path_rate', 0.1),
            norm_layer=config.model.get('norm_layer', 'LayerNorm'),
            ape=config.model.get('ape', False),
            patch_norm=config.model.get('patch_norm', True),
            use_checkpoint=config.model.get('use_checkpoint', False),
            final_upsample=config.model.get('final_upsample', 'expand_first'),
            mlp_ratio=config.model.get('mlp_ratio', 4.0),
            qkv_bias=config.model.get('qkv_bias', True),
            qk_scale=config.model.get('qk_scale', None),
            post_conv3x3=post_conv3x3,
        )
    model = build_model(ws)
    model = model.to(dev)
    model.eval()

    ckpt = load_checkpoint(str(ckpt_path), dev)
    sd = ckpt.get('model_state_dict', ckpt)
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith('module.') else k
        new_sd[nk] = v
    # 仅加载形状匹配的权重，避免大小不一致报错
    model_sd = model.state_dict()
    filtered_sd = {}
    for k, v in new_sd.items():
        if (k in model_sd) and (tuple(model_sd[k].shape) == tuple(v.shape)):
            filtered_sd[k] = v
    try:
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    except RuntimeError:
        # 可能是window_size不匹配，回退到4
        if ws != 4:
            model = build_model(4).to(dev)
            model_sd = model.state_dict()
            filtered_sd = {k: v for k, v in new_sd.items() if (k in model_sd) and (tuple(model_sd[k].shape) == tuple(v.shape))}
            missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        else:
            raise

    image_size = (config.data.get('img_size', 128), config.data.get('img_size', 128))
    metrics = MetricsCalculator(image_size=image_size, boundary_width=16)

    visualizer = ARVisualizer(str(out_dir))

    results = []
    with torch.no_grad():
        count = 0
        for bidx, batch in enumerate(test_loader):
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = model(batch['baseline'])
            rel_l2 = metrics.compute_rel_l2(pred, batch['target']).mean().item()
            mae = metrics.compute_mae(pred, batch['target']).mean().item()
            psnr = metrics.compute_psnr(pred, batch['target']).mean().item()
            results.append({'rel_l2': rel_l2, 'mae': mae, 'psnr': psnr})

            # 保存少量可视化样本
            if bidx < 3:
                save_dir = out_dir / f'batch_{bidx:04d}'
                save_dir.mkdir(parents=True, exist_ok=True)
                # 统一色标四连图（观测/真实/预测/误差）
                visualizer.plot_obs_gt_pred_err_horizontal(
                    observation=batch['baseline'],
                    targets=batch['target'],
                    predictions=pred,
                    save_path=str(save_dir / 'obs_gt_pred_err.svg'),
                    num_samples=min(4, pred.shape[0])
                )

            count += pred.shape[0]
            if count >= max_samples:
                break

    if results:
        avg = {
            'rel_l2': float(np.mean([r['rel_l2'] for r in results])),
            'mae': float(np.mean([r['mae'] for r in results])),
            'psnr': float(np.mean([r['psnr'] for r in results])),
            'samples': int(count)
        }
    else:
        avg = {}

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(avg, f, indent=2)

    print(json.dumps(avg, indent=2))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, required=True)
    ap.add_argument('--encoder_out_channels', type=int, default=4)
    ap.add_argument('--post_conv3x3', action='store_true')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--max_samples', type=int, default=64)
    ap.add_argument('--override_window_size', type=int, default=None)
    args = ap.parse_args()
    main(args.run_dir, args.encoder_out_channels, args.post_conv3x3, args.device, args.max_samples, args.override_window_size)
