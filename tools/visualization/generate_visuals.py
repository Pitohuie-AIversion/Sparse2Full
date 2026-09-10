#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=8)
    args = parser.parse_args()

    exp_name = args.exp_name
    run_dir = Path("runs") / exp_name
    pkg_dir = Path("paper_package/figs") / exp_name
    viz_run = run_dir / "visualizations"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    viz_run.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config_merged.yaml"
    if not cfg_path.exists():
        cfg_path = Path("configs/train/ar_training_config debug.yaml")
    cfg = OmegaConf.load(str(cfg_path))

    # Try RealDiffusionReactionDataModule first, fallback to PDEBenchDataModule
    try:
        from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
        dm = RealDiffusionReactionDataModule(cfg)
    except Exception:
        from datasets.pdebench import PDEBenchDataModule
        dm = PDEBenchDataModule(cfg.data)
    dm.setup("fit")
    loader = dm.val_dataloader()

    from models import create_model
    mcfg = cfg.model
    model = create_model(
        mcfg.name,
        in_channels=mcfg.in_channels,
        out_channels=mcfg.out_channels,
        img_size=mcfg.img_size,
        patch_size=mcfg.patch_size,
        window_size=mcfg.window_size,
        depths=list(mcfg.depths),
        num_heads=list(mcfg.num_heads),
        embed_dim=mcfg.embed_dim,
        mlp_ratio=mcfg.mlp_ratio,
        drop_rate=mcfg.drop_rate,
        attn_drop_rate=mcfg.attn_drop_rate,
        drop_path_rate=mcfg.drop_path_rate,
    )
    # Load checkpoint if available
    ckpt = None
    for name in ["best.ckpt", "last.ckpt"]:
        p = run_dir / name
        if p.exists():
            ckpt = p
            break
    if ckpt is not None:
        state = torch.load(str(ckpt), map_location="cpu")
        sd = state.get("model_state_dict", {})
        sd = strip_module_prefix(sd)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch = next(iter(loader))
    lr_obs = None
    if "input_sequence" in batch and "target_sequence" in batch:
        input_seq = batch["input_sequence"].to(device)  # [B,T_in,C,H,W]
        target_seq = batch["target_sequence"].to(device)  # [B,T_out,C,H,W]
        x = input_seq[:, 0, 0:1]
        gt = target_seq[:, 0, 0:1]
        if "original_observation" in batch:
            lr_obs = batch["original_observation"].to(device)[:, 0:1]
        elif "lr_observation" in batch:
            lr_obs = batch["lr_observation"].to(device)[:, 0:1]
    else:
        baseline = batch["baseline"].to(device)  # [B,1,H,W]
        coords = batch["coords"].to(device)      # [B,2,H,W]
        mask = batch["mask"].to(device)          # [B,1,H,W]
        x_in = torch.cat([baseline, coords, mask], dim=1)
        x = baseline
        gt = batch["target"].to(device)[:, 0:1]
        if "original_observation" in batch:
            lr_obs = batch["original_observation"].to(device)[:, 0:1]
        elif "lr_observation" in batch:
            lr_obs = batch["lr_observation"].to(device)[:, 0:1]
    B, _, H, W = x.shape
    if "input_sequence" in batch:
        y_lin = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
        x_lin = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)
        coords = torch.stack([x_lin, y_lin], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        mask = torch.ones((B, 1, H, W), device=device)
        x_in = torch.cat([x, coords, mask], dim=1)
    with torch.no_grad():
        y = model(x_in)
    err = (y - gt).abs()

    n = min(B, args.max_samples)
    svg_names = []
    for b in range(n):
        obs_img = x[b, 0].detach().cpu().numpy()
        lr_up_img = None
        if lr_obs is not None:
            lr_up = F.interpolate(lr_obs[b:b+1], size=(H, W), mode="bilinear", align_corners=False)
            lr_up_img = lr_up[0, 0].detach().cpu().numpy()
        gt_img = gt[b, 0].detach().cpu().numpy()
        pr_img = y[b, 0].detach().cpu().numpy()
        er_img = err[b, 0].detach().cpu().numpy()
        vmin = float(min(np.min(obs_img), np.min(gt_img), np.min(pr_img)))
        vmax = float(max(np.max(obs_img), np.max(gt_img), np.max(pr_img)))
        cols = 5 if lr_up_img is not None else 4
        fig, axes = plt.subplots(1, cols, figsize=(15 if cols==5 else 12, 3))
        col_idx = 0
        if lr_up_img is not None:
            im_lr = axes[col_idx].imshow(lr_up_img, cmap="viridis")
            axes[col_idx].set_title("LR-Obs(raw↑)")
            plt.colorbar(im_lr, ax=axes[col_idx], fraction=0.046, pad=0.04)
            col_idx += 1
        im0 = axes[col_idx].imshow(obs_img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[col_idx].set_title("Obs")
        plt.colorbar(im0, ax=axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1
        im1 = axes[col_idx].imshow(gt_img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[col_idx].set_title("GT")
        plt.colorbar(im1, ax=axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1
        im2 = axes[col_idx].imshow(pr_img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[col_idx].set_title("Pred")
        plt.colorbar(im2, ax=axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1
        im3 = axes[col_idx].imshow(er_img, cmap="magma")
        axes[col_idx].set_title("Err")
        plt.colorbar(im3, ax=axes[col_idx], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        name_svg = f"epoch_vis_sample_{b:03d}.svg"
        (pkg_dir / name_svg).write_text("") if False else None  # ensure dir exists
        plt.savefig(pkg_dir / name_svg)
        plt.savefig(viz_run / name_svg)
        plt.savefig(pkg_dir / f"epoch_vis_sample_{b:03d}.png", dpi=150)
        plt.savefig(viz_run / f"epoch_vis_sample_{b:03d}.png", dpi=150)
        plt.close(fig)
        svg_names.append(name_svg)

    index_html = pkg_dir / "index.html"
    with open(index_html, "w") as f:
        f.write("<html><body>")
        for name in svg_names:
            f.write(f"<img src='{name}' style='width:800px'><br/>")
        f.write("</body></html>")

    print("Saved visualizations to:", pkg_dir)

if __name__ == "__main__":
    main()
