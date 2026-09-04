"""
Centralized TensorBoard Logger for Flow Field Prediction & World Models.

Provides unified logging of:
1. Scalars (Losses, Metrics, Learning Rate, System Resources)
2. Spatial Flow Field Grids (Observed Input, Ground Truth, Prediction, Error Maps)
3. Temporal Rollout Strips (Spatiotemporal Evolution over time steps)
4. Residual/Error Histograms & Model Weight/Gradient Distributions
5. Physical Energy Spectrum Plots (Wavenumber vs Power Spectral Density)
6. Custom Dashboard Layouts (add_custom_scalars)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    """Check if the current process is the main process in DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


class TensorBoardLogger:
    """
    Enhanced TensorBoard Logger tailored for PDE & Flow Field Prediction tasks.
    """
    def __init__(
        self,
        log_dir: Union[str, Path],
        enabled: bool = True,
        comment: str = "",
        purge_step: Optional[int] = None
    ):
        self.enabled = enabled and is_main_process()
        self.log_dir = Path(log_dir)
        self.writer: Optional[SummaryWriter] = None

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                comment=comment,
                purge_step=purge_step
            )
            self.setup_custom_layout()
            logger.info(f"[TensorBoardLogger] Initialized SummaryWriter at {self.log_dir}")

    def setup_custom_layout(self) -> None:
        """Organize TensorBoard scalars into structured dashboard tabs."""
        if not self.enabled or self.writer is None:
            return

        layout = {
            "Loss Components": {
                "Losses": ["Multiline", ["train/loss", "val/loss", "epoch_train/total_loss", "epoch_val/total_loss"]],
                "Physics Losses": ["Multiline", ["train/spatial_loss", "train/spectral_loss", "train/dc_loss", "val/spectral_loss"]],
            },
            "Evaluation Metrics": {
                "Relative L2 & MAE": ["Multiline", ["epoch_val/rel_l2", "epoch_val/mae", "epoch_val/rmse"]],
                "PSNR & SSIM": ["Multiline", ["epoch_val/psnr", "epoch_val/ssim"]],
            },
            "System & Optimization": {
                "Learning Rate": ["Multiline", ["train/lr", "epoch_train/lr"]],
                "GPU Memory (GB)": ["Multiline", ["resources/gpu_memory_allocated_gb", "resources/gpu_memory_reserved_gb"]],
            }
        }
        try:
            self.writer.add_custom_scalars(layout)
        except Exception as e:
            logger.debug(f"[TensorBoardLogger] Custom layout setup skipped or failed: {e}")

    def log_scalars(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: int, prefix: str = "") -> None:
        """
        Log dictionary of scalar metrics.
        """
        if not self.enabled or self.writer is None:
            return

        for key, val in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().mean().item() if val.numel() > 1 else val.detach().cpu().item()
            elif isinstance(val, (np.ndarray, np.number)):
                val = float(np.mean(val))
            
            try:
                self.writer.add_scalar(tag, float(val), step)
            except Exception as e:
                logger.warning(f"[TensorBoardLogger] Failed to log scalar {tag}: {e}")

    def _to_numpy_2d(self, tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert a 2D/3D tensor or array to 2D numpy array [H, W]."""
        if isinstance(tensor_or_array, torch.Tensor):
            arr = tensor_or_array.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor_or_array)
        
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]  # Take first channel or slice if multi-dimensional
        return arr

    def log_flow_field_grid(
        self,
        gt_field: Union[torch.Tensor, np.ndarray],
        pred_field: Union[torch.Tensor, np.ndarray],
        step: int,
        tag: str = "FlowField/Comparison",
        input_sparse: Optional[Union[torch.Tensor, np.ndarray]] = None,
        channel_names: Optional[List[str]] = None,
        cmap: str = "viridis"
    ) -> None:
        """
        Log a 4-panel comparison grid: [Sparse Input | Ground Truth | Model Prediction | Error Map].
        """
        if not self.enabled or self.writer is None:
            return

        try:
            gt_img = self._to_numpy_2d(gt_field)
            pred_img = self._to_numpy_2d(pred_field)
            err_img = np.abs(pred_img - gt_img)

            num_cols = 4 if input_sparse is not None else 3
            fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 3.5), squeeze=False)
            axes = axes[0]

            col_idx = 0
            if input_sparse is not None:
                inp_img = self._to_numpy_2d(input_sparse)
                im0 = axes[col_idx].imshow(inp_img, cmap=cmap, origin='lower')
                axes[col_idx].set_title("Sparse Input")
                plt.colorbar(im0, ax=axes[col_idx], fraction=0.046, pad=0.04)
                col_idx += 1

            vmin = min(gt_img.min(), pred_img.min())
            vmax = max(gt_img.max(), pred_img.max())

            im1 = axes[col_idx].imshow(gt_img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[col_idx].set_title("Ground Truth")
            plt.colorbar(im1, ax=axes[col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            im2 = axes[col_idx].imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[col_idx].set_title("Prediction")
            plt.colorbar(im2, ax=axes[col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            im3 = axes[col_idx].imshow(err_img, cmap="magma", origin='lower')
            axes[col_idx].set_title("Abs Error")
            plt.colorbar(im3, ax=axes[col_idx], fraction=0.046, pad=0.04)

            plt.tight_layout()
            self.writer.add_figure(tag, fig, global_step=step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"[TensorBoardLogger] Failed to log flow field grid '{tag}': {e}")

    def log_temporal_rollout_strip(
        self,
        gt_seq: Union[torch.Tensor, np.ndarray],
        pred_seq: Union[torch.Tensor, np.ndarray],
        step: int,
        tag: str = "FlowField/RolloutSequence",
        num_frames: int = 5,
        cmap: str = "coolwarm"
    ) -> None:
        """
        Log temporal rollout sequence comparison across time steps.
        gt_seq, pred_seq: shape [T, H, W] or [1, T, C, H, W].
        """
        if not self.enabled or self.writer is None:
            return

        try:
            if isinstance(gt_seq, torch.Tensor):
                gt_seq = gt_seq.detach().cpu().numpy()
            if isinstance(pred_seq, torch.Tensor):
                pred_seq = pred_seq.detach().cpu().numpy()

            gt_seq = np.squeeze(gt_seq)
            pred_seq = np.squeeze(pred_seq)

            if gt_seq.ndim == 4:  # [T, C, H, W]
                gt_seq = gt_seq[:, 0]
                pred_seq = pred_seq[:, 0]

            T = gt_seq.shape[0]
            frame_indices = np.linspace(0, T - 1, num=min(num_frames, T), dtype=int)

            fig, axes = plt.subplots(3, len(frame_indices), figsize=(3 * len(frame_indices), 8), squeeze=False)

            vmin = min(gt_seq.min(), pred_seq.min())
            vmax = max(gt_seq.max(), pred_seq.max())

            for i, t in enumerate(frame_indices):
                gt_f = gt_seq[t]
                pred_f = pred_seq[t]
                err_f = np.abs(pred_f - gt_f)

                axes[0, i].imshow(gt_f, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                axes[0, i].set_title(f"GT (t={t})")
                axes[0, i].axis('off')

                axes[1, i].imshow(pred_f, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                axes[1, i].set_title(f"Pred (t={t})")
                axes[1, i].axis('off')

                axes[2, i].imshow(err_f, cmap="magma", origin='lower')
                axes[2, i].set_title(f"Err (t={t})")
                axes[2, i].axis('off')

            plt.tight_layout()
            self.writer.add_figure(tag, fig, global_step=step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"[TensorBoardLogger] Failed to log temporal rollout strip '{tag}': {e}")

    def log_error_histogram(
        self,
        gt: Union[torch.Tensor, np.ndarray],
        pred: Union[torch.Tensor, np.ndarray],
        step: int,
        tag: str = "Histograms/ResidualError"
    ) -> None:
        """
        Log residual error distribution (Pred - GT) histogram.
        """
        if not self.enabled or self.writer is None:
            return

        try:
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()

            residuals = (pred - gt).flatten()
            self.writer.add_histogram(tag, residuals, global_step=step)
        except Exception as e:
            logger.warning(f"[TensorBoardLogger] Failed to log error histogram '{tag}': {e}")

    def log_energy_spectrum(
        self,
        gt_field: Union[torch.Tensor, np.ndarray],
        pred_field: Union[torch.Tensor, np.ndarray],
        step: int,
        tag: str = "Physics/EnergySpectrum"
    ) -> None:
        """
        Log 2D Fourier energy spectrum comparison (Wavenumber k vs Power E(k)).
        """
        if not self.enabled or self.writer is None:
            return

        try:
            gt_img = self._to_numpy_2d(gt_field)
            pred_img = self._to_numpy_2d(pred_field)

            def calc_spectrum(field_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                fft2 = np.fft.fft2(field_2d)
                fft2_shift = np.fft.fftshift(fft2)
                power = np.abs(fft2_shift) ** 2

                ny, nx = field_2d.shape
                cy, cx = ny // 2, nx // 2
                y, x = np.ogrid[:ny, :nx]
                r = np.hypot(x - cx, y - cy).astype(int)

                tbin = np.bincount(r.ravel(), power.ravel())
                nr = np.bincount(r.ravel())
                radial_profile = tbin / np.maximum(nr, 1)
                k = np.arange(len(radial_profile))
                return k[1:], radial_profile[1:]

            k_gt, e_gt = calc_spectrum(gt_img)
            k_pred, e_pred = calc_spectrum(pred_img)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.loglog(k_gt, e_gt, 'b-', label='Ground Truth $E(k)$', linewidth=2)
            ax.loglog(k_pred, e_pred, 'r--', label='Prediction $E(k)$', linewidth=2)
            ax.set_xlabel('Wavenumber $k$')
            ax.set_ylabel('Power Spectrum $E(k)$')
            ax.set_title('Energy Spectrum Comparison')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()

            self.writer.add_figure(tag, fig, global_step=step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"[TensorBoardLogger] Failed to log energy spectrum '{tag}': {e}")

    def log_model_gradients_and_weights(self, model: nn.Module, step: int, tag_prefix: str = "Model") -> None:
        """
        Log distributions of model weights and gradients.
        """
        if not self.enabled or self.writer is None:
            return

        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    clean_name = name.replace('.', '/')
                    self.writer.add_histogram(f"{tag_prefix}/Weights/{clean_name}", param.data.cpu().numpy(), global_step=step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"{tag_prefix}/Gradients/{clean_name}", param.grad.cpu().numpy(), global_step=step)
        except Exception as e:
            logger.warning(f"[TensorBoardLogger] Failed to log weights/gradients: {e}")

    def flush(self) -> None:
        """Flush pending writes."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close writer safely."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            logger.info("[TensorBoardLogger] Writer closed successfully.")
