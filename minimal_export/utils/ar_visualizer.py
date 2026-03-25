#!/usr/bin/env python3
"""
AR Training Visualizer
Provides training curves, prediction visualization, and error analysis.
"""

import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Force non-GUI backend for headless/HPC environments
import sys
import warnings

import matplotlib.pyplot as plt
import torch

from ops.degradation import apply_degradation_operator

warnings.filterwarnings("ignore")

# 添加tools/visualization路径以导入可视化工具
project_root = Path(__file__).resolve().parents[1]
viz_tools_path = project_root / "tools" / "visualization"
if str(viz_tools_path) not in sys.path:
    sys.path.append(str(viz_tools_path))


# Safe English font configuration to avoid missing glyph boxes
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300  # High-quality output


class ARTrainingVisualizer:
    """AR training visualizer"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.vis_dir / "training_curves").mkdir(exist_ok=True)
        (self.vis_dir / "predictions").mkdir(exist_ok=True)
        (self.vis_dir / "error_analysis").mkdir(exist_ok=True)
        (self.vis_dir / "temporal_analysis").mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_format = os.environ.get("VIZ_FORMAT", "svg")
        self.max_time_cols = int(os.environ.get("VIZ_MAX_TIME_COLS", "6"))

    def plot_training_curves(
        self, history: dict[str, list], save_name: str = "training_curves"
    ):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("AR Training Monitoring", fontsize=16, fontweight="bold")

        epochs = history.get("epochs", [])
        # 如果没有epochs，但有损失数据，使用长度生成1..N的索引
        max_series_len = max(
            len(history.get("train_losses", [])),
            len(history.get("val_losses", [])),
            len(history.get("learning_rates", [])),
        )
        if not epochs and max_series_len > 0:
            epochs = list(range(1, max_series_len + 1))
        if not epochs:
            print("Warning: no training history found")
            return

        def x_for(series_len: int) -> list[int]:
            """根据序列长度生成与y匹配的x轴，优先使用epochs的前缀"""
            if series_len <= 0:
                return []
            if len(epochs) >= series_len:
                return epochs[:series_len]
            return list(range(1, series_len + 1))

        # 训练和验证损失
        ax1 = axes[0, 0]
        train_losses = history.get("train_losses", [])
        val_losses = history.get("val_losses", [])
        if train_losses:
            ax1.plot(
                x_for(len(train_losses)),
                train_losses,
                "b-",
                label="Training Loss",
                linewidth=2,
            )
        if val_losses:
            ax1.plot(
                x_for(len(val_losses)),
                val_losses,
                "r-",
                label="Validation Loss",
                linewidth=2,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        if any([train_losses, val_losses]):
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # 学习率曲线
        ax2 = axes[0, 1]
        learning_rates = history.get("learning_rates", [])
        if learning_rates:
            ax2.plot(x_for(len(learning_rates)), learning_rates, "g-", linewidth=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale("log")

        # Curriculum learning stages
        ax3 = axes[1, 0]
        if "curriculum_stages" in history:
            stages = history["curriculum_stages"]
            if stages:
                stage_epochs = []
                stage_T_outs = []
                for stage in stages:
                    # 某些记录可能缺少epoch，跳过以避免与x轴不对齐
                    if "epoch" in stage:
                        stage_epochs.append(stage["epoch"])
                        stage_T_outs.append(stage.get("T_out", 0))
                if stage_epochs and stage_T_outs:
                    ax3.step(
                        stage_epochs, stage_T_outs, "o-", linewidth=2, markersize=8
                    )
                    ax3.set_xlabel("Epoch")
                    ax3.set_ylabel("Prediction steps (T_out)")
                    ax3.set_title("Curriculum Progress")
                    ax3.grid(True, alpha=0.3)

        # Validation metrics
        ax4 = axes[1, 1]
        if "val_metrics" in history and history["val_metrics"]:
            # 提取指标数据
            metrics_data = {}
            for epoch_metrics in history["val_metrics"]:
                for metric_name, value in epoch_metrics.items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(value)

            # 绘制主要指标，确保x轴长度与y一致
            for metric_name, values in metrics_data.items():
                if metric_name in ["rel_l2", "mae", "mse"] and values:
                    ax4.plot(
                        x_for(len(values)),
                        values,
                        "o-",
                        label=metric_name.upper(),
                        linewidth=2,
                    )

            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Metric Value")
            ax4.set_title("Validation Metrics")
            if any(values for values in metrics_data.values()):
                ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale("log")

        plt.tight_layout()
        save_path = (
            self.vis_dir / "training_curves" / f"{save_name}.{self.image_format}"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format=self.image_format)
        if self.image_format != "png":
            png_path = self.vis_dir / "training_curves" / f"{save_name}.png"
            plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
        plt.close()

        print(f"✅ Training curves saved: {save_path}")

    def visualize_ar_predictions(
        self,
        input_seq: torch.Tensor,
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        timestep_idx: int = 0,
        save_name: str = "ar_predictions",
        norm_stats: dict = None,
        h_params: dict = None,
        sample_idx: object | None = None,
    ):
        """Visualize AR predictions with unified colorbar"""
        is_tensor = isinstance(target_seq, torch.Tensor)

        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and "mean" in norm_stats and "std" in norm_stats:
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0

            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，AR可视化使用z-score域数据")

        # 统一处理5D输入 [B, T, C, H, W] -> 4D [T, C, H, W]
        # 我们总是取 batch 中的第一个样本进行可视化
        if len(target_seq.shape) == 5:
            input_frame_t = input_seq[0]
            target_frames_t = target_seq[0]
            pred_frames_t = pred_seq[0]
        elif len(target_seq.shape) == 4:
            # 兼容 4D 输入: [B, C, H, W] (视为 Batch) 或 [T, C, H, W] (视为 Sequence)
            # 这里 ar_visualizer 默认偏向 [B, C, H, W] 语义，但在 AR 任务中常传入 [T, C, H, W]
            # 如果传入的是 Tensor 且 dimension=4，通常 AR 脚本会传入 [T, C, H, W]
            input_frame_t = input_seq[0] if len(input_seq.shape) > 3 else input_seq
            target_frames_t = (
                target_seq[0] if len(target_seq.shape) == 4 else target_seq
            )
            pred_frames_t = pred_seq[0] if len(pred_seq.shape) == 4 else pred_seq
        elif len(input_seq.shape) == 4:  # [B, C, H, W] - 旧格式兼容
            input_frame_t = input_seq[0]
            target_frames_t = (
                target_seq[0] if len(target_seq.shape) == 4 else target_seq
            )
            pred_frames_t = pred_seq[0] if len(pred_seq.shape) == 4 else pred_seq
        else:
            input_frame_t = input_seq
            target_frames_t = target_seq
            pred_frames_t = pred_seq

        if not is_tensor:
            input_frame_t = torch.as_tensor(input_frame_t)
            target_frames_t = torch.as_tensor(target_frames_t)
            pred_frames_t = torch.as_tensor(pred_frames_t)

        # 确保维度正确 - 处理多通道情况
        # 对于输入 input_frame_t，它可能是 [T_in, C, H, W] 或 [C, H, W]
        # 如果是序列，我们只展示最后一帧作为观测
        if len(input_frame_t.shape) == 4:  # [T, C, H, W]
            input_frame_t = input_frame_t[-1]  # 取最后一帧

        if len(input_frame_t.shape) == 3:
            # 对于3D输入 [C, H, W]，取第一个通道 [H, W]
            if input_frame_t.shape[0] >= 1:
                input_frame_t = input_frame_t[0]
            else:
                input_frame_t = input_frame_t.squeeze(0)  # 移除通道维度

        if len(target_frames_t.shape) == 4:
            # 对于4D序列 [T, C, H, W]，取第一个通道 [T, H, W]
            if target_frames_t.shape[1] >= 1:
                target_frames_t = target_frames_t[:, 0]
            else:
                target_frames_t = target_frames_t.squeeze(1)  # 移除通道维度

        if len(pred_frames_t.shape) == 4:
            # 对于4D序列 [T, C, H, W]，取第一个通道 [T, H, W]
            if pred_frames_t.shape[1] >= 1:
                pred_frames_t = pred_frames_t[:, 0]
            else:
                pred_frames_t = pred_frames_t.squeeze(1)  # 移除通道维度

        input_frame = (input_frame_t * std_val + mean_val).detach().cpu().numpy()
        target_frames = (target_frames_t * std_val + mean_val).detach().cpu().numpy()
        pred_frames = (pred_frames_t * std_val + mean_val).detach().cpu().numpy()

        obs_frame = None
        if h_params is not None:
            # 如果有退化参数，基于GT的第一帧生成观测
            gt0 = target_frames_t[0]
            # 统一为 [B,C,H,W]
            if gt0.ndim == 2:  # [H,W]
                gt0_4d = gt0.unsqueeze(0).unsqueeze(0)
            elif gt0.ndim == 3:  # [C,H,W]
                gt0_4d = gt0.unsqueeze(0)
            else:
                # 回退：尝试强制成 [1,1,H,W]
                arr = gt0
                while arr.ndim < 4:
                    arr = arr.unsqueeze(0)
                gt0_4d = arr
            obs = apply_degradation_operator(gt0_4d, h_params)
            # 取 [H,W]
            obs_frame = obs[0, 0].detach().cpu().numpy()
        else:
            obs_frame = input_frame

        # 确定需要可视化的时间步数 T_out
        T_out = min(target_frames.shape[0], pred_frames.shape[0])

        # 计算统一的数值范围（基于百分位数避免异常值影响）
        all_values = np.concatenate(
            [
                obs_frame.flatten(),
                target_frames[: min(T_out, 6)].flatten(),
                pred_frames[: min(T_out, 6)].flatten(),
            ]
        )
        vmin = np.percentile(all_values, 2)  # 2%分位数
        vmax = np.percentile(all_values, 98)  # 98%分位数

        # 创建更合理的布局：时间序列可视化
        # 限制最大列数，如果 T_out 很大，可以分行或截断，这里按 max_time_cols 截断
        n_cols = min(T_out, self.max_time_cols)

        # 修正：确保图表足够大以容纳所有子图
        fig = plt.figure(figsize=(4 * n_cols + 2, 12))  # 增加宽度给colorbar

        # 创建网格布局，为colorbar预留右侧空间
        gs = fig.add_gridspec(
            3, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.1, hspace=0.2
        )

        title_suffix = f"sample={str(sample_idx)} " if sample_idx is not None else ""
        fig.suptitle(
            f"AR Predictions Sequence - {title_suffix}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 逐时间步绘制
        for t in range(n_cols):
            # 第一行：Observed (仅在 t=0 显示，或者每列都显示相同的 Obs 以便对比)
            # 为了美观，第一行我们在 t=0 显示 H(GT) 观测，后续列留白或显示 t 时刻的 GT 观测（如果是动态观测）
            # 这里我们假设初始观测是静态的或只关注初始条件，所以在 t=0 绘制 Obs

            if t == 0:
                obs_2d = self._ensure_2d_for_imshow(obs_frame)
                ax_input = fig.add_subplot(gs[0, t])
                im1 = ax_input.imshow(
                    obs_2d, cmap="RdBu_r", aspect="equal", vmin=vmin, vmax=vmax
                )
                ax_input.set_title(
                    f"Observed H(x_0)\n[{obs_2d.shape[0]}×{obs_2d.shape[1]}]",
                    fontsize=10,
                    fontweight="bold",
                )
                ax_input.set_xticks([])
                ax_input.set_yticks([])
            else:
                # 后续列留白，或者可以显示 input_seq 的演化（如果有）
                ax_input = fig.add_subplot(gs[0, t])
                ax_input.axis("off")

            # 第二行：Ground Truth
            target_2d = self._ensure_2d_for_imshow(target_frames[t])
            ax_target = fig.add_subplot(gs[1, t])
            im2 = ax_target.imshow(
                target_2d, cmap="RdBu_r", aspect="equal", vmin=vmin, vmax=vmax
            )
            ax_target.set_title(
                f"GT (t={t+1})\n[{target_2d.shape[0]}×{target_2d.shape[1]}]",
                fontsize=10,
                fontweight="bold",
            )
            ax_target.set_xticks([])
            ax_target.set_yticks([])

            # 第三行：Prediction
            pred_2d = self._ensure_2d_for_imshow(pred_frames[t])
            ax_pred = fig.add_subplot(gs[2, t])
            im3 = ax_pred.imshow(
                pred_2d, cmap="RdBu_r", aspect="equal", vmin=vmin, vmax=vmax
            )
            ax_pred.set_title(
                f"Pred (t={t+1})\n[{pred_2d.shape[0]}×{pred_2d.shape[1]}]",
                fontsize=10,
                fontweight="bold",
            )
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

        # 添加统一的颜色条 - 在最右侧，跨越所有行
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Value", fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        save_path = (
            self.vis_dir / "predictions" / f"{save_name}_sequence.{self.image_format}"
        )
        final_path = self._save_with_suffix(save_path, fig)
        if self.image_format != "png":
            png_path = self.vis_dir / "predictions" / f"{save_name}_sequence.png"
            fig.savefig(
                png_path, dpi=300, bbox_inches="tight", pad_inches=0.1, format="png"
            )
        plt.close(fig)
        print(f"✅ AR prediction sequence visualization saved: {final_path}")

    def visualize_single_frame(
        self,
        obs: torch.Tensor,
        gt: torch.Tensor,
        pred: torch.Tensor,
        save_name: str = "seq_last_frame",
        norm_stats: dict = None,
    ):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        if obs.ndim == 4:
            obs0 = obs[0]
        elif obs.ndim == 3:
            obs0 = obs
        elif obs.ndim == 2:
            obs0 = obs
        else:
            t = obs
            while t.ndim > 3:
                t = t[0]
            obs0 = t
        if gt.ndim == 4:
            gt0 = gt[0]
        elif gt.ndim == 3:
            gt0 = gt
        else:
            t = gt
            while t.ndim > 3:
                t = t[0]
            gt0 = t
        if pred.ndim == 4:
            pr0 = pred[0]
        elif pred.ndim == 3:
            pr0 = pred
        else:
            t = pred
            while t.ndim > 3:
                t = t[0]
            pr0 = t
        if norm_stats is not None:
            if "mean" in norm_stats and "std" in norm_stats:
                mean = norm_stats["mean"]
                std = norm_stats["std"]
                mean_val = (
                    float(mean[0])
                    if isinstance(mean, torch.Tensor) and mean.numel() > 0
                    else (float(mean) if np.isscalar(mean) else 0.0)
                )
                std_val = (
                    float(std[0])
                    if isinstance(std, torch.Tensor) and std.numel() > 0
                    else (float(std) if np.isscalar(std) else 1.0)
                )
            elif "u_mean" in norm_stats and "u_std" in norm_stats:
                mean_val = (
                    float(norm_stats["u_mean"])
                    if not np.isscalar(norm_stats["u_mean"])
                    else float(norm_stats["u_mean"])
                )
                std_val = (
                    float(norm_stats["u_std"])
                    if not np.isscalar(norm_stats["u_std"])
                    else float(norm_stats["u_std"])
                )
            else:
                mean_val, std_val = 0.0, 1.0
        else:
            mean_val, std_val = 0.0, 1.0
        if obs0.ndim == 3 and obs0.shape[0] > 1:
            obs0 = obs0[0]
        if gt0.ndim == 3 and gt0.shape[0] > 1:
            gt0 = gt0[0]
        if pr0.ndim == 3 and pr0.shape[0] > 1:
            pr0 = pr0[0]
        obs_v = (obs0 * std_val + mean_val).detach().cpu().numpy()
        gt_v = (gt0 * std_val + mean_val).detach().cpu().numpy()
        pr_v = (pr0 * std_val + mean_val).detach().cpu().numpy()

        # Auto-resize if shapes mismatch (e.g. SR task: gt/pred are HR, obs is LR)
        if gt_v.shape != pr_v.shape:
            print(
                f"⚠️ Visualization shape mismatch: GT {gt_v.shape} vs Pred {pr_v.shape}. Resizing Pred to match GT."
            )
            from skimage.transform import resize

            # Assuming (H, W) or (C, H, W)
            if pr_v.ndim == 2:
                pr_v = resize(
                    pr_v, gt_v.shape, order=1, preserve_range=True, anti_aliasing=False
                )
            else:
                # (C, H, W) -> transpose to (H, W, C) for resize -> transpose back
                pr_v_t = pr_v.transpose(1, 2, 0)
                gt_v_t = gt_v.transpose(1, 2, 0)
                pr_v_resized = resize(
                    pr_v_t,
                    gt_v_t.shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                )
                pr_v = pr_v_resized.transpose(2, 0, 1)

        err_v = np.abs(gt_v - pr_v)
        vals = np.concatenate([obs_v.flatten(), gt_v.flatten(), pr_v.flatten()])
        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(
            obs_v if obs_v.ndim == 2 else obs_v.squeeze(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 0].set_title("Observed")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[0, 1].imshow(
            gt_v if gt_v.ndim == 2 else gt_v.squeeze(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        axes[1, 0].imshow(
            pr_v if pr_v.ndim == 2 else pr_v.squeeze(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 0].set_title("Prediction")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        axes[1, 1].imshow(err_v if err_v.ndim == 2 else err_v.squeeze(), cmap="inferno")
        axes[1, 1].set_title("Error")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        plt.tight_layout()
        save_path = self.vis_dir / "predictions" / f"{save_name}.{self.image_format}"
        final_path = self._save_with_suffix(save_path, fig)
        if self.image_format != "png":
            png_path = self.vis_dir / "predictions" / f"{save_name}.png"
            fig.savefig(
                png_path, dpi=300, bbox_inches="tight", pad_inches=0.1, format="png"
            )
        plt.close(fig)
        print(f"✅ Single-frame prediction visualization saved: {final_path}")

    def visualize_obs_gt_pred_error(
        self,
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        save_name: str = "obs_gt_pred_error",
        norm_stats: dict = None,
        h_params: dict = None,
        timestep_idx: int = 0,
        sample_idx: object | None = None,
        observation_seq: torch.Tensor | None = None,
    ):
        # 归一化统计
        if norm_stats is not None:
            if "mean" in norm_stats and "std" in norm_stats:
                mean = norm_stats["mean"]
                std = norm_stats["std"]
                mean_val = (
                    float(mean[0])
                    if isinstance(mean, torch.Tensor) and mean.numel() > 0
                    else (float(mean) if np.isscalar(mean) else 0.0)
                )
                std_val = (
                    float(std[0])
                    if isinstance(std, torch.Tensor) and std.numel() > 0
                    else (float(std) if np.isscalar(std) else 1.0)
                )
            elif "u_mean" in norm_stats and "u_std" in norm_stats:
                mean_val = (
                    float(norm_stats["u_mean"])
                    if not np.isscalar(norm_stats["u_mean"])
                    else float(norm_stats["u_mean"])
                )
                std_val = (
                    float(norm_stats["u_std"])
                    if not np.isscalar(norm_stats["u_std"])
                    else float(norm_stats["u_std"])
                )
            else:
                mean_val, std_val = 0.0, 1.0
        else:
            mean_val, std_val = 0.0, 1.0

        # 统一取第一个样本，并取第一个时间步，确保形状为 [C,H,W]
        if not isinstance(target_seq, torch.Tensor):
            tgt = torch.as_tensor(target_seq)
        else:
            tgt = target_seq
        if not isinstance(pred_seq, torch.Tensor):
            pr = torch.as_tensor(pred_seq)
        else:
            pr = pred_seq

        if tgt.ndim == 5:  # [B,T,C,H,W]
            t_sel = max(0, min(int(timestep_idx), tgt.shape[1] - 1))
            tgt0 = tgt[0, t_sel]
        elif tgt.ndim == 4:  # [T,C,H,W] 或 [B,C,H,W]
            tgt0 = tgt[0]
        elif tgt.ndim == 3:  # [C,H,W]
            tgt0 = tgt
        else:
            # 回退：试图压到3维
            while tgt.ndim > 3:
                tgt = tgt[0]
            tgt0 = tgt
        gt0 = tgt0 * std_val + mean_val
        if gt0.ndim == 3 and gt0.shape[0] > 1:
            gt0 = gt0[0]

        if pr.ndim == 5:
            t_sel = max(0, min(int(timestep_idx), pr.shape[1] - 1))
            pr0 = pr[0, t_sel]
        elif pr.ndim == 4:
            pr0 = pr[0]
        elif pr.ndim == 3:
            pr0 = pr
        else:
            while pr.ndim > 3:
                pr = pr[0]
            pr0 = pr
        pred0 = pr0 * std_val + mean_val
        if pred0.ndim == 3 and pred0.shape[0] > 1:
            pred0 = pred0[0]

        # 构造观测：优先使用数据中的真实观测序列，否则回退为对GT应用H
        if observation_seq is not None:
            try:
                if isinstance(observation_seq, torch.Tensor):
                    if observation_seq.ndim == 5:
                        t_sel = max(
                            0, min(int(timestep_idx), observation_seq.shape[1] - 1)
                        )
                        obs0 = observation_seq[0, t_sel, 0].detach().cpu().numpy()
                    elif observation_seq.ndim == 4:
                        obs0 = observation_seq[0, 0].detach().cpu().numpy()
                    elif observation_seq.ndim == 3:
                        obs0 = observation_seq[0].detach().cpu().numpy()
                    else:
                        obs0 = observation_seq.detach().cpu().numpy()
                else:
                    obs0 = np.array(observation_seq)
            except Exception:
                obs0 = self._ensure_2d_for_imshow(gt0)
        else:
            if h_params is not None:
                gt0_4d = gt0
                if gt0_4d.ndim == 2:
                    gt0_4d = gt0_4d.unsqueeze(0).unsqueeze(0)
                elif gt0_4d.ndim == 3:
                    gt0_4d = gt0_4d.unsqueeze(0)
                obs = apply_degradation_operator(gt0_4d, h_params)
                try:
                    task = str(h_params.get("task", ""))
                    scale_raw = h_params.get("scale", h_params.get("scale_factor", 1))
                    scale = int(scale_raw) if scale_raw is not None else 1
                    if task == "SR" and scale > 1:
                        import torch.nn.functional as F

                        H_gt, W_gt = gt0_4d.shape[-2], gt0_4d.shape[-1]
                        exp_h, exp_w = max(1, H_gt // scale), max(1, W_gt // scale)
                        H_obs, W_obs = obs.shape[-2], obs.shape[-1]
                        if (H_obs != exp_h) or (W_obs != exp_w):
                            obs = F.interpolate(obs, size=(exp_h, exp_w), mode="area")
                except Exception:
                    pass
                obs0 = obs[0, 0].detach().cpu().numpy()
                try:
                    print(
                        f"[Viz Debug] Obs shape: {obs0.shape}, GT shape: {gt0.shape}, Pred shape: {pred0.shape}, H params: {h_params}"
                    )
                    print(f"[Viz Debug] Norm(mean,std): ({mean_val},{std_val})")
                except Exception:
                    pass
            else:
                obs0 = self._ensure_2d_for_imshow(gt0)

        # 2. 转换并确保形状一致
        gt_v = gt0.detach().cpu().numpy()
        pred_v = pred0.detach().cpu().numpy()
        obs_v = obs0

        # 确保是2D (但这里如果是多通道，squeeze可能会失败或不当)
        if gt_v.ndim > 2 and gt_v.shape[0] == 1:
            gt_v = gt_v.squeeze()
        if pred_v.ndim > 2 and pred_v.shape[0] == 1:
            pred_v = pred_v.squeeze()
        if obs_v.ndim > 2 and obs_v.shape[0] == 1:
            obs_v = obs_v.squeeze()

        # Auto-resize if shapes mismatch
        if gt_v.shape != pred_v.shape:
            print(
                f"⚠️ Obs/GT/Pred/Error shape mismatch: GT {gt_v.shape} vs Pred {pred_v.shape}. Resizing Pred to match GT."
            )
            from skimage.transform import resize

            # Assuming (H, W) or (C, H, W)
            if pred_v.ndim == 2:
                pred_v = resize(
                    pred_v,
                    gt_v.shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                )
            else:
                # (C, H, W) -> transpose to (H, W, C) for resize -> transpose back
                pr_v_t = pred_v.transpose(1, 2, 0)
                gt_v_t = gt_v.transpose(1, 2, 0)
                pr_v_resized = resize(
                    pr_v_t,
                    gt_v_t.shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                )
                pred_v = pr_v_resized.transpose(2, 0, 1)

        err_v = np.abs(gt_v - pred_v)

        # 传递处理后的numpy数组给ensure_2d_for_imshow，而不是原始Tensor
        gt0 = self._ensure_2d_for_imshow(gt_v)
        pred0 = self._ensure_2d_for_imshow(pred_v)
        obs0 = self._ensure_2d_for_imshow(obs_v)
        err0 = self._ensure_2d_for_imshow(err_v)

        # 统一色标范围基于百分位
        vals = np.concatenate([obs0.flatten(), gt0.flatten(), pred0.flatten()])
        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        title_suffix = f"sample={str(sample_idx)} " if sample_idx is not None else ""
        fig.suptitle(
            f"Obs/GT/Pred/Error - {title_suffix}t={timestep_idx}",
            fontsize=14,
            fontweight="bold",
        )
        im_obs = axes[0, 0].imshow(obs0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Observed H(GT) [{obs0.shape[0]}×{obs0.shape[1]}]")
        axes[0, 0].axis("off")
        im_gt = axes[0, 1].imshow(gt0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Ground Truth [{gt0.shape[0]}×{gt0.shape[1]}]")
        axes[0, 1].axis("off")
        im_pred = axes[1, 0].imshow(pred0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f"Prediction [{pred0.shape[0]}×{pred0.shape[1]}]")
        axes[1, 0].axis("off")
        im_err = axes[1, 1].imshow(
            err0, cmap="Reds", vmin=0, vmax=float(np.percentile(err0, 98))
        )
        axes[1, 1].set_title(f"|GT - Pred| [{err0.shape[0]}×{err0.shape[1]}]")
        axes[1, 1].axis("off")

        cb_val = fig.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cb_val.set_label("Value", fontsize=10)
        cb_err = fig.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cb_err.set_label("Abs Error", fontsize=10)

        plt.tight_layout()
        save_path = (
            self.vis_dir
            / "predictions"
            / f"{save_name}_t{timestep_idx}.{self.image_format}"
        )
        final_path = self._save_with_suffix(save_path, fig)
        if self.image_format != "png":
            png_path = self.vis_dir / "predictions" / f"{save_name}_t{timestep_idx}.png"
            fig.savefig(
                png_path, dpi=300, bbox_inches="tight", pad_inches=0.1, format="png"
            )
        plt.close(fig)
        print(f"✅ Obs/GT/Pred/Error saved: {final_path}")

    def create_error_analysis(
        self,
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        save_name: str = "error_analysis",
        norm_stats: dict = None,
    ):
        """Create error analysis visualization"""
        # 转换为numpy
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()

        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and "mean" in norm_stats and "std" in norm_stats:
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0

            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，误差分析使用z-score域数据")

        # 选择第一个样本
        if len(target_seq.shape) == 4:  # [B, T, C, H, W]
            target_seq = target_seq[0]  # [T, C, H, W]
            pred_seq = pred_seq[0]  # [T, C, H, W]

        # 反归一化到真实数据尺度
        target_seq = target_seq * std_val + mean_val
        pred_seq = pred_seq * std_val + mean_val

        T_out = min(target_seq.shape[0], pred_seq.shape[0])

        # 计算误差（在真实数据尺度上）
        # Auto-resize if shapes mismatch
        if target_seq.shape != pred_seq.shape:
            print(
                f"⚠️ Error analysis shape mismatch: Target {target_seq.shape} vs Pred {pred_seq.shape}. Resizing Pred."
            )
            from skimage.transform import resize

            # Target: [T, C, H, W]
            # We resize pred_seq to match target_seq
            # resize expects [H, W, C] or similar, so we process frame by frame or use n-dim resize

            # Safest way: iterate over T
            new_pred = np.zeros_like(target_seq)
            for t in range(min(target_seq.shape[0], pred_seq.shape[0])):
                tgt_t = target_seq[t]  # [C, H, W]
                pred_t = pred_seq[t]  # [C, H, W] usually, or mismatch

                if tgt_t.shape != pred_t.shape:
                    if pred_t.ndim == 3:  # [C, H, W]
                        pred_t_tr = pred_t.transpose(1, 2, 0)  # [H, W, C]
                        tgt_t_tr = tgt_t.transpose(1, 2, 0)
                        resized = resize(
                            pred_t_tr,
                            tgt_t_tr.shape,
                            order=1,
                            preserve_range=True,
                            anti_aliasing=False,
                        )
                        new_pred[t] = resized.transpose(2, 0, 1)
                    else:  # [H, W] or other
                        new_pred[t] = resize(
                            pred_t,
                            tgt_t.shape,
                            order=1,
                            preserve_range=True,
                            anti_aliasing=False,
                        )
                else:
                    new_pred[t] = pred_t
            pred_seq = new_pred

        errors = np.abs(target_seq - pred_seq)

        # 创建误差分析图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("AR Prediction Error Analysis", fontsize=16, fontweight="bold")

        # 时间步误差演化
        ax1 = axes[0, 0]
        mse_per_step = []
        mae_per_step = []
        for t in range(T_out):
            mse = np.mean((target_seq[t] - pred_seq[t]) ** 2)
            mae = np.mean(np.abs(target_seq[t] - pred_seq[t]))
            mse_per_step.append(mse)
            mae_per_step.append(mae)

        ax1.plot(range(1, T_out + 1), mse_per_step, "ro-", label="MSE", linewidth=2)
        ax1.plot(range(1, T_out + 1), mae_per_step, "bo-", label="MAE", linewidth=2)
        ax1.set_xlabel("Prediction Steps")
        ax1.set_ylabel("Error")
        ax1.set_title("Error Evolution Over Time")
        if mse_per_step or mae_per_step:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # 误差分布直方图
        ax2 = axes[0, 1]
        all_errors = errors.flatten()
        ax2.hist(all_errors, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.set_xlabel("Absolute Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution Histogram")
        ax2.grid(True, alpha=0.3)

        # 误差热图（最后一个时间步）
        ax3 = axes[0, 2]
        if T_out > 0:
            # errors 形状可能是 [T, C, H, W] 或 [C, H, W]
            if errors.ndim == 4:  # [T, C, H, W]
                error_map = errors[-1, 0]  # 获取最后时间步的第一个通道 [H, W]
            elif errors.ndim == 3:  # [C, H, W]
                error_map = errors[0]  # 获取第一个通道 [H, W]
            else:  # [H, W]
                error_map = errors

            # 确保是2D数组
            while error_map.ndim > 2:
                error_map = error_map[0]  # 继续取第一个元素直到2D

            if error_map.ndim == 1:
                # 如果是1D，尝试重塑为2D
                size = int(np.sqrt(error_map.shape[0]))
                if size * size == error_map.shape[0]:
                    error_map = error_map.reshape(size, size)

            im = ax3.imshow(error_map, cmap="Reds", aspect="equal")
            ax3.set_title(
                f"Error Heatmap (t={T_out}) [{error_map.shape[0]}×{error_map.shape[1]}]"
            )
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # 空间误差分析
        ax4 = axes[1, 0]
        if T_out > 0:
            # 计算空间平均误差
            if errors.ndim == 4:  # [T, C, H, W]
                spatial_error = np.mean(errors, axis=0)  # 平均时间步 [C, H, W]
            else:  # [C, H, W] 或其他
                spatial_error = errors

            # 使用helper函数确保2D
            spatial_error_2d = self._ensure_2d_for_imshow(spatial_error)

            im = ax4.imshow(spatial_error_2d, cmap="Reds", aspect="equal")
            ax4.set_title(
                f"Spatial Mean Error [{spatial_error_2d.shape[0]}×{spatial_error_2d.shape[1]}]"
            )
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        # 通道误差对比
        ax5 = axes[1, 1]
        if target_seq.shape[1] > 1:  # 多通道
            channel_errors = []
            for c in range(target_seq.shape[1]):
                channel_error = np.mean(errors[:, c])
                channel_errors.append(channel_error)

            ax5.bar(range(len(channel_errors)), channel_errors, color=["red", "blue"])
            ax5.set_xlabel("Channel")
            ax5.set_ylabel("Mean Absolute Error")
            ax5.set_title("Channel Error Comparison")
            ax5.set_xticks(range(len(channel_errors)))
            ax5.set_xticklabels([f"Ch{i}" for i in range(len(channel_errors))])

        # 累积误差
        ax6 = axes[1, 2]
        cumulative_error = np.cumsum([np.mean(errors[t]) for t in range(T_out)])
        ax6.plot(range(1, T_out + 1), cumulative_error, "go-", linewidth=2)
        ax6.set_xlabel("Prediction Steps")
        ax6.set_ylabel("Cumulative Error")
        ax6.set_title("Cumulative Error Growth")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.vis_dir / "error_analysis" / f"{save_name}.{self.image_format}"
        final_path = self._save_with_suffix(save_path, fig)
        if self.image_format != "png":
            png_path = self.vis_dir / "error_analysis" / f"{save_name}.png"
            fig.savefig(
                png_path, dpi=300, bbox_inches="tight", pad_inches=0.1, format="png"
            )
        plt.close(fig)
        print(f"✅ Error analysis visualization saved: {final_path}")

    def create_temporal_analysis(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        save_name: str = "temporal_analysis",
        norm_stats: dict = None,
    ):
        """Create temporal analysis visualization"""
        # 转换为numpy
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()

        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and "mean" in norm_stats and "std" in norm_stats:
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0

            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，时间分析使用z-score域数据")

        # 统一处理5D输入 [B, T, C, H, W] -> 4D [T, C, H, W]
        if len(pred_seq.shape) == 5:  # [B, T, C, H, W]
            pred_seq = pred_seq[0]  # [T, C, H, W]
            target_seq = target_seq[0]  # [T, C, H, W]
        elif len(pred_seq.shape) == 4:  # [B, T, C, H] 或其他4D格式
            pred_seq = pred_seq[0] if pred_seq.shape[0] == 1 else pred_seq
            target_seq = target_seq[0] if target_seq.shape[0] == 1 else target_seq

        # 反归一化到真实数据尺度
        pred_seq = pred_seq * std_val + mean_val
        target_seq = target_seq * std_val + mean_val

        T_out = min(pred_seq.shape[0], target_seq.shape[0])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Temporal Evolution Analysis", fontsize=16, fontweight="bold")

        # 能量演化
        ax1 = axes[0, 0]
        pred_energy = []
        target_energy = []
        for t in range(T_out):
            p = pred_seq[t]
            g = target_seq[t]
            if p.ndim == 3:  # [C,H,W]
                p_energy = np.sum(p**2)
            else:  # [H,W]
                p_energy = np.sum(p**2)
            if g.ndim == 3:
                g_energy = np.sum(g**2)
            else:
                g_energy = np.sum(g**2)
            pred_energy.append(p_energy)
            target_energy.append(g_energy)

        ax1.plot(
            range(1, T_out + 1),
            target_energy,
            "r-",
            label="Ground Truth Energy",
            linewidth=2,
        )
        ax1.plot(
            range(1, T_out + 1),
            pred_energy,
            "b--",
            label="Prediction Energy",
            linewidth=2,
        )
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Total Energy")
        ax1.set_title("Energy Conservation Analysis")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 相关性演化
        ax2 = axes[0, 1]
        correlations = []
        for t in range(T_out):
            p = pred_seq[t]
            g = target_seq[t]
            # 对齐通道与空间尺寸：通道取均值为单通道，空间取交集
            if p.ndim == 3:  # [C,H,W]
                p = p.mean(axis=0)
            if g.ndim == 3:
                g = g.mean(axis=0)
            h = min(p.shape[-2], g.shape[-2])
            w = min(p.shape[-1], g.shape[-1])
            p = p[:h, :w]
            g = g[:h, :w]
            try:
                corr = float(np.corrcoef(p.flatten(), g.flatten())[0, 1])
            except Exception:
                corr = 0.0
            correlations.append(corr)

        ax2.plot(range(1, T_out + 1), correlations, "go-", linewidth=2)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Correlation Coefficient")
        ax2.set_title("Prediction–Ground Truth Correlation")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # 频谱分析
        ax3 = axes[1, 0]
        if T_out > 1:
            # 计算最后时间步的频谱
            p_last = pred_seq[-1]
            g_last = target_seq[-1]
            if p_last.ndim == 3:
                p_last = p_last.mean(axis=0)
            if g_last.ndim == 3:
                g_last = g_last.mean(axis=0)
            h = min(p_last.shape[-2], g_last.shape[-2])
            w = min(p_last.shape[-1], g_last.shape[-1])
            p_last = p_last[:h, :w]
            g_last = g_last[:h, :w]
            pred_fft = np.fft.fft2(p_last)
            target_fft = np.fft.fft2(g_last)

            pred_power = np.abs(pred_fft) ** 2
            target_power = np.abs(target_fft) ** 2

            # 确保是2D数组
            if pred_power.ndim > 2:
                pred_power = pred_power.squeeze()
            if target_power.ndim > 2:
                target_power = target_power.squeeze()

            # 如果仍然不是2D，取第一个通道
            if pred_power.ndim > 2:
                pred_power = pred_power[0]
            if target_power.ndim > 2:
                target_power = target_power[0]

            # 径向平均
            h, w = pred_power.shape
            center = (h // 2, w // 2)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

            r_int = r.astype(int)
            max_r = min(h // 2, w // 2)

            pred_radial = []
            target_radial = []
            for i in range(max_r):
                mask = r_int == i
                if np.any(mask):
                    pred_radial.append(np.mean(pred_power[mask]))
                    target_radial.append(np.mean(target_power[mask]))

            ax3.loglog(
                pred_radial, "b-", label="Prediction Power Spectrum", linewidth=2
            )
            ax3.loglog(
                target_radial, "r-", label="Ground Truth Power Spectrum", linewidth=2
            )
            ax3.set_xlabel("Wavenumber")
            ax3.set_ylabel("Power")
            ax3.set_title("Power Spectrum Comparison")
            try:
                ax3.legend()
            except Exception:
                pass
            ax3.grid(True, alpha=0.3)

        # 稳定性分析
        ax4 = axes[1, 1]
        if T_out > 1:
            stability = []
            for t in range(1, T_out):
                diff = np.mean(np.abs(pred_seq[t] - pred_seq[t - 1]))
                stability.append(diff)

            ax4.plot(range(2, T_out + 1), stability, "mo-", linewidth=2)
            ax4.set_xlabel("Timestep")
            ax4.set_ylabel("Frame Difference")
            ax4.set_title("Prediction Stability")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = (
            self.vis_dir / "temporal_analysis" / f"{save_name}.{self.image_format}"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format=self.image_format)
        if self.image_format != "png":
            png_path = self.vis_dir / "temporal_analysis" / f"{save_name}.png"
            plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
        plt.close()

        print(f"✅ Temporal analysis visualization saved: {save_path}")

    def plot_obs_gt_pred_err_horizontal(
        self, obs, gt, pred, save_path=None, num_samples=4, crop_params=None
    ):
        """
        Plot Obs | GT | Pred | Error horizontally for multiple samples.
        Compatible with the fallback logic in train_real_data_ar.py.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure inputs are numpy arrays
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        if hasattr(gt, "detach"):
            gt = gt.detach().cpu().numpy()
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()

        # Calculate error
        err = np.abs(gt - pred)

        # If obs looks like full image (Canvas Mode), try to mask it for visualization if crop_params provided
        if crop_params is not None and obs.shape == gt.shape:
            # Try to infer crop box from crop_params or just trust the obs
            # If obs has constant value regions, it's likely already masked
            pass

        n = min(obs.shape[0], num_samples)

        for i in range(n):
            obs_img = self._ensure_2d_for_imshow(obs[i])
            gt_img = self._ensure_2d_for_imshow(gt[i])
            pr_img = self._ensure_2d_for_imshow(pred[i])
            er_img = self._ensure_2d_for_imshow(err[i])

            # Unified color range for physical values
            vmin_phys = float(min(np.min(obs_img), np.min(gt_img), np.min(pr_img)))
            vmax_phys = float(max(np.max(obs_img), np.max(gt_img), np.max(pr_img)))

            # Symmetric range for error
            max_err = float(np.max(er_img))

            fig = plt.figure(figsize=(16, 4))
            gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.05)

            # Obs
            ax0 = fig.add_subplot(gs[0])
            im0 = ax0.imshow(obs_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
            ax0.set_title("Obs", fontsize=11, fontweight="bold")
            ax0.axis("off")

            # GT
            ax1 = fig.add_subplot(gs[1])
            im1 = ax1.imshow(gt_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
            ax1.set_title("GT", fontsize=11, fontweight="bold")
            ax1.axis("off")

            # Pred
            ax2 = fig.add_subplot(gs[2])
            im2 = ax2.imshow(pr_img, cmap="viridis", vmin=vmin_phys, vmax=vmax_phys)
            ax2.set_title("Pred", fontsize=11, fontweight="bold")
            ax2.axis("off")

            # Error
            ax3 = fig.add_subplot(gs[3])
            im3 = ax3.imshow(
                er_img, cmap="coolwarm", vmin=0, vmax=max_err
            )  # Error is absolute, so 0 to max
            ax3.set_title("Error", fontsize=11, fontweight="bold")
            ax3.axis("off")

            # Colorbar
            cbar = fig.colorbar(im0, cax=fig.add_subplot(gs[4]), orientation="vertical")
            cbar.set_label("Physical Value", fontsize=10)

            # Save
            if save_path:
                p = Path(save_path)
                p_sample = p.parent / f"{p.stem}_{i:03d}{p.suffix}"
                self._save_with_suffix(p_sample, fig)

            plt.close(fig)

        return save_path

    def create_boundary_and_frequency_metrics(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        save_name: str = "diagnostics",
        band_width: int = 16,
    ):
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()
        if pred_seq.ndim == 5:
            pred_seq = pred_seq[0]
        if target_seq.ndim == 5:
            target_seq = target_seq[0]
        if pred_seq.ndim == 4:
            pred_last = pred_seq[-1]
        else:
            pred_last = pred_seq
        if target_seq.ndim == 4:
            tgt_last = target_seq[-1]
        else:
            tgt_last = target_seq
        if pred_last.ndim == 3:
            pred_last = pred_last.mean(axis=0)
        if tgt_last.ndim == 3:
            tgt_last = tgt_last.mean(axis=0)
        h, w = int(pred_last.shape[-2]), int(pred_last.shape[-1])
        h = max(h, 1)
        w = max(w, 1)
        bw = int(max(1, min(band_width, h // 2, w // 2)))
        err = pred_last - tgt_last
        rmse_all = float(np.sqrt(np.mean(err**2)))
        mask_b = np.zeros((h, w), dtype=bool)
        mask_b[:bw, :] = True
        mask_b[-bw:, :] = True
        mask_b[:, :bw] = True
        mask_b[:, -bw:] = True
        mask_c = ~mask_b
        b_rmse = (
            float(np.sqrt(np.mean(err[mask_b] ** 2))) if np.any(mask_b) else rmse_all
        )
        c_rmse = (
            float(np.sqrt(np.mean(err[mask_c] ** 2))) if np.any(mask_c) else rmse_all
        )
        pred_fft = np.fft.fft2(pred_last)
        tgt_fft = np.fft.fft2(tgt_last)
        diff_fft = np.abs(pred_fft) - np.abs(tgt_fft)
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_int = r.astype(int)
        max_r = int(r_int.max())
        t1 = int(min(16, max_r))
        t2 = int(min(32, max_r))
        low_mask = r_int <= t1
        mid_mask = (r_int > t1) & (r_int <= t2)
        high_mask = r_int > t2
        f_low = (
            float(np.sqrt(np.mean(diff_fft[low_mask] ** 2)))
            if np.any(low_mask)
            else 0.0
        )
        f_mid = (
            float(np.sqrt(np.mean(diff_fft[mid_mask] ** 2)))
            if np.any(mid_mask)
            else 0.0
        )
        f_high = (
            float(np.sqrt(np.mean(diff_fft[high_mask] ** 2)))
            if np.any(high_mask)
            else 0.0
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        names = ["bRMSE", "cRMSE", "fRMSE-low", "fRMSE-mid", "fRMSE-high"]
        values = [b_rmse, c_rmse, f_low, f_mid, f_high]
        bars = ax.bar(
            names,
            values,
            color=["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"],
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )
        ax.set_ylabel("Value")
        ax.set_title("Boundary and Frequency RMSE Metrics")
        plt.xticks(rotation=0)
        plt.tight_layout()
        save_path = self.vis_dir / "error_analysis" / f"{save_name}.{self.image_format}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format=self.image_format)
        if self.image_format != "png":
            png_path = self.vis_dir / "error_analysis" / f"{save_name}.png"
            plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
        plt.close()

    def create_comprehensive_report(
        self, history: dict, sample_data: dict | None = None
    ):
        """Create comprehensive report"""
        report_path = self.vis_dir / "comprehensive_report.html"
        # 安全地计算统计值，避免空列表导致的索引错误
        epochs_list = history.get("epochs", []) or []
        train_losses = history.get("train_losses", []) or []
        val_losses = history.get("val_losses", []) or []
        total_epochs = len(epochs_list)
        final_train_loss = train_losses[-1] if len(train_losses) > 0 else 0.0
        final_val_loss = val_losses[-1] if len(val_losses) > 0 else 0.0
        best_val_loss = min(val_losses) if len(val_losses) > 0 else 0.0

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AR Training Comprehensive Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
                .metric-label {{ color: #7f8c8d; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AR Training Comprehensive Report</h1>
                <p>Generated at: {Path().cwd().name} - {np.datetime64('now')}</p>
            </div>
            
            <div class="section">
                <h2>Training Overview</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{total_epochs}</div>
                        <div class="metric-label">Epochs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{final_train_loss:.6f}</div>
                        <div class="metric-label">Final Training Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{final_val_loss:.6f}</div>
                        <div class="metric-label">Final Validation Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{best_val_loss:.6f}</div>
                        <div class="metric-label">Best Validation Loss</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image-grid">
        """

        # 添加图片
        for img_dir in [
            "training_curves",
            "predictions",
            "error_analysis",
            "temporal_analysis",
        ]:
            img_path = self.vis_dir / img_dir
            if img_path.exists():
                for img_file in list(img_path.glob("*.png")) + list(
                    img_path.glob("*.svg")
                ):
                    rel_path = img_file.relative_to(self.vis_dir)
                    html_content += f'<div><h3>{img_file.stem}</h3><img src="{rel_path}" alt="{img_file.stem}"></div>\n'

        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Comprehensive report saved: {report_path}")

        return str(report_path)

    def _ensure_2d_for_imshow(self, arr):
        """确保数组是2D的，适合imshow显示"""
        # 如果是tensor，转换为numpy
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()

        # 如果维度大于2，取第一个通道/元素
        while arr.ndim > 2:
            arr = arr[0]

        # 如果是1D，尝试重塑为2D
        if arr.ndim == 1:
            size = int(np.sqrt(arr.shape[0]))
            if size * size == arr.shape[0]:
                arr = arr.reshape(size, size)
            else:
                # 如果不是完全平方数，创建一个合理的2D形状
                h = int(np.sqrt(arr.shape[0]))
                w = arr.shape[0] // h
                if h * w <= arr.shape[0]:
                    arr = arr[: h * w].reshape(h, w)
                else:
                    # 最后的备选方案：创建一个小的2D数组
                    arr = np.zeros((8, 8))

        # 确保是numpy数组且是2D
        arr = np.asarray(arr)
        if arr.ndim != 2:
            # 如果仍然不是2D，强制创建一个2D数组
            arr = np.zeros((8, 8))

        return arr

    def _save_with_suffix(
        self,
        save_path: Path,
        fig,
        dpi: int = 300,
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
    ) -> Path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.exists():
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                format=save_path.suffix.lstrip("."),
            )
            return save_path
        stem = save_path.stem
        suffix = save_path.suffix
        i = 1
        while True:
            candidate = save_path.parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                fig.savefig(
                    candidate,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    format=suffix.lstrip("."),
                )
                return candidate
            i += 1
