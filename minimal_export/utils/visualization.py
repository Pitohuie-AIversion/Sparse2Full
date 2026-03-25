"""
AR训练可视化工具

提供训练过程中的可视化功能
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


class ARVisualizer:
    """AR训练可视化器"""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self, history: list[dict[str, float]], save_path: str | None = None
    ):
        """
        绘制训练曲线

        Args:
            history: 训练历史
            save_path: 保存路径
        """
        if not history:
            return

        epochs = range(1, len(history) + 1)

        # 提取指标
        train_losses = [h.get("train_loss", 0) for h in history]
        val_losses = [h.get("val_loss", 0) for h in history]

        plt.figure(figsize=(10, 6))

        # 绘制训练损失
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)

        # 绘制验证损失
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            fmt = Path(save_path).suffix.lstrip(".") or "svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight", format=fmt)
        else:
            save_path = str(self.save_dir / "training_curves.svg")
            plt.savefig(save_path, dpi=150, bbox_inches="tight", format="svg")

        plt.close()

        return save_path

    def save_training_curves(
        self, history: list[dict[str, float]], save_path: str | None = None
    ):
        """兼容接口：保存训练曲线（调用 plot_training_curves）"""
        return self.plot_training_curves(history, save_path)

    def plot_obs_gt_pred_err_horizontal(
        self,
        observation: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        save_path: str | None = None,
        num_samples: int = 4,
        channel: int = 0,
        cmap_main: str = "viridis",
        cmap_err: str = "magma",
    ) -> str:
        """
        标准四列水平排列可视化：观测→真实(GT)→预测→误差。

        - 统一色标：观测/真实/预测使用相同 vmin/vmax 与相同 cmap
        - 误差使用独立色标（非负），cmap_err

        Args:
            observation: 观测张量 [B, C, H, W]
            targets: 目标张量 [B, C, H, W]
            predictions: 预测张量 [B, C, H, W]
            save_path: 保存路径（png）；为空时保存到默认目录
            num_samples: 可视化的样本数（行数）
            channel: 可视化通道索引（默认0）
            cmap_main: 观测/真实/预测的色图
            cmap_err: 误差图色图
        Returns:
            保存路径字符串
        """
        # 输入校验
        if not (observation.dim() == targets.dim() == predictions.dim() == 4):
            raise ValueError("Inputs must be [B, C, H, W] tensors")

        bsz = min(
            observation.size(0), targets.size(0), predictions.size(0), num_samples
        )

        # 统一色标范围（按当前批次的全局范围）
        def extract_ch(x: torch.Tensor) -> np.ndarray:
            arr = x.detach().cpu().numpy()
            if arr.shape[1] <= channel:
                # 若指定通道不存在，回退到0
                ch_idx = 0
            else:
                ch_idx = channel
            return arr[:, ch_idx]

        obs_np = extract_ch(observation)
        tgt_np = extract_ch(targets)
        pred_np = extract_ch(predictions)
        err_np = np.abs(pred_np - tgt_np)

        # 统一色标：主图的 vmin/vmax 按三者联合取范围
        vmin_main = float(np.min([obs_np.min(), tgt_np.min(), pred_np.min()]))
        vmax_main = float(np.max([obs_np.max(), tgt_np.max(), pred_np.max()]))
        vmin_err = 0.0
        vmax_err = float(err_np.max())

        # 创建画布：每样本一行，4列（观测/真实/预测/误差）
        fig, axes = plt.subplots(bsz, 4, figsize=(18, 4 * bsz))
        if bsz == 1:
            axes = axes.reshape(1, 4)

        for i in range(bsz):
            # 观测
            im0 = axes[i, 0].imshow(
                obs_np[i], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main
            )
            axes[i, 0].set_title(f"观测 Observation #{i+1}")
            axes[i, 0].axis("off")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # 真值
            im1 = axes[i, 1].imshow(
                tgt_np[i], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main
            )
            axes[i, 1].set_title(f"真实 Ground Truth #{i+1}")
            axes[i, 1].axis("off")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # 预测
            im2 = axes[i, 2].imshow(
                pred_np[i], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main
            )
            axes[i, 2].set_title(f"预测 Prediction #{i+1}")
            axes[i, 2].axis("off")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # 误差
            im3 = axes[i, 3].imshow(
                err_np[i],
                cmap=cmap_err,
                vmin=vmin_err,
                vmax=max(vmin_err + 1e-12, vmax_err),
            )
            axes[i, 3].set_title(f"误差 Error #{i+1}")
            axes[i, 3].axis("off")
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path is None:
            save_path = str(self.save_dir / "obs_gt_pred_err.svg")
        fmt = Path(save_path).suffix.lstrip(".") or "svg"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format=fmt)
        plt.close()

        return save_path

    def plot_predictions_comparison(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        save_path: str | None = None,
        num_samples: int = 4,
    ):
        """
        绘制预测结果对比

        Args:
            predictions: 预测值 [B, C, H, W]
            targets: 目标值 [B, C, H, W]
            save_path: 保存路径
            num_samples: 样本数量
        """
        if predictions.dim() != 4 or targets.dim() != 4:
            return

        batch_size = min(predictions.size(0), num_samples)

        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for i in range(batch_size):
            # 预测结果
            pred_img = predictions[i].detach().cpu().numpy()
            if pred_img.shape[0] == 1:  # 单通道
                pred_img = pred_img.squeeze(0)
            else:  # 多通道，取第一个通道
                pred_img = pred_img[0]

            # 目标结果
            target_img = targets[i].detach().cpu().numpy()
            if target_img.shape[0] == 1:  # 单通道
                target_img = target_img.squeeze(0)
            else:  # 多通道，取第一个通道
                target_img = target_img[0]

            # 误差图
            error_img = np.abs(pred_img - target_img)

            # 绘制
            axes[i, 0].imshow(pred_img, cmap="viridis")
            axes[i, 0].set_title(f"Prediction {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(target_img, cmap="viridis")
            axes[i, 1].set_title(f"Target {i+1}")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(error_img, cmap="hot")
            axes[i, 2].set_title(f"Error {i+1}")
            axes[i, 2].axis("off")

            # 添加颜色条
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            save_path = str(self.save_dir / "predictions_comparison.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()

        return save_path

    def create_training_report(
        self,
        training_results: dict[str, Any],
        test_results: dict[str, float],
        save_dir: str,
    ) -> str:
        """
        创建训练报告

        Args:
            training_results: 训练结果
            test_results: 测试结果
            save_dir: 保存目录

        Returns:
            报告路径
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        report_path = save_dir / "training_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("AR Training Report\n")
            f.write("=" * 50 + "\n\n")

            # 实验信息
            f.write(f"Experiment: {training_results['experiment_name']}\n")
            f.write(f"Total Epochs: {training_results['total_epochs']}\n")
            f.write(
                f"Best Validation Loss: {training_results['best_val_loss']:.6f}\n\n"
            )

            # 测试结果
            f.write("Test Results:\n")
            f.write("-" * 30 + "\n")
            for key, value in test_results.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\n")

            # 训练历史
            if "training_history" in training_results:
                history = training_results["training_history"]
                f.write("Training History (Last 5 epochs):\n")
                f.write("-" * 40 + "\n")

                start_idx = max(0, len(history) - 5)
                for i, metrics in enumerate(history[start_idx:], start=start_idx + 1):
                    f.write(f"Epoch {i}:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")

        return str(report_path)


class PDEBenchVisualizer:
    """PDEBench统一可视化器

    - 创建标准子目录：`fields/`、`spectra/`、`analysis/`、`comparisons/`
    - 提供测试与评估脚本所需的最小方法集
    """

    def __init__(
        self,
        save_dir: str = "visualizations",
        dpi: int = 300,
        output_format: str = "png",
        figsize: tuple = (12, 8),
        colormap: str = "viridis",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.fields_dir = self.save_dir / "fields"
        self.spectra_dir = self.save_dir / "spectra"
        self.analysis_dir = self.save_dir / "analysis"
        self.comparisons_dir = self.save_dir / "comparisons"
        for d in [
            self.fields_dir,
            self.spectra_dir,
            self.analysis_dir,
            self.comparisons_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.output_format = output_format
        self.figsize = figsize
        self.colormap = colormap

    # -------- helpers ---------
    def _tensor_to_numpy(self, x: torch.Tensor, channel: int = 0) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)

        if arr.ndim == 4:  # [B, C, H, W]
            b = 0 if arr.shape[0] > 0 else -1
            c = channel if arr.shape[1] > channel else 0
            return arr[b, c]
        if arr.ndim == 3:  # [C, H, W]
            c = channel if arr.shape[0] > channel else 0
            return arr[c]
        if arr.ndim == 2:  # [H, W]
            return arr
        # 尝试最后两维作为图像
        return arr.reshape(arr.shape[-2], arr.shape[-1])

    def _save_fig(self, fig, subdir: Path, save_name: str) -> str:
        ext = (
            f".{self.output_format}"
            if not save_name.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
            else ""
        )
        path = subdir / f"{save_name}{ext}"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # -------- public API ---------
    def plot_field_comparison(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        degraded: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
        save_name: str = "field_comparison",
        channel: int = 0,
    ) -> str:
        if degraded is None and baseline is not None:
            degraded = baseline
        legacy_save_root = (
            isinstance(degraded, (str, Path)) and save_name == "field_comparison"
        )
        if legacy_save_root:
            save_name = str(degraded)
            degraded = None
        gt_np = self._tensor_to_numpy(gt, channel)
        pred_np = self._tensor_to_numpy(pred, channel)
        vmin = float(min(gt_np.min(), pred_np.min()))
        vmax = float(max(gt_np.max(), pred_np.max()))

        cols = 3 if degraded is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=self.figsize)
        if cols == 2:
            ax_gt, ax_pred = axes
        else:
            ax_deg, ax_gt, ax_pred = axes

        if degraded is not None:
            deg_np = self._tensor_to_numpy(degraded, channel)
            im0 = ax_deg.imshow(deg_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
            ax_deg.set_title("Observed/Degraded")
            ax_deg.axis("off")
            plt.colorbar(im0, ax=ax_deg, fraction=0.046, pad=0.04)

        im1 = ax_gt.imshow(gt_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        plt.colorbar(im1, ax=ax_gt, fraction=0.046, pad=0.04)

        im2 = ax_pred.imshow(pred_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        plt.colorbar(im2, ax=ax_pred, fraction=0.046, pad=0.04)

        target_dir = self.save_dir if legacy_save_root else self.fields_dir
        return self._save_fig(fig, target_dir, save_name)

    def plot_training_curves(
        self,
        train_logs: dict[str, list[float]],
        val_logs: dict[str, list[float]],
        save_name: str = "training_curves",
    ) -> str:
        train_loss = train_logs.get("loss", [])
        val_loss = val_logs.get("loss", [])
        epochs = range(1, max(len(train_loss), len(val_loss)) + 1)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        if train_loss:
            ax.plot(
                list(epochs)[: len(train_loss)],
                train_loss,
                label="train_loss",
                linewidth=2,
            )
        if val_loss:
            ax.plot(
                list(epochs)[: len(val_loss)], val_loss, label="val_loss", linewidth=2
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return self._save_fig(fig, self.analysis_dir, save_name)

    def create_quadruplet_visualization(
        self,
        observed: torch.Tensor,
        gt: torch.Tensor,
        pred: torch.Tensor,
        save_name: str = "quadruplet",
        channel: int = 0,
    ) -> str:
        """创建四联图：Observed / GT / Pred / Error。"""
        obs_np = self._tensor_to_numpy(observed, channel)
        gt_np = self._tensor_to_numpy(gt, channel)
        pred_np = self._tensor_to_numpy(pred, channel)
        err_np = np.abs(pred_np - gt_np)

        vmin = float(min(obs_np.min(), gt_np.min(), pred_np.min()))
        vmax = float(max(obs_np.max(), gt_np.max(), pred_np.max()))

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        im0 = axes[0].imshow(obs_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[0].set_title("Observed")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(gt_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(pred_np, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(
            err_np, cmap="magma", vmin=0.0, vmax=max(1e-12, float(err_np.max()))
        )
        axes[3].set_title("Error")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        return self._save_fig(fig, self.fields_dir, save_name)

    def create_correlation_heatmap(
        self, corr_matrix: np.ndarray, save_name: str = "correlation"
    ) -> str:
        """创建相关性热图。"""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._save_fig(fig, self.analysis_dir, save_name)

    def plot_power_spectrum_comparison(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        save_name: str = "power_spectrum_comparison",
        log_scale: bool = True,
        channel: int = 0,
    ) -> str:
        """生成功率谱对比图（GT/Pred）。"""
        gt_np = self._tensor_to_numpy(gt, channel)
        pred_np = self._tensor_to_numpy(pred, channel)

        def spectrum(img: np.ndarray) -> np.ndarray:
            spec = np.fft.fftshift(np.abs(np.fft.fft2(img)))
            if log_scale:
                spec = np.log1p(spec)
            return spec

        gt_spec = spectrum(gt_np)
        pred_spec = spectrum(pred_np)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axes[0].imshow(gt_spec, cmap="inferno")
        axes[0].set_title("GT Spectrum")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_spec, cmap="inferno")
        axes[1].set_title("Pred Spectrum")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        return self._save_fig(fig, self.spectra_dir, save_name)

    def create_failure_case_analysis(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        metrics: dict[str, float],
        save_name: str = "failure_case",
        failure_type: str = "",
        channel: int = 0,
    ) -> str:
        """生成失败案例分析图：GT/Pred/Err + 指标摘要。"""
        gt_np = self._tensor_to_numpy(gt, channel)
        pred_np = self._tensor_to_numpy(pred, channel)
        err_np = np.abs(pred_np - gt_np)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        im0 = axes[0, 0].imshow(gt_np, cmap=self.colormap)
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].axis("off")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(pred_np, cmap=self.colormap)
        axes[0, 1].set_title("Prediction")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(err_np, cmap="magma")
        axes[1, 0].set_title("Error")
        axes[1, 0].axis("off")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 指标文本框
        axes[1, 1].axis("off")
        text_lines = [f"Failure Type: {failure_type}"] + [
            f"{k}: {v:.4f}" for k, v in metrics.items()
        ]
        axes[1, 1].text(
            0.05, 0.95, "\n".join(text_lines), va="top", ha="left", fontsize=12
        )

        return self._save_fig(fig, self.analysis_dir, save_name)

    def create_metrics_summary_plot(
        self,
        metrics_by_model: dict[str, dict[str, list]],
        save_name: str = "metrics_summary",
    ) -> str:
        """创建指标汇总图（简单条形图，取各指标均值）。"""
        # 计算均值
        model_names = list(metrics_by_model.keys())
        metric_names = (
            list(next(iter(metrics_by_model.values())).keys()) if model_names else []
        )

        means = np.array(
            [
                [np.mean(metrics_by_model[m].get(k, [0])) for k in metric_names]
                for m in model_names
            ]
        )

        fig, ax = plt.subplots(1, 1, figsize=(max(8, 2 * len(metric_names)), 6))
        x = np.arange(len(metric_names))
        width = 0.8 / max(1, len(model_names))
        for i, m in enumerate(model_names):
            ax.bar(x + i * width, means[i], width=width, label=m)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=30, ha="right")
        ax.set_ylabel("Mean Value")
        ax.set_title("Metrics Summary")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig, self.comparisons_dir, save_name)


# -------- functional wrappers expected by some integration tests --------
def create_comparison_plot(gt: torch.Tensor, pred: torch.Tensor, save_path: str) -> str:
    """函数式封装：创建GT/Pred对比图。

    Args:
        gt: 真实值张量
        pred: 预测值张量
        save_path: 完整文件路径或文件名（将保存到fields目录）
    Returns:
        保存的文件路径字符串
    """
    vis = PDEBenchVisualizer(save_dir=str(Path(save_path).parent))
    # 提取文件名（不带扩展）
    name = Path(save_path).stem
    return vis.plot_field_comparison(gt=gt, pred=pred, save_name=name)


def create_spectrum_plot(
    data: torch.Tensor, save_path: str, log_scale: bool = True
) -> str:
    """函数式封装：生成功率谱图。

    Args:
        data: 输入数据张量（GT或Pred）
        save_path: 完整文件路径或文件名（将保存到spectra目录）
        log_scale: 是否对谱取log1p
    Returns:
        保存的文件路径字符串
    """
    vis = PDEBenchVisualizer(save_dir=str(Path(save_path).parent))
    name = Path(save_path).stem
    # 使用自身与自身的对比来生成单图谱
    # 这里调用对比函数，但只绘制data的谱；为了简化，绘制data与data的谱。
    return vis.plot_power_spectrum_comparison(
        gt=data, pred=data, save_name=name, log_scale=log_scale
    )


# -------- temporal visualization support (for tests expecting these symbols) --------
class TemporalVisualizer:
    """简易时序可视化器

    - 提供最小接口以满足测试导入与基本使用
    - 支持将时序 GT 与 Pred 进行网格对比
    """

    def __init__(
        self,
        save_dir: str = "visualizations",
        dpi: int = 200,
        colormap: str = "viridis",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.colormap = colormap

    def plot_temporal_comparison(
        self,
        gt_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        save_name: str = "temporal_comparison",
        channel: int = 0,
    ) -> str:
        """绘制时序比较图：每列一个时间步，行包含 GT / Pred / Error。

        接受形状：[T, C, H, W] 或 [B, T, C, H, W]（将使用第0个样本）。
        """

        # 统一成 [T, C, H, W]
        def to_TCHW(x: torch.Tensor) -> np.ndarray:
            arr = (
                x.detach().cpu().numpy()
                if isinstance(x, torch.Tensor)
                else np.asarray(x)
            )
            if arr.ndim == 5:  # [B, T, C, H, W]
                arr = arr[0]
            return arr

        gt = to_TCHW(gt_seq)
        pred = to_TCHW(pred_seq)

        assert (
            gt.ndim == 4 and pred.ndim == 4
        ), "gt_seq/pred_seq 必须是 [T, C, H, W] 或 [B, T, C, H, W]"
        T = gt.shape[0]

        # 选择通道
        ch_gt = gt[:, channel if gt.shape[1] > channel else 0]
        ch_pred = pred[:, channel if pred.shape[1] > channel else 0]
        ch_err = np.abs(ch_pred - ch_gt)

        vmin = float(min(ch_gt.min(), ch_pred.min()))
        vmax = float(max(ch_gt.max(), ch_pred.max()))

        fig, axes = plt.subplots(3, T, figsize=(3 * T, 9))
        if T == 1:
            axes = axes.reshape(3, 1)

        for t in range(T):
            im0 = axes[0, t].imshow(ch_gt[t], cmap=self.colormap, vmin=vmin, vmax=vmax)
            axes[0, t].set_title(f"GT t={t+1}")
            axes[0, t].axis("off")
            plt.colorbar(im0, ax=axes[0, t], fraction=0.046, pad=0.04)

            im1 = axes[1, t].imshow(
                ch_pred[t], cmap=self.colormap, vmin=vmin, vmax=vmax
            )
            axes[1, t].set_title(f"Pred t={t+1}")
            axes[1, t].axis("off")
            plt.colorbar(im1, ax=axes[1, t], fraction=0.046, pad=0.04)

            im2 = axes[2, t].imshow(
                ch_err[t], cmap="magma", vmin=0.0, vmax=max(1e-12, float(ch_err.max()))
            )
            axes[2, t].set_title(f"Error t={t+1}")
            axes[2, t].axis("off")
            plt.colorbar(im2, ax=axes[2, t], fraction=0.046, pad=0.04)

        plt.tight_layout()
        path = self.save_dir / f"{save_name}.png"
        fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)


def create_temporal_comparison(
    gt_seq: torch.Tensor,
    pred_seq: torch.Tensor,
    save_path: str,
    channel: int = 0,
) -> str:
    """函数式封装：创建时序GT/Pred/Err对比图。

    接受 [T, C, H, W] 或 [B, T, C, H, W]；保存到 `save_path` 所在目录。
    """
    vis = TemporalVisualizer(save_dir=str(Path(save_path).parent))
    name = Path(save_path).stem
    return vis.plot_temporal_comparison(
        gt_seq=gt_seq, pred_seq=pred_seq, save_name=name, channel=channel
    )
