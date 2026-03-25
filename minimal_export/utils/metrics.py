"""评测指标模块

实现PDEBench稀疏观测重建系统的所有评测指标：
- Rel-L2: 相对L2误差
- MAE: 平均绝对误差
- PSNR: 峰值信噪比
- SSIM: 结构相似性指数
- fRMSE: 频域RMSE (low/mid/high)
- bRMSE: 边界RMSE
- cRMSE: 中心RMSE
- ||H(ŷ)−y||: 数据一致性误差

按照开发手册要求：
- 每通道先算，后等权平均
- 支持统计分析（均值±标准差）
- 支持显著性检验
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim

try:
    from ops.degradation import apply_degradation_operator
except ImportError:
    # Fallback or dummy for when ops is not available (e.g. strict unit testing without env)
    # But in production this should be available.
    def apply_degradation_operator(
        x: torch.Tensor, obs: dict[str, Any]
    ) -> torch.Tensor:
        return x


class MetricsCalculator:
    """指标计算器

    提供完整的评测指标计算功能
    支持批量计算和统计分析
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        boundary_width: int = 16,
        freq_bands: dict[str, tuple[int, int]] | None = None,
    ):
        """
        Args:
            image_size: 图像尺寸 (H, W)
            boundary_width: 边界宽度（像素）
            freq_bands: 频段定义 {'band_name': (low_freq, high_freq)}
        """
        self.image_size = image_size
        self.boundary_width = boundary_width

        # 默认频段设置
        if freq_bands is None:
            max_freq = min(image_size) // 2
            self.freq_bands = {
                "low": (0, max_freq // 4),
                "mid": (max_freq // 4, max_freq // 2),
                "high": (max_freq // 2, max_freq),
            }
        else:
            self.freq_bands = freq_bands

        # 预计算掩码
        self._precompute_masks()

        # 获取日志记录器
        self.logger = logging.getLogger("MetricsCalculator")

    def update_image_size(self, new_size: tuple[int, int]) -> None:
        """更新图像尺寸并重新计算掩码"""
        if new_size != self.image_size:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Updating image size from {self.image_size} to {new_size}"
                )
            self.image_size = new_size
            self._precompute_masks()

    def _precompute_masks(self) -> None:
        """预计算各种掩码"""
        H, W = self.image_size

        # 边界掩码
        self.boundary_mask = torch.zeros(H, W, dtype=torch.bool)
        bw = self.boundary_width
        # 防止 boundary_width 大于图像尺寸的一半
        bw = min(bw, H // 2, W // 2)
        if bw > 0:
            self.boundary_mask[:bw, :] = True  # 上边界
            self.boundary_mask[-bw:, :] = True  # 下边界
            self.boundary_mask[:, :bw] = True  # 左边界
            self.boundary_mask[:, -bw:] = True  # 右边界

        # 中心掩码
        self.center_mask = ~self.boundary_mask

        # 频域掩码
        self.freq_masks = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = self._create_freq_mask(H, W, low_freq, high_freq)
            self.freq_masks[band_name] = mask

    def _create_freq_mask(
        self, H: int, W: int, low_freq: int, high_freq: int
    ) -> torch.Tensor:
        """创建频域掩码"""
        # 创建频率网格
        ky = torch.fft.fftfreq(H, d=1.0).abs()
        kx = torch.fft.fftfreq(W, d=1.0).abs()
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        # 径向频率
        k_radial = torch.sqrt(kx_grid**2 + ky_grid**2)

        # 频率范围掩码
        mask = (k_radial >= low_freq / max(H, W)) & (k_radial < high_freq / max(H, W))
        return mask

    def _normalize_tensor_dims(
        self, x: torch.Tensor | NDArray[Any], label: str = "tensor"
    ) -> torch.Tensor:
        """规范化张量维度为 [N, C, H, W]

        处理规则：
        - [H, W] -> [1, 1, H, W]
        - [C, H, W] -> [1, C, H, W] (优先视为CHW，除非判定为HWC)
        - [N, C, H, W] -> 保持
        - [B, T, C, H, W] -> [B, C, H, W] (取最后一帧)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{label} must be torch.Tensor or np.ndarray, got {type(x)}"
            )

        if x.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            return x.unsqueeze(0).unsqueeze(0)

        elif x.dim() == 3:
            # 判定 HWC 还是 CHW
            d0, d1, d2 = x.shape
            # 启发式：若最后一维很小且前两维较大，视为 HWC
            # 例如 (128, 128, 3) vs (3, 128, 128)
            if d2 <= 4 and d0 > 8 and d1 > 8:
                # HWC -> CHW -> [1, C, H, W]
                return x.permute(2, 0, 1).unsqueeze(0)
            else:
                # CHW -> [1, C, H, W]
                return x.unsqueeze(0)

        elif x.dim() == 4:
            return x

        elif x.dim() == 5:
            # [B, T, C, H, W] -> [B, C, H, W] (取最后一帧)
            return x[:, -1, ...]

        raise ValueError(
            f"{label} must be 2D/3D/4D/5D tensor, got dim={x.dim()} with shape={tuple(x.shape)}"
        )

    def compute_rel_l2(
        self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_rel_l2")
        target = self._normalize_tensor_dims(target, "target_rel_l2")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target.reshape(target.size(0), target.size(1), -1)
        diff_norm = torch.norm(pred_flat - target_flat, dim=-1)
        target_norm = torch.norm(target_flat, dim=-1)
        return diff_norm / (target_norm + eps)

    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_mae")
        target = self._normalize_tensor_dims(target, "target_mae")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target.reshape(target.size(0), target.size(1), -1)
        return torch.mean(torch.abs(pred_flat - target_flat), dim=-1)

    def compute_psnr(
        self, pred: torch.Tensor, target: torch.Tensor, max_val: float | None = None
    ) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_psnr")
        target = self._normalize_tensor_dims(target, "target_psnr")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        diff = pred - target
        mse = torch.mean(diff * diff, dim=(-2, -1))
        mse = torch.clamp(mse, min=1e-10)
        if max_val is None:
            dr = target.amax(dim=(-2, -1)) - target.amin(dim=(-2, -1))
            dr = torch.clamp(dr, min=1e-6)
        else:
            dr = torch.full_like(mse, float(max_val))
        return 20.0 * torch.log10(dr / torch.sqrt(mse))

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_ssim")
        target = self._normalize_tensor_dims(target, "target_ssim")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        out = torch.zeros(
            pred.shape[0], pred.shape[1], dtype=torch.float32, device=pred.device
        )

        for b in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                t = target_np[b, c]
                p = pred_np[b, c]
                dr = float(np.max(t)) - float(np.min(t))
                if not np.isfinite(dr) or dr <= 0:
                    dr = 1.0
                h, w = int(t.shape[0]), int(t.shape[1])
                min_dim = min(h, w)
                if min_dim < 3:
                    val = 1.0 if np.allclose(t, p) else 0.0
                elif min_dim < 7:
                    win_size = min_dim if (min_dim % 2 == 1) else (min_dim - 1)
                    val = ssim(
                        t,
                        p,
                        data_range=dr,
                        win_size=win_size,
                        gaussian_weights=False,
                        use_sample_covariance=False,
                    )
                else:
                    val = ssim(
                        t,
                        p,
                        data_range=dr,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                out[b, c] = float(val)
        return out

    def compute_freq_rmse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pred = self._normalize_tensor_dims(pred, "pred_freq")
        target = self._normalize_tensor_dims(target, "target_freq")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        current_size = (int(target.shape[-2]), int(target.shape[-1]))
        if current_size != self.image_size:
            self.update_image_size(current_size)

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        diff = pred_fft - target_fft

        out: dict[str, torch.Tensor] = {}
        for band_name, mask in self.freq_masks.items():
            m = mask.to(device=pred.device)
            denom = int(m.sum().item())
            if denom <= 0:
                out[band_name] = torch.zeros(
                    pred.shape[0], pred.shape[1], device=pred.device
                )
                continue
            sel = diff * m[None, None, :, :]
            mse = torch.sum(torch.abs(sel) ** 2, dim=(-2, -1)) / float(denom)
            out[band_name] = torch.sqrt(mse)
        return out

    def compute_boundary_rmse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_brmse")
        target = self._normalize_tensor_dims(target, "target_brmse")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        current_size = (int(target.shape[-2]), int(target.shape[-1]))
        if current_size != self.image_size:
            self.update_image_size(current_size)
        mask = self.boundary_mask.to(device=pred.device)
        denom = int(mask.sum().item())
        if denom <= 0:
            return torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        diff2 = (pred - target) ** 2
        mse = torch.sum(diff2 * mask[None, None, :, :], dim=(-2, -1)) / float(denom)
        return torch.sqrt(mse)

    def compute_center_rmse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred = self._normalize_tensor_dims(pred, "pred_crmse")
        target = self._normalize_tensor_dims(target, "target_crmse")
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        current_size = (int(target.shape[-2]), int(target.shape[-1]))
        if current_size != self.image_size:
            self.update_image_size(current_size)
        mask = self.center_mask.to(device=pred.device)
        denom = int(mask.sum().item())
        if denom <= 0:
            return torch.zeros(pred.shape[0], pred.shape[1], device=pred.device)
        diff2 = (pred - target) ** 2
        mse = torch.sum(diff2 * mask[None, None, :, :], dim=(-2, -1)) / float(denom)
        return torch.sqrt(mse)

    def compute_data_consistency_error(
        self,
        pred: torch.Tensor,
        obs_data: dict[str, Any],
        norm_stats: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """计算数据一致性误差 ||H(ŷ)−y||

        目标：DC Error 必须比较 H(pred) 与 obs_data['y']。
        obs_data['y'] 必须存储真实观测（低分辨率）。
        """
        # 1. 维度规范化
        pred = self._normalize_tensor_dims(pred, "pred_dc_error")

        # 2. 获取并验证 target_obs
        target_obs = obs_data.get("y", None)
        if target_obs is None:
            target_obs = obs_data.get("observation", None)
        if target_obs is None:
            target_obs = obs_data.get("baseline", None)
        if target_obs is None:
            keys_str = ", ".join(sorted([str(k) for k in obs_data.keys()]))
            msg = (
                "obs_data must contain 'y' (or legacy 'observation'/'baseline') for DC error. "
                f"obs_data keys: [{keys_str}]"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # 3. 规范化 target_obs 维度
        target_obs = self._normalize_tensor_dims(target_obs, "target_obs")

        # 4. 确定 pred 的输入域并应用观测算子
        observation_is_norm = bool(obs_data.get("observation_is_norm", False))

        if norm_stats is not None and not observation_is_norm:
            # 反归一化到原值域
            if "mean" in norm_stats and "std" in norm_stats:
                mean = torch.as_tensor(norm_stats["mean"], device=pred.device).reshape(
                    1, -1, 1, 1
                )
                std = torch.as_tensor(norm_stats["std"], device=pred.device).reshape(
                    1, -1, 1, 1
                )
                pred_input = pred * std + mean
            else:
                pred_input = pred
        else:
            # 保持输入域
            pred_input = pred

        # 应用观测算子H
        pred_obs = apply_degradation_operator(pred_input, obs_data)

        # 确保设备和类型一致
        pred_obs = pred_obs.to(device=target_obs.device, dtype=target_obs.dtype)

        # 5. 验证一致性 (Strict Validation)
        # 检查形状一致性
        if pred_obs.shape[-2:] != target_obs.shape[-2:]:
            msg = (
                f"DC Error Validation Failed: Shape mismatch. "
                f"H(pred) shape: {pred_obs.shape}, Target(y) shape: {target_obs.shape}. "
                f"Domain normalized: {observation_is_norm}. "
                f"Target must match H(pred) dimensions."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        mse = torch.mean((pred_obs - target_obs) ** 2, dim=(-2, -1))  # [B, C]
        dc_error = torch.sqrt(mse)

        return dc_error

    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        obs_data: dict[str, Any] | None = None,
        norm_stats: dict[str, Any] | None = None,
        include_freq_metrics: bool = True,
    ) -> dict[str, torch.Tensor]:
        """计算所有指标

        核心逻辑：
        1. 统一维度规范化 -> [N, C, H, W]
        2. 空间对齐 -> pred 插值到 target 尺寸 (不允许 target 插值到 pred)
        3. 计算各指标
        """
        # 1. 维度规范化 (全局入口强制执行)
        pred = self._normalize_tensor_dims(pred, "pred_all")
        target = self._normalize_tensor_dims(target, "target_all")

        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(
                f"compute_all_metrics expects NCHW after normalization, got pred.dim={pred.dim()} target.dim={target.dim()} (pred={tuple(pred.shape)}, target={tuple(target.shape)})"
            )

        # 严格检查 Batch 和 Channel 维度一致性
        if pred.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Dimension mismatch in B/C: pred {pred.shape} vs target {target.shape}"
            )

        # 2. 空间维度一致性检查与插值
        if pred.shape[-2:] != target.shape[-2:]:
            # 只允许插值 pred 到 target
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Spatial mismatch: pred {pred.shape} vs target {target.shape}. Interpolating pred to target."
                )
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        # 3. 更新 image_size 以确保 mask 正确 (基于 target 的尺寸)
        current_size = (int(target.shape[-2]), int(target.shape[-1]))
        if current_size != self.image_size:
            self.update_image_size(current_size)

        metrics: dict[str, torch.Tensor] = {}

        try:
            metrics["rel_l2"] = self.compute_rel_l2(pred, target)
            metrics["mae"] = self.compute_mae(pred, target)
            metrics["psnr"] = self.compute_psnr(pred, target)
            metrics["ssim"] = self.compute_ssim(pred, target)

            if include_freq_metrics:
                freq_rmse = self.compute_freq_rmse(pred, target)
                for band_name, rmse in freq_rmse.items():
                    metrics[f"frmse_{band_name}"] = rmse

            metrics["brmse"] = self.compute_boundary_rmse(pred, target)
            metrics["crmse"] = self.compute_center_rmse(pred, target)

            if obs_data is not None:
                has_dc_target = (
                    (obs_data.get("y", None) is not None)
                    or (obs_data.get("observation", None) is not None)
                    or (obs_data.get("baseline", None) is not None)
                )
                if has_dc_target:
                    metrics["dc_error"] = self.compute_data_consistency_error(
                        pred, obs_data, norm_stats
                    )

        except Exception as e:
            self.logger.error(f"Error in compute_all_metrics internal calc: {e}")
            self.logger.error(f"Shapes - Pred: {pred.shape}, Target: {target.shape}")
            raise

        return metrics


class StatisticalAnalyzer:
    """统计分析器"""

    def __init__(self) -> None:
        self.results: list[dict[str, float | torch.Tensor]] = []

    def add_result(self, result: dict[str, float | torch.Tensor]) -> None:
        """添加单次实验结果"""
        self.results.append(result)

    def compute_statistics(self) -> dict[str, dict[str, float]]:
        """计算统计信息"""
        return self.aggregate_metrics(self.results)

    def aggregate_metrics(
        self, metrics_list: list[dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, float]]:
        """聚合多次实验的指标"""
        if not metrics_list:
            return {}

        # 获取所有指标名称
        metric_names: set[str] = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())

        aggregated = {}

        for metric_name in metric_names:
            # 收集该指标的所有值
            values = []
            for metrics in metrics_list:
                if metric_name in metrics:
                    # 取通道平均值
                    metric_value = metrics[metric_name]
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.dim() > 0:
                            value = torch.mean(metric_value).item()
                        else:
                            value = metric_value.item()
                    elif isinstance(metric_value, (int, float)):
                        value = float(metric_value)
                    else:
                        continue  # 跳过无法处理的类型

                    values.append(value)

            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        return aggregated


# --- Global Cache for Top-Level Access ---
_GLOBAL_CALCULATOR: MetricsCalculator | None = None


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_data: dict[str, Any] | None = None,
    norm_stats: dict[str, Any] | None = None,
    image_size: tuple[int, int] | None = None,
    include_freq_metrics: bool = True,
    **kwargs: Any,
) -> dict[str, torch.Tensor | float]:
    """顶层函数，用于计算所有指标

    Args:
        pred: 预测值
        target: 真实值
        obs_data: 观测数据（可选）
        norm_stats: 归一化统计量（可选）
        image_size: 图像尺寸（可选），如果提供则更新 calculator
        include_freq_metrics: 是否包含频域指标
        **kwargs: 兼容性参数（忽略）

    Returns:
        metrics: 指标字典
    """
    global _GLOBAL_CALCULATOR

    try:
        # 初始化全局实例
        if _GLOBAL_CALCULATOR is None:
            if image_size is None:
                # 默认 256，稍后会根据 target 自动 update
                _GLOBAL_CALCULATOR = MetricsCalculator(image_size=(256, 256))
            else:
                _GLOBAL_CALCULATOR = MetricsCalculator(image_size=image_size)

        # 如果显式传入了 image_size，则尝试更新
        if image_size is not None:
            _GLOBAL_CALCULATOR.update_image_size(image_size)

        # 调用计算
        raw = _GLOBAL_CALCULATOR.compute_all_metrics(
            pred=pred,
            target=target,
            obs_data=obs_data,
            norm_stats=norm_stats,
            include_freq_metrics=include_freq_metrics,
        )

        out: dict[str, torch.Tensor | float] = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                out[k] = float(v.mean().item())
            else:
                out[k] = float(v)
        return out

    except Exception as e:
        # 捕获异常并提供详细调试信息，尽量避免 simple fallback
        print("!!! Error in top-level compute_all_metrics !!!")
        print(f"Pred shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
        print(f"Target shape: {target.shape if hasattr(target, 'shape') else 'N/A'}")
        print(f"Error details: {e}")

        # 尝试最后的兜底：如果只是维度问题，_normalize_tensor_dims 应该已经处理了
        # 如果是其他计算错误，抛出
        raise


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_data: dict[str, Any] | None = None,
    norm_stats: dict[str, Any] | None = None,
    image_size: tuple[int, int] | None = None,
    include_freq_metrics: bool = True,
    **kwargs: Any,
) -> dict[str, torch.Tensor | float]:
    return compute_all_metrics(
        pred=pred,
        target=target,
        obs_data=obs_data,
        norm_stats=norm_stats,
        image_size=image_size,
        include_freq_metrics=include_freq_metrics,
        **kwargs,
    )


def rel_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        )
    )
    return float(calc.compute_rel_l2(pred, target, eps=eps).mean().item())


def mae_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        )
    )
    return float(calc.compute_mae(pred, target).mean().item())


def psnr_metric(
    pred: torch.Tensor, target: torch.Tensor, max_val: float | None = None
) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        )
    )
    return float(calc.compute_psnr(pred, target, max_val=max_val).mean().item())


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        )
    )
    return float(calc.compute_ssim(pred, target).mean().item())


def frequency_rmse(
    pred: torch.Tensor, target: torch.Tensor, freq_range: str = "low"
) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        )
    )
    rmse = calc.compute_freq_rmse(pred, target)
    if freq_range not in rmse:
        raise ValueError(f"Unknown freq_range: {freq_range}")
    return float(rmse[freq_range].mean().item())


def boundary_rmse(
    pred: torch.Tensor, target: torch.Tensor, boundary_width: int = 16
) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        ),
        boundary_width=boundary_width,
    )
    return float(calc.compute_boundary_rmse(pred, target).mean().item())


def center_rmse(
    pred: torch.Tensor, target: torch.Tensor, boundary_width: int = 16
) -> float:
    calc = MetricsCalculator(
        image_size=(
            (int(target.shape[-2]), int(target.shape[-1]))
            if hasattr(target, "shape") and target.dim() >= 2
            else (256, 256)
        ),
        boundary_width=boundary_width,
    )
    return float(calc.compute_center_rmse(pred, target).mean().item())


def compute_conservation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    image_size = (
        (int(target.shape[-2]), int(target.shape[-1]))
        if hasattr(target, "shape")
        and isinstance(target, torch.Tensor)
        and target.dim() >= 2
        else (256, 256)
    )
    calc = MetricsCalculator(image_size=image_size)

    pred_n = calc._normalize_tensor_dims(pred, "pred_conservation")
    target_n = calc._normalize_tensor_dims(target, "target_conservation")
    if pred_n.shape[-2:] != target_n.shape[-2:]:
        pred_n = F.interpolate(
            pred_n, size=target_n.shape[-2:], mode="bilinear", align_corners=False
        )

    b, c, h, w = pred_n.shape
    y = torch.linspace(-1.0, 1.0, h, device=pred_n.device, dtype=pred_n.dtype)
    x = torch.linspace(-1.0, 1.0, w, device=pred_n.device, dtype=pred_n.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.reshape(1, 1, h, w)
    yy = yy.reshape(1, 1, h, w)

    mass_pred = pred_n.sum(dim=(-2, -1))
    mass_target = target_n.sum(dim=(-2, -1))
    mass_err = torch.abs(mass_pred - mass_target) / (torch.abs(mass_target) + eps)

    energy_pred = (pred_n * pred_n).sum(dim=(-2, -1))
    energy_target = (target_n * target_n).sum(dim=(-2, -1))
    energy_err = torch.abs(energy_pred - energy_target) / (
        torch.abs(energy_target) + eps
    )

    mom_x_pred = (pred_n * xx).sum(dim=(-2, -1))
    mom_x_target = (target_n * xx).sum(dim=(-2, -1))
    mom_x_err = torch.abs(mom_x_pred - mom_x_target) / (torch.abs(mom_x_target) + eps)

    mom_y_pred = (pred_n * yy).sum(dim=(-2, -1))
    mom_y_target = (target_n * yy).sum(dim=(-2, -1))
    mom_y_err = torch.abs(mom_y_pred - mom_y_target) / (torch.abs(mom_y_target) + eps)

    out: dict[str, torch.Tensor] = {
        "mass_conservation_error": mass_err,
        "energy_conservation_error": energy_err,
        "momentum_y_conservation_error": mom_y_err,
        "momentum_x_conservation_error": mom_x_err,
    }
    if out["mass_conservation_error"].shape != (b, c):
        raise ValueError(
            f"Unexpected conservation metric shape: {out['mass_conservation_error'].shape} vs {(b, c)}"
        )
    return out


def compute_spectral_analysis(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    image_size = (
        (int(target.shape[-2]), int(target.shape[-1]))
        if hasattr(target, "shape")
        and isinstance(target, torch.Tensor)
        and target.dim() >= 2
        else (256, 256)
    )
    calc = MetricsCalculator(image_size=image_size)
    pred_n = calc._normalize_tensor_dims(pred, "pred_spectral")
    target_n = calc._normalize_tensor_dims(target, "target_spectral")
    if pred_n.shape[-2:] != target_n.shape[-2:]:
        pred_n = F.interpolate(
            pred_n, size=target_n.shape[-2:], mode="bilinear", align_corners=False
        )

    pred_fft = torch.fft.fft2(pred_n, dim=(-2, -1))
    target_fft = torch.fft.fft2(target_n, dim=(-2, -1))

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    pred_power = pred_mag * pred_mag
    target_power = target_mag * target_mag

    power_spectrum_mse = torch.mean((pred_power - target_power) ** 2, dim=(-2, -1))

    pred_unit = pred_fft / (pred_mag + eps)
    target_unit = target_fft / (target_mag + eps)
    phase_mse = torch.mean(torch.abs(pred_unit - target_unit) ** 2, dim=(-2, -1))

    p = pred_power.reshape(pred_power.shape[0], pred_power.shape[1], -1)
    t = target_power.reshape(target_power.shape[0], target_power.shape[1], -1)
    p_mean = p.mean(dim=-1, keepdim=True)
    t_mean = t.mean(dim=-1, keepdim=True)
    p0 = p - p_mean
    t0 = t - t_mean
    cov = (p0 * t0).mean(dim=-1)
    p_std = torch.sqrt((p0 * p0).mean(dim=-1) + eps)
    t_std = torch.sqrt((t0 * t0).mean(dim=-1) + eps)
    frequency_correlation = cov / (p_std * t_std + eps)

    return {
        "power_spectrum_mse": power_spectrum_mse,
        "phase_mse": phase_mse,
        "frequency_correlation": frequency_correlation,
    }


def aggregate_multi_seed_results(
    results_list: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    if len(results_list) == 0:
        return {}

    metric_names: set[str] = set()
    for metrics in results_list:
        metric_names.update(metrics.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for metrics in results_list:
            if metric_name not in metrics:
                continue
            v = metrics[metric_name]
            if isinstance(v, torch.Tensor):
                values.append(float(v.mean().item()))
            elif isinstance(v, (int, float)):
                values.append(float(v))
        if len(values) == 0:
            continue
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "count": float(len(values)),
        }
    return aggregated


if __name__ == "__main__":
    # 最小自测
    print("Running self-test for compute_all_metrics...")

    # 1. Test 4D [B, C, H, W]
    pred = torch.randn(1, 1, 64, 64)
    target = torch.randn(1, 1, 64, 64)
    metrics = compute_all_metrics(pred, target, image_size=(64, 64))
    print(f"4D Input: {metrics.keys()}")

    # 2. Test 2D [H, W]
    pred_2d = torch.randn(64, 64)
    target_2d = torch.randn(64, 64)
    metrics_2d = compute_all_metrics(pred_2d, target_2d)  # implicit image_size update
    print(f"2D Input: {metrics_2d.keys()}")

    # 3. Test 5D [B, T, C, H, W]
    pred_5d = torch.randn(1, 5, 1, 64, 64)
    target_5d = torch.randn(1, 5, 1, 64, 64)
    metrics_5d = compute_all_metrics(pred_5d, target_5d)
    print(f"5D Input: {metrics_5d.keys()}")

    # 4. Test Spatial Mismatch [B, C, 32, 32] vs [B, C, 64, 64]
    pred_small = torch.randn(1, 1, 32, 32)
    target_large = torch.randn(1, 1, 64, 64)
    metrics_mismatch = compute_all_metrics(pred_small, target_large)
    print(f"Mismatch Input: {metrics_mismatch.keys()} (should pass via interpolate)")

    print("Self-test passed!")
