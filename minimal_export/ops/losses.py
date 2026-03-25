"""损失函数系统

实现三件套损失：L = L_rec + λ_s L_spec + λ_dc L_dc
严格按照开发手册要求，确保值域正确处理。

同时支持自回归(AR)时序预测的损失函数。
"""

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# 为测试兼容提供CombinedLoss别名，复用utils.losses.TotalLoss实现
try:
    from utils.losses import TotalLoss as CombinedLoss  # noqa: F401
except Exception:
    CombinedLoss = None  # 在缺少依赖时保持可导入但不可用


class ARLoss(torch.nn.Module):
    """自回归时序预测损失函数

    支持多步预测的损失计算，包括：
    - 逐步损失：每个时间步的预测损失
    - 累积损失：整个序列的总损失
    - 教师强制损失：使用真实值作为输入的损失
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.get("loss_type", "mse")
        self.step_weights = config.get("step_weights", None)  # 每步的权重
        self.accumulate_loss = config.get("accumulate_loss", True)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: [B, T_out, C, H, W] 或 [B, C, H, W] 预测序列
            targets: [B, T_out, C, H, W] 或 [B, C, H, W] 目标序列
            mask: [B, T_out, C, H, W] 可选的掩码

        Returns:
            Dict包含step_losses, total_loss等
        """
        # 处理不同维度的输入
        if len(predictions.shape) == 4:
            # 如果是4D，添加时间维度
            predictions = predictions.unsqueeze(1)  # [B, 1, C, H, W]
        if len(targets.shape) == 4:
            targets = targets.unsqueeze(1)  # [B, 1, C, H, W]

        B, T_out, C, H, W = predictions.shape

        # 计算每个时间步的损失
        step_losses = []
        for t in range(T_out):
            pred_t = predictions[:, t]  # [B, C, H, W]
            target_t = targets[:, t]  # [B, C, H, W]

            if self.loss_type == "mse":
                loss_t = F.mse_loss(pred_t, target_t, reduction="none")
            elif self.loss_type == "l1":
                loss_t = F.l1_loss(pred_t, target_t, reduction="none")
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            # 应用掩码
            if mask is not None:
                mask_t = mask[:, t]
                loss_t = loss_t * mask_t
                loss_t = loss_t.sum() / (mask_t.sum() + 1e-8)
            else:
                loss_t = loss_t.mean()

            step_losses.append(loss_t)

        # 计算加权总损失
        if self.step_weights is not None:
            weights = torch.tensor(self.step_weights, device=predictions.device)
            total_loss = sum(w * loss for w, loss in zip(weights, step_losses))
        else:
            total_loss = sum(step_losses) / len(step_losses)

        return {
            "step_losses": step_losses,
            "total_loss": total_loss,
            "ar_loss": total_loss,
        }


class SpectralLoss(torch.nn.Module):
    """频谱损失函数"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.k_max = config.get("k_max", 16)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频谱损失"""
        # 计算FFT
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        # 只比较低频部分
        pred_fft_low = pred_fft[..., : self.k_max, : self.k_max]
        target_fft_low = target_fft[..., : self.k_max, : self.k_max]

        # 计算损失
        loss = F.mse_loss(pred_fft_low.real, target_fft_low.real) + F.mse_loss(
            pred_fft_low.imag, target_fft_low.imag
        )

        return loss


class DCLoss(torch.nn.Module):
    """数据一致性损失函数"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, pred_obs: torch.Tensor, target_obs: torch.Tensor) -> torch.Tensor:
        """计算数据一致性损失

        Args:
            pred_obs: 经过观测算子H处理后的预测值 [B, C, H, W]
            target_obs: 观测数据 [B, C, H, W]
        """
        # 确保尺寸匹配
        if pred_obs.shape != target_obs.shape:
            # 如果尺寸不匹配，将target_obs调整到pred_obs的尺寸
            target_obs = F.interpolate(
                target_obs,
                size=pred_obs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = F.mse_loss(pred_obs, target_obs)

        return loss


def compute_total_loss_base(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    obs_data: dict,
    norm_stats: dict[str, torch.Tensor] | None,
    config: DictConfig,
    loss_weights_override: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """计算总损失，包含重建损失、频谱损失和数据一致性损失

    **值域说明**：
    - 模型输出默认在z-score域（标准化后）
    - DC损失和谱损失在原值域计算（需反归一化：pred_orig = pred_z * sigma + mu）
    - 重建损失可在z-score域直接计算

    **损失计算规则**：
    - 输入期望：pred_z, target_z（z-score域），norm_stats（归一化统计量）
    - 频域损失：默认只比较前kx=ky=16的rFFT系数，非周期边界用镜像延拓
    - DC验收：对GT调用H与生成y的MSE < 1e-8视为通过

    Args:
        pred_z: 模型预测（z-score域）[B, C, H, W]
        target_z: 真值标签（z-score域）[B, C, H, W]
        obs_data: 观测数据字典，包含baseline、mask、coords、h_params、observation
        norm_stats: 归一化统计量，用于反归一化到原值域
        config: 损失权重配置

    Returns:
        Dict包含各损失分量：reconstruction_loss, spectral_loss, dc_loss, total_loss
    """
    device = pred_z.device
    B, C, H, W = pred_z.shape

    # 获取损失权重（健壮兼容：支持 train.loss_weights 与 loss.* 两种结构）
    w_rec = 1.0
    w_spec = 0.0
    w_dc = 0.0
    w_grad = 0.0

    # 优先使用 loss_weights_override 参数
    if loss_weights_override is not None:
        w_rec = loss_weights_override.get("reconstruction", w_rec)
        w_spec = loss_weights_override.get("spectral", w_spec)
        w_dc = loss_weights_override.get("data_consistency", w_dc)
        w_grad = loss_weights_override.get("gradient", w_grad)
    else:
        # 优先使用 train.loss_weights 结构
        has_train_loss_weights = hasattr(config, "train") and hasattr(
            config.train, "loss_weights"
        )
        if has_train_loss_weights:
            try:
                if hasattr(config.train.loss_weights, "reconstruction"):
                    w_rec = float(config.train.loss_weights.reconstruction)
                if hasattr(config.train.loss_weights, "spectral"):
                    w_spec = float(config.train.loss_weights.spectral)
                if hasattr(config.train.loss_weights, "data_consistency"):
                    w_dc = float(config.train.loss_weights.data_consistency)
                # 可选：梯度项
                if hasattr(config.train.loss_weights, "gradient"):
                    w_grad = float(getattr(config.train.loss_weights, "gradient", 0.0))
            except Exception:
                # 若读取失败，回退到默认值
                pass

        # 兼容旧版 loss.* 结构
        if hasattr(config, "loss") and not has_train_loss_weights:
            # reconstruction
            if hasattr(config.loss, "reconstruction") and hasattr(
                config.loss.reconstruction, "weight"
            ):
                try:
                    w_rec = float(config.loss.reconstruction.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, "reconstruction") and isinstance(
                config.loss.reconstruction, (int, float)
            ):
                w_rec = float(config.loss.reconstruction)
            # spectral
            if hasattr(config.loss, "spectral") and hasattr(
                config.loss.spectral, "weight"
            ):
                try:
                    w_spec = float(config.loss.spectral.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, "spectral") and isinstance(
                config.loss.spectral, (int, float)
            ):
                w_spec = float(config.loss.spectral)
            # data consistency aliases
            if hasattr(config.loss, "data_consistency") and hasattr(
                config.loss.data_consistency, "weight"
            ):
                try:
                    w_dc = float(config.loss.data_consistency.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, "degradation_consistency") and hasattr(
                config.loss.degradation_consistency, "weight"
            ):
                try:
                    w_dc = float(config.loss.degradation_consistency.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, "data_consistency") and isinstance(
                config.loss.data_consistency, (int, float)
            ):
                w_dc = float(config.loss.data_consistency)
            elif hasattr(config.loss, "degradation_consistency") and isinstance(
                config.loss.degradation_consistency, (int, float)
            ):
                w_dc = float(config.loss.degradation_consistency)
            # 梯度项
            w_grad = getattr(config.loss, "gradient_weight", w_grad)

    pred_z = torch.nan_to_num(pred_z, nan=0.0, posinf=1e6, neginf=-1e6)
    target_z = torch.nan_to_num(target_z, nan=0.0, posinf=1e6, neginf=-1e6)
    losses = {}

    # 1. 重建损失（在z-score域计算）
    # 支持选择 r2 作为唯一重建损失（通过配置 loss.reconstruction.type: r2）
    use_r2 = False
    try:
        if hasattr(config, "loss") and hasattr(config.loss, "reconstruction"):
            use_r2 = (
                str(getattr(config.loss.reconstruction, "type", "")).lower() == "r2"
            )
    except Exception:
        use_r2 = False
    if use_r2:
        reconstruction_loss = _compute_r2_loss(pred_z, target_z)
    else:
        reconstruction_loss = _compute_reconstruction_loss(pred_z, target_z, obs_data)
    losses["reconstruction_loss"] = reconstruction_loss

    # 2. 频谱损失（在原值域计算）
    if w_spec > 0:
        # 获取数据键，支持不同的配置结构
        data_keys = config.data.get("keys", None) if hasattr(config, "data") else None
        if data_keys is None:
            # 如果没有keys，使用默认的反归一化
            pred_orig = _denormalize_tensor(pred_z, norm_stats, None)
            target_orig = _denormalize_tensor(target_z, norm_stats, None)
        else:
            pred_orig = _denormalize_tensor(pred_z, norm_stats, data_keys)
            target_orig = _denormalize_tensor(target_z, norm_stats, data_keys)
        spectral_loss = _compute_spectral_loss(pred_orig, target_orig, config)
        losses["spectral_loss"] = spectral_loss
    else:
        losses["spectral_loss"] = torch.tensor(0.0, device=device)

    # 3. 数据一致性损失（在原值域计算）
    if w_dc > 0:
        # 使用预先计算好的 pred_obs（由Trainer提供）
        pred_obs = obs_data.get("pred_obs")
        if pred_obs is not None:
            dc_loss = _compute_data_consistency_loss(pred_obs, obs_data)
        else:
            # 如果未提供pred_obs，则跳过DC损失（禁止在Loss中调用degradation）
            dc_loss = torch.tensor(0.0, device=device)
        losses["dc_loss"] = dc_loss
    else:
        losses["dc_loss"] = torch.tensor(0.0, device=device)

    # 4. 梯度损失（可选，在z-score域计算）
    if w_grad > 0:
        gradient_loss = _compute_gradient_loss(pred_z, target_z)
        losses["gradient_loss"] = gradient_loss
    else:
        losses["gradient_loss"] = torch.tensor(0.0, device=device)

    # 5. 总损失
    total_loss = (
        w_rec * losses["reconstruction_loss"]
        + w_spec * losses["spectral_loss"]
        + w_dc * losses["dc_loss"]
        + w_grad * losses["gradient_loss"]
    )
    losses["total_loss"] = total_loss

    return losses


def compute_total_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    obs_data: dict,
    norm_stats: dict[str, torch.Tensor] | None,
    config: DictConfig,
) -> dict[str, torch.Tensor]:
    """时序损失（供测试与真实任务使用）

    - 输入为 z-score 域；频谱与 DC 在原值域计算。
    - 支持 4D [B,C,H,W]（单步）与 5D [B,T_out,C,H,W]（多步）。
    - 多步时对各分量按时间维平均，保持与黄金法则一致。
    """
    if pred_z.dim() == 4 and target_z.dim() == 4:
        return compute_total_loss_base(
            pred_z, target_z, obs_data or {}, norm_stats, config
        )

    if pred_z.dim() != 5 or target_z.dim() != 5:
        raise ValueError("compute_temporal_loss expects 4D or 5D tensors")

    device = pred_z.device
    B, T_out, C, H, W = pred_z.shape

    rec_list = []
    spec_list = []
    dc_list = []
    grad_list = []

    for t in range(T_out):
        pred_t = pred_z[:, t]
        target_t = target_z[:, t]

        # 构造步级 obs_data
        obs_t: dict[str, Any] = {}
        if isinstance(obs_data, dict):
            # 观测序列
            if "observation" in obs_data:
                obs_val = obs_data["observation"]
                obs_t["observation"] = (
                    obs_val[:, t]
                    if isinstance(obs_val, torch.Tensor) and obs_val.dim() == 5
                    else obs_val
                )
            # baseline/mask 可能带时间维
            if "baseline" in obs_data:
                base_val = obs_data["baseline"]
                obs_t["baseline"] = (
                    base_val[:, t]
                    if isinstance(base_val, torch.Tensor) and base_val.dim() == 5
                    else base_val
                )
            if "mask" in obs_data:
                mask_val = obs_data["mask"]
                obs_t["mask"] = (
                    mask_val[:, t]
                    if isinstance(mask_val, torch.Tensor) and mask_val.dim() == 5
                    else mask_val
                )
            # 预计算的观测预测 pred_obs 也可能带时间维
            if "pred_obs" in obs_data:
                pred_obs_val = obs_data["pred_obs"]
                obs_t["pred_obs"] = (
                    pred_obs_val[:, t]
                    if isinstance(pred_obs_val, torch.Tensor)
                    and pred_obs_val.dim() == 5
                    else pred_obs_val
                )
            # H 参数直接复用
            if "h_params" in obs_data:
                obs_t["h_params"] = obs_data["h_params"]
        elif isinstance(obs_data, torch.Tensor):
            # 直接给出观测序列张量
            if obs_data.dim() == 5:
                obs_t["observation"] = obs_data[:, t]
        else:
            obs_t = {}

        losses_t = compute_total_loss_base(pred_t, target_t, obs_t, norm_stats, config)
        rec_list.append(
            losses_t.get("reconstruction_loss", torch.tensor(0.0, device=device))
        )
        spec_list.append(
            losses_t.get("spectral_loss", torch.tensor(0.0, device=device))
        )
        dc_list.append(losses_t.get("dc_loss", torch.tensor(0.0, device=device)))
        grad_list.append(
            losses_t.get("gradient_loss", torch.tensor(0.0, device=device))
        )

    def _mean_stack(lst: list[torch.Tensor]) -> torch.Tensor:
        if not lst:
            return torch.tensor(0.0, device=device)
        return torch.stack(lst).mean()

    reconstruction_loss = _mean_stack(rec_list)
    spectral_loss = _mean_stack(spec_list)
    dc_loss = _mean_stack(dc_list)
    gradient_loss = _mean_stack(grad_list)

    # 读取权重，遵循 compute_total_loss 的约定
    w_rec = 1.0
    w_spec = 0.0
    w_dc = 0.0
    w_grad = 0.0

    has_train_loss_weights = hasattr(config, "training") and hasattr(
        config.training, "loss_weights"
    )
    if has_train_loss_weights:
        try:
            if hasattr(config.training.loss_weights, "reconstruction"):
                w_rec = float(config.training.loss_weights.reconstruction)
            if hasattr(config.training.loss_weights, "spectral"):
                w_spec = float(config.training.loss_weights.spectral)
            if hasattr(config.training.loss_weights, "data_consistency"):
                w_dc = float(config.training.loss_weights.data_consistency)
            if hasattr(config.training.loss_weights, "gradient"):
                w_grad = float(getattr(config.training.loss_weights, "gradient", 0.0))
        except Exception:
            pass
    elif hasattr(config, "loss"):
        if hasattr(config.loss, "reconstruction") and hasattr(
            config.loss.reconstruction, "weight"
        ):
            try:
                w_rec = float(config.loss.reconstruction.weight)
            except Exception:
                pass
        elif hasattr(config.loss, "reconstruction") and isinstance(
            config.loss.reconstruction, (int, float)
        ):
            w_rec = float(config.loss.reconstruction)
        if hasattr(config.loss, "spectral") and hasattr(config.loss.spectral, "weight"):
            try:
                w_spec = float(config.loss.spectral.weight)
            except Exception:
                pass
        elif hasattr(config.loss, "spectral") and isinstance(
            config.loss.spectral, (int, float)
        ):
            w_spec = float(config.loss.spectral)
        if hasattr(config.loss, "data_consistency") and hasattr(
            config.loss.data_consistency, "weight"
        ):
            try:
                w_dc = float(config.loss.data_consistency.weight)
            except Exception:
                pass
        elif hasattr(config.loss, "degradation_consistency") and hasattr(
            config.loss.degradation_consistency, "weight"
        ):
            try:
                w_dc = float(config.loss.degradation_consistency.weight)
            except Exception:
                pass
        elif hasattr(config.loss, "data_consistency") and isinstance(
            config.loss.data_consistency, (int, float)
        ):
            w_dc = float(config.loss.data_consistency)
        elif hasattr(config.loss, "degradation_consistency") and isinstance(
            config.loss.degradation_consistency, (int, float)
        ):
            w_dc = float(config.loss.degradation_consistency)
        w_grad = getattr(config.loss, "gradient_weight", w_grad)

    total_loss = (
        w_rec * reconstruction_loss
        + w_spec * spectral_loss
        + w_dc * dc_loss
        + w_grad * gradient_loss
    )

    return {
        "reconstruction_loss": reconstruction_loss,
        "spectral_loss": spectral_loss,
        "dc_loss": dc_loss,
        "gradient_loss": gradient_loss,
        "total_loss": total_loss,
    }


def _compute_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, obs_data: dict
) -> torch.Tensor:
    """计算重建损失

    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]
        obs_data: 观测数据

    Returns:
        重建损失
    """
    # 使用相对L2损失作为主要重建损失
    rel_l2 = _compute_relative_l2_loss(pred, target)

    # 可选：添加MAE损失
    mae = F.l1_loss(pred, target)

    # 组合损失（主要使用Rel-L2）
    reconstruction_loss = rel_l2 + 0.1 * mae

    return reconstruction_loss


def _compute_r2_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """R²损失（1 − R²），在 z-score 域计算。

    R² = 1 − SSE/SST，其中 SSE = Σ(y−ŷ)²，SST = Σ(y−ȳ)²。
    当 SST≈0（目标方差接近0）时，回退为 MSE。
    返回批次与通道平均的标量损失。
    """
    # [B, C, H, W]
    B, C = target.size(0), target.size(1)
    # 展平空间维度
    pred_f = pred.view(B, C, -1)
    tgt_f = target.view(B, C, -1)
    # SSE
    sse = torch.sum((tgt_f - pred_f) ** 2, dim=2)  # [B, C]
    # SST
    tgt_mean = torch.mean(tgt_f, dim=2, keepdim=True)  # [B, C, 1]
    sst = torch.sum((tgt_f - tgt_mean) ** 2, dim=2)  # [B, C]
    # 处理零方差：回退为MSE
    denom = torch.clamp(sst, min=eps)
    r2 = 1.0 - (sse / denom)  # [B, C]
    # 将异常值裁剪到合理范围
    r2 = torch.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
    loss = 1.0 - r2  # 1 − R²
    return loss.mean()


def _compute_spectral_loss(
    pred: torch.Tensor, target: torch.Tensor, config: DictConfig
) -> torch.Tensor:
    """计算频谱损失

    仅比较前kx=ky=16的rFFT系数，非周期边界用镜像延拓

    Args:
        pred: 预测（原值域）[B, C, H, W]
        target: 真值（原值域）[B, C, H, W]
        config: 配置

    Returns:
        频谱损失
    """
    # 兼容读取：优先 train.spectral_loss.*，其次 loss.*
    low_freq_modes = 16
    use_rfft = False
    normalize = False
    boundary_mode = None

    if hasattr(config, "train") and hasattr(config.train, "spectral_loss"):
        low_freq_modes = getattr(
            config.train.spectral_loss, "low_freq_modes", low_freq_modes
        )
        use_rfft = getattr(config.train.spectral_loss, "use_rfft", use_rfft)
        normalize = getattr(config.train.spectral_loss, "normalize", normalize)
        boundary_mode = getattr(
            config.train.spectral_loss, "boundary_mode", boundary_mode
        )

    if hasattr(config, "loss"):
        low_freq_modes = getattr(config.loss, "low_freq_modes", low_freq_modes)
        use_rfft = getattr(config.loss, "use_rfft", use_rfft)
        normalize = getattr(config.loss, "normalize", normalize)
        # 统一边界策略：mirror/zero/none（none表示周期）
        if hasattr(config.loss, "spectral") and hasattr(
            config.loss.spectral, "boundary_mode"
        ):
            boundary_mode = getattr(
                config.loss.spectral, "boundary_mode", boundary_mode
            )
        elif hasattr(config.loss, "boundary_mode"):
            boundary_mode = getattr(config.loss, "boundary_mode", boundary_mode)
    boundary_mode = boundary_mode or "mirror"

    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    # 确保FFT使用支持的dtype
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    B, C, H, W = pred.shape

    # 边界处理：镜像延拓 / 零填充 / 无处理（周期）
    if (boundary_mode or "mirror") == "mirror":
        pred_extended = _mirror_extend(pred)
        target_extended = _mirror_extend(target)
    elif boundary_mode == "zero":
        pred_extended = _zero_extend(pred)
        target_extended = _zero_extend(target)
    else:
        # none/periodic: 不进行延拓，直接FFT
        pred_extended = pred
        target_extended = target
    # 再次确保扩展后为float32
    pred_extended = pred_extended.to(torch.float32)
    target_extended = target_extended.to(torch.float32)

    spectral_losses = []

    for c in range(C):
        pred_c = pred_extended[:, c]  # [B, H_ext, W_ext]
        target_c = target_extended[:, c]

        if use_rfft:
            # 使用实数FFT
            pred_fft = torch.fft.rfft2(
                pred_c.to(torch.float32), norm="ortho" if normalize else None
            )
            target_fft = torch.fft.rfft2(
                target_c.to(torch.float32), norm="ortho" if normalize else None
            )
        else:
            # 使用复数FFT
            pred_fft = torch.fft.fft2(
                pred_c.to(torch.float32), norm="ortho" if normalize else None
            )
            target_fft = torch.fft.fft2(
                target_c.to(torch.float32), norm="ortho" if normalize else None
            )

        # 只比较低频部分
        low_freq_modes_int = int(low_freq_modes)
        pred_fft_low = pred_fft[:, :low_freq_modes_int, :low_freq_modes_int]
        target_fft_low = target_fft[:, :low_freq_modes_int, :low_freq_modes_int]

        # 计算频谱损失（使用相对L2损失）
        # ||F_pred - F_target||^2 / (||F_target||^2 + eps)
        diff_real = torch.nan_to_num(pred_fft_low.real) - torch.nan_to_num(
            target_fft_low.real
        )
        diff_imag = torch.nan_to_num(pred_fft_low.imag) - torch.nan_to_num(
            target_fft_low.imag
        )

        target_real = torch.nan_to_num(target_fft_low.real)
        target_imag = torch.nan_to_num(target_fft_low.imag)

        diff_sq = diff_real**2 + diff_imag**2
        target_sq = target_real**2 + target_imag**2 + 1e-8

        spectral_loss_c = diff_sq.sum() / target_sq.sum()

        spectral_losses.append(spectral_loss_c)

    # 多通道平均
    spectral_loss = torch.stack(spectral_losses).mean()

    return spectral_loss


def _compute_data_consistency_loss(
    pred_obs: torch.Tensor,
    obs_data: dict,
) -> torch.Tensor:
    """计算数据一致性损失

    DC损失：‖H(ŷ)−y‖₂

    Args:
        pred_obs: 经过观测算子H处理后的预测值 [B, C, H, W]
        obs_data: 观测数据字典

    Returns:
        数据一致性损失
    """
    # 获取对应的观测数据（原值域）
    observation = obs_data.get("observation")
    if observation is None:
        # 如果没有observation，尝试从baseline获取（注意：这里假设baseline已经是观测域或者能通过外部逻辑转换，
        # 但Loss函数本身不再负责转换。如果baseline是z-score域且无H，无法计算DC）
        return torch.tensor(0.0, device=pred_obs.device)

    # 确保observation在原值域且维度匹配
    if observation.shape != pred_obs.shape:
        # 检查维度是否匹配
        if observation.dim() != pred_obs.dim():
            return torch.tensor(0.0, device=pred_obs.device)

        # 检查通道数是否匹配
        if observation.shape[1] != pred_obs.shape[1]:
            # 调整通道数
            if observation.shape[1] > pred_obs.shape[1]:
                observation = observation[:, : pred_obs.shape[1]]
            else:
                return torch.tensor(0.0, device=pred_obs.device)

        # 调整空间尺寸
        if observation.shape[-2:] != pred_obs.shape[-2:]:
            observation = F.interpolate(
                observation,
                size=pred_obs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

    # 计算DC损失
    dc_loss = F.mse_loss(pred_obs, observation)

    return dc_loss


def _compute_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算梯度损失

    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]

    Returns:
        梯度损失
    """
    # 计算梯度
    # 差分近似梯度；针对小尺寸（H<2或W<2）避免空张量导致的NaN
    B, C, H, W = pred.shape

    components = []
    if W > 1:
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        components.append(F.l1_loss(pred_grad_x, target_grad_x))
    if H > 1:
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        components.append(F.l1_loss(pred_grad_y, target_grad_y))

    if not components:
        # 单像素或无法计算梯度的情况，返回0
        return torch.tensor(0.0, device=pred.device)

    gradient_loss = sum(components)

    return gradient_loss


def _compute_relative_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """计算相对L2损失

    Rel-L2 = ‖pred - target‖₂ / (‖target‖₂ + eps)

    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]
        eps: 数值稳定性常数

    Returns:
        相对L2损失
    """
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    # 计算每个样本的相对L2损失
    diff_norm = torch.norm(pred - target, p=2, dim=(1, 2, 3))  # [B]
    target_norm = torch.norm(target, p=2, dim=(1, 2, 3))  # [B]

    rel_l2 = diff_norm / (target_norm + eps)

    # 返回批次平均
    rel = diff_norm / (target_norm + eps)
    rel = torch.nan_to_num(rel, nan=0.0, posinf=1e6, neginf=1e6)
    return rel.mean()


def _denormalize_tensor(
    tensor_z: torch.Tensor, norm_stats: dict[str, torch.Tensor] | None, keys: list
) -> torch.Tensor:
    """反归一化张量到原值域

    Args:
        tensor_z: z-score域张量 [B, C, H, W]
        norm_stats: 归一化统计量
        keys: 数据键名列表

    Returns:
        原值域张量
    """
    if norm_stats is None:
        return tensor_z

    tensor_orig = tensor_z.clone()

    # 回退策略：当未提供keys时，尝试使用通道级 mean/std 或全局 mean/std
    if keys is None:
        # 支持字典形式：{'mean': Tensor[C], 'std': Tensor[C]} 或标量
        if isinstance(norm_stats, dict) and (
            "mean" in norm_stats and "std" in norm_stats
        ):
            mean = norm_stats["mean"].to(tensor_z.device)
            std = norm_stats["std"].to(tensor_z.device)
            if mean.dim() == 0:
                mean = mean.repeat(tensor_z.size(1))
            if std.dim() == 0:
                std = std.repeat(tensor_z.size(1))
            tensor_orig = tensor_z * std.reshape(1, -1, 1, 1) + mean.reshape(
                1, -1, 1, 1
            )
            return tensor_orig
        # 支持列表/元组 (mean, std)
        if isinstance(norm_stats, (tuple, list)) and len(norm_stats) == 2:
            mean = norm_stats[0].to(tensor_z.device)
            std = norm_stats[1].to(tensor_z.device)
            if mean.dim() == 0:
                mean = mean.repeat(tensor_z.size(1))
            if std.dim() == 0:
                std = std.repeat(tensor_z.size(1))
            tensor_orig = tensor_z * std.reshape(1, -1, 1, 1) + mean.reshape(
                1, -1, 1, 1
            )
            return tensor_orig
        # 若无法推断，直接返回原张量（保持z-score域）
        return tensor_z

    for i, key in enumerate(keys):
        if i >= tensor_z.size(1):
            break

        mean_key = f"{key}_mean"
        std_key = f"{key}_std"

        if mean_key in norm_stats and std_key in norm_stats:
            mean = norm_stats[mean_key].to(tensor_z.device)
            std = norm_stats[std_key].to(tensor_z.device)

            # 确保mean和std的形状正确
            if mean.dim() == 0:
                mean = mean.unsqueeze(0)
            if std.dim() == 0:
                std = std.unsqueeze(0)

            # 反归一化：x_orig = x_z * std + mean
            tensor_orig[:, i : i + 1] = tensor_z[:, i : i + 1] * std.reshape(
                1, 1, 1, 1
            ) + mean.reshape(1, 1, 1, 1)
        else:
            # 回退到全局通道级 mean/std
            if isinstance(norm_stats, dict) and (
                "mean" in norm_stats and "std" in norm_stats
            ):
                mean = norm_stats["mean"].to(tensor_z.device)
                std = norm_stats["std"].to(tensor_z.device)
                if mean.dim() == 0:
                    mean = mean.repeat(tensor_z.size(1))
                if std.dim() == 0:
                    std = std.repeat(tensor_z.size(1))
                tensor_orig[:, i : i + 1] = tensor_z[:, i : i + 1] * std[i].reshape(
                    1, 1, 1, 1
                ) + mean[i].reshape(1, 1, 1, 1)
            else:
                print(
                    f"Warning: No normalization stats found for key '{key}', keeping original values"
                )

    return tensor_orig


def _mirror_extend(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """镜像延拓张量（用于非周期边界的FFT）

    Args:
        x: 输入张量 [B, C, H, W]
        factor: 延拓倍数

    Returns:
        延拓后的张量 [B, C, H*factor, W*factor]
    """
    B, C, H, W = x.shape

    # 水平镜像
    x_h_mirror = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)  # [B, C, H, 2W]

    # 垂直镜像
    x_extended = torch.cat(
        [x_h_mirror, torch.flip(x_h_mirror, dims=[-2])], dim=-2
    )  # [B, C, 2H, 2W]

    return x_extended


def _zero_extend(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """零填充延拓张量到更大尺寸

    Args:
        x: 输入张量 [B, C, H, W]
        factor: 延拓倍数（默认为2）
    Returns:
        延拓后的张量 [B, C, H*factor, W*factor]
    """
    B, C, H, W = x.shape
    pad_w = W * (factor - 1)
    pad_h = H * (factor - 1)
    # 仅在右侧和下方进行零填充，保持原图在左上角
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x_padded


def compute_loss_weights_schedule(
    epoch: int, total_epochs: int, base_weights: dict[str, float]
) -> dict[str, float]:
    """计算损失权重调度

    可以实现课程学习，例如：
    - 早期阶段重点关注重建损失
    - 后期阶段增加数据一致性损失权重

    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        base_weights: 基础权重

    Returns:
        调度后的权重
    """
    progress = epoch / total_epochs

    # 简单的线性调度示例
    weights = {}

    # 处理每个权重，确保从DictConfig中提取数值
    for key, value in base_weights.items():
        # 如果value是DictConfig，提取其中的weight字段
        if hasattr(value, "weight"):
            base_weight = float(value.weight)
        elif hasattr(value, "_content") and isinstance(value._content, dict):
            # 处理嵌套的DictConfig
            if "weight" in value._content:
                base_weight = float(value._content["weight"])
            else:
                base_weight = 1.0  # 默认权重
        elif isinstance(value, str):
            # 如果是字符串，跳过或设置默认值
            if (
                key == "rec_loss_type"
                or key == "spec_loss_type"
                or key == "dc_loss_type"
            ):
                continue  # 跳过损失类型配置
            else:
                base_weight = 1.0  # 默认权重
        elif hasattr(value, "__dict__"):
            # 处理其他类型的配置对象
            try:
                base_weight = float(value)
            except (TypeError, ValueError):
                base_weight = 1.0  # 默认权重
        else:
            try:
                base_weight = float(value)
            except (TypeError, ValueError):
                base_weight = 1.0  # 默认权重

        weights[key] = base_weight

    # DC损失权重随训练进度增加
    if "data_consistency" in weights:
        weights["data_consistency"] = weights["data_consistency"] * (
            0.1 + 0.9 * progress
        )

    # 频谱损失权重在中期达到峰值
    if "spectral" in weights:
        spectral_factor = 4 * progress * (1 - progress)  # 在0.5处达到峰值1.0
        weights["spectral"] = weights["spectral"] * (0.5 + 0.5 * spectral_factor)

    return weights


def l1_mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算L1 MAE损失

    Args:
        x: 预测张量
        y: 目标张量

    Returns:
        L1 MAE损失
    """
    return (x - y).abs().mean()


def rel_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算相对L2损失，兼容 [B,T,C,H,W] 与 [B,C,H,W]"""
    if x.dim() == 5 and y.dim() == 5:
        num = torch.sqrt(((x - y) ** 2).sum(dim=(2, 3, 4)))  # [B, T]
        den = torch.sqrt((y**2).sum(dim=(2, 3, 4))) + eps  # [B, T]
        return (num / den).mean()
    if x.dim() == 4 and y.dim() == 4:
        num = torch.sqrt(((x - y) ** 2).sum(dim=(1, 2, 3)))  # [B]
        den = torch.sqrt((y**2).sum(dim=(1, 2, 3))) + eps  # [B]
        return (num / den).mean()
    raise ValueError(
        f"Unsupported shapes for rel_l2: x={tuple(x.shape)}, y={tuple(y.shape)}"
    )


def compute_ar_loss(
    pred_seq: torch.Tensor, gt_seq: torch.Tensor, cfg_loss: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算自回归模型的损失

    Args:
        pred_seq: 预测序列 [B, T_out, C, H, W]
        gt_seq: 真值序列 [B, T_out, C, H, W]
        cfg_loss: 损失配置

    Returns:
        总损失和损失项字典
    """
    assert gt_seq is not None, "AR训练需要teacher（target_seq）"

    # 获取损失权重
    w_rel2 = cfg_loss.get("rel2_weight", 1.0)
    w_mae = cfg_loss.get("mae_weight", 0.1)

    # 计算损失
    rel2_loss = rel_l2(pred_seq, gt_seq)
    mae_loss = l1_mae(pred_seq, gt_seq)

    # 总损失
    loss = w_rel2 * rel2_loss + w_mae * mae_loss

    # 返回总损失和损失项
    loss_items = {"rel2": rel2_loss.item(), "mae": mae_loss.item()}

    return loss, loss_items


def compute_ar_total_loss(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    obs_data: dict,
    norm_stats: dict[str, torch.Tensor] | None,
    config: DictConfig,
) -> dict[str, torch.Tensor]:
    """计算自回归模型的总损失，包含重建损失、频谱损失和数据一致性损失"""
    device = pred_seq.device
    # 输入稳定化
    pred_seq = torch.nan_to_num(pred_seq, nan=0.0, posinf=1e3, neginf=-1e3)
    gt_seq = torch.nan_to_num(gt_seq, nan=0.0, posinf=1e3, neginf=-1e3)
    # 对齐时间长度（课程阶段切换时可能出现 T_pred≠T_gt）
    try:
        T_pred = int(pred_seq.shape[1])
        T_gt = int(gt_seq.shape[1])
        T_use = min(T_pred, T_gt)
        if T_pred != T_use:
            pred_seq = pred_seq[:, :T_use]
        if T_gt != T_use:
            gt_seq = gt_seq[:, :T_use]
    except Exception:
        pass
    B, T, C, H, W = pred_seq.shape

    # -------------------------------------------------------------------------
    # 统一权重读取逻辑 (优先 training.loss_weights > loss.*)
    # -------------------------------------------------------------------------
    w_rec_scale = 1.0  # 全局重建权重缩放
    w_spec = 0.0
    w_dc = 0.0

    # 1. 尝试从 training.loss_weights 读取
    has_train_weights = hasattr(config, "training") and hasattr(
        config.training, "loss_weights"
    )
    if has_train_weights:
        lw = config.training.loss_weights
        w_rec_scale = float(getattr(lw, "reconstruction", 1.0))
        w_spec = float(getattr(lw, "spectral", 0.0))
        # 兼容 data_consistency 和 degradation_consistency
        w_dc = float(
            getattr(lw, "data_consistency", getattr(lw, "degradation_consistency", 0.0))
        )

    # 2. 如果未在 training 中定义，回退到 loss.* (兼容旧配置)
    else:
        if hasattr(config, "loss"):
            # Spectral
            if hasattr(config.loss, "spectral"):
                if hasattr(config.loss.spectral, "weight"):
                    w_spec = float(config.loss.spectral.weight)
                elif isinstance(config.loss.spectral, (int, float)):
                    w_spec = float(config.loss.spectral)

            # DC
            if hasattr(config.loss, "data_consistency"):
                if hasattr(config.loss.data_consistency, "weight"):
                    w_dc = float(config.loss.data_consistency.weight)
                elif isinstance(config.loss.data_consistency, (int, float)):
                    w_dc = float(config.loss.data_consistency)
            elif hasattr(config.loss, "degradation_consistency"):
                if hasattr(config.loss.degradation_consistency, "weight"):
                    w_dc = float(config.loss.degradation_consistency.weight)
                elif isinstance(config.loss.degradation_consistency, (int, float)):
                    w_dc = float(config.loss.degradation_consistency)

            # Reconstruction scale
            if hasattr(config.loss, "reconstruction"):
                if hasattr(config.loss.reconstruction, "weight"):
                    w_rec_scale = float(config.loss.reconstruction.weight)
                elif isinstance(config.loss.reconstruction, (int, float)):
                    w_rec_scale = float(config.loss.reconstruction)

    # DEBUG: 打印最终权重
    # print(f"[DEBUG] Weights - Rec: {w_rec_scale}, Spec: {w_spec}, DC: {w_dc}")

    # 内部重建分量权重 (保持原样，从 loss.* 读取)
    w_rel2 = 1.0
    w_mae = 0.1
    if hasattr(config, "loss"):
        w_rel2 = float(getattr(config.loss, "rel2_weight", 1.0))
        w_mae = float(getattr(config.loss, "mae_weight", 0.1))

    pred_z = torch.nan_to_num(pred_seq, nan=0.0, posinf=1e6, neginf=-1e6)
    target_z = torch.nan_to_num(gt_seq, nan=0.0, posinf=1e6, neginf=-1e6)
    losses = {}

    # 1. 重建损失（在z-score域计算）
    rel2_loss = torch.nan_to_num(
        rel_l2(pred_seq, gt_seq), nan=0.0, posinf=1e3, neginf=1e3
    )
    mae_loss = torch.nan_to_num(
        l1_mae(pred_seq, gt_seq), nan=0.0, posinf=1e3, neginf=1e3
    )
    # 时序一致性：导数一致性 + 能量演化一致性（课程调度）
    try:
        # 时间差分
        pred_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
        gt_diff = gt_seq[:, 1:] - gt_seq[:, :-1]
        diff_err = pred_diff - gt_diff
        num = torch.sqrt((diff_err**2).sum(dim=(-3, -2, -1)) + 1e-8)
        den = torch.sqrt((gt_diff**2).sum(dim=(-3, -2, -1)) + 1e-8)
        derivative_consistency = torch.nan_to_num(
            torch.mean(num / den), nan=0.0, posinf=1e3, neginf=1e3
        )
        # 能量演化
        pred_energy = (pred_seq**2).sum(dim=(-3, -2, -1))
        gt_energy = (gt_seq**2).sum(dim=(-3, -2, -1))
        pred_energy_diff = pred_energy[:, 1:] - pred_energy[:, :-1]
        gt_energy_diff = gt_energy[:, 1:] - gt_energy[:, :-1]
        energy_err = torch.abs(pred_energy_diff - gt_energy_diff)
        energy_norm = torch.abs(gt_energy_diff) + 1e-8
        energy_consistency = torch.nan_to_num(
            torch.mean(energy_err / energy_norm), nan=0.0, posinf=1e3, neginf=1e3
        )
        lw = getattr(getattr(config, "training", None), "loss_weights", None)
        if (
            lw is not None
            and hasattr(lw, "derivative_consistency")
            and hasattr(lw, "energy_consistency")
        ):
            try:
                w_deriv = float(lw.derivative_consistency)
            except Exception:
                w_deriv = 0.0
            try:
                w_energy = float(lw.energy_consistency)
            except Exception:
                w_energy = 0.0
        else:
            tf_decay = getattr(getattr(config, "training", None), "curriculum", None)
            if tf_decay is not None:
                decay_val = float(getattr(tf_decay, "teacher_forcing_decay", 0.95))
            else:
                decay_val = 0.95
            w_deriv = 0.2 * (1.0 - decay_val)
            w_energy = 0.1 * (1.0 - decay_val)
    except Exception:
        derivative_consistency = torch.tensor(0.0, device=device)
        energy_consistency = torch.tensor(0.0, device=device)
        w_deriv = 0.0
        w_energy = 0.0
    reconstruction_loss = (
        w_rel2 * rel2_loss
        + w_mae * mae_loss
        + w_deriv * derivative_consistency
        + w_energy * energy_consistency
    )
    reconstruction_loss = torch.nan_to_num(
        reconstruction_loss, nan=0.0, posinf=1e6, neginf=1e6
    )
    # Apply global reconstruction scale (from training.loss_weights.reconstruction)
    reconstruction_loss = w_rec_scale * reconstruction_loss
    losses["reconstruction_loss"] = reconstruction_loss
    losses["rel2_loss"] = rel2_loss
    losses["mae_loss"] = mae_loss
    losses["derivative_consistency"] = derivative_consistency
    losses["energy_consistency"] = energy_consistency

    # 2. 频谱损失（在原值域计算）
    if w_spec > 0:
        # 将序列转换为批次处理
        pred_flat = pred_seq.reshape(B * T, C, H, W).contiguous()
        gt_flat = gt_seq.reshape(B * T, C, H, W).contiguous()

        # 安全解析 keys：若未提供或类型异常则置为 None，启用通道级 mean/std
        data_keys = None
        try:
            if hasattr(config, "data"):
                maybe_keys = getattr(config.data, "keys", None)
                # 仅当 keys 是通道名（纯字母字符串）时接受；否则视为误配置
                if isinstance(maybe_keys, (list, tuple)) and all(
                    isinstance(k, str) and k.isalpha() for k in maybe_keys
                ):
                    data_keys = list(maybe_keys)
        except Exception:
            data_keys = None

        # 反归一化到原值域
        pred_orig = _denormalize_tensor(pred_flat, norm_stats, data_keys)
        target_orig = _denormalize_tensor(gt_flat, norm_stats, data_keys)

        # 计算频谱损失
        spectral_loss = _compute_spectral_loss(pred_orig, target_orig, config)
        spectral_loss = torch.nan_to_num(spectral_loss, nan=0.0, posinf=1e6, neginf=1e6)
        losses["spectral_loss"] = spectral_loss
    else:
        losses["spectral_loss"] = torch.tensor(0.0, device=device)

    # 3. 数据一致性损失（在原值域计算）
    if w_dc > 0:
        # 优先使用预计算的 pred_obs (由 Trainer 通过 H 算子生成)
        # obs_data 中应包含 'pred_obs' (单步) 或 'pred_obs_seq' (多步)
        pred_obs_seq = obs_data.get("pred_obs_seq")
        if pred_obs_seq is None and "pred_obs" in obs_data:
            # 兼容单步命名
            pred_obs_seq = obs_data["pred_obs"]

        if pred_obs_seq is not None:
            # 确保维度匹配
            if pred_obs_seq.dim() == 4 and T > 1:
                pred_obs_seq = pred_obs_seq.unsqueeze(1).expand(-1, T, -1, -1, -1)

            # 展平处理
            try:
                B_obs, T_obs = pred_obs_seq.shape[:2]
                pred_obs_flat = pred_obs_seq.reshape(
                    B_obs * T_obs, *pred_obs_seq.shape[2:]
                ).contiguous()
            except Exception:
                # 容错：如果无法展平，可能已经展平或维度不对，跳过DC
                pred_obs_flat = None

            if pred_obs_flat is not None:
                # 准备观测真值
                # obs_data['observation_seq'] -> flatten
                observation_seq = obs_data.get("observation_seq")
                if observation_seq is not None:
                    # 对齐时间步
                    if observation_seq.shape[1] != pred_obs_seq.shape[1]:
                        T_common = min(observation_seq.shape[1], pred_obs_seq.shape[1])
                        observation_seq = observation_seq[:, :T_common]
                        # 同时截断 pred_obs_flat (需要重新flatten)
                        pred_obs_seq_cut = pred_obs_seq[:, :T_common]
                        pred_obs_flat = pred_obs_seq_cut.reshape(
                            pred_obs_seq_cut.shape[0] * pred_obs_seq_cut.shape[1],
                            *pred_obs_seq_cut.shape[2:],
                        )

                    obs_flat = observation_seq.reshape(
                        observation_seq.shape[0] * observation_seq.shape[1],
                        *observation_seq.shape[2:],
                    )

                    # 构造临时 obs_data 供 _compute_data_consistency_loss 使用
                    obs_data_dc = {"observation": obs_flat}

                    # 计算DC损失
                    dc_loss = _compute_data_consistency_loss(pred_obs_flat, obs_data_dc)
                    dc_loss = torch.nan_to_num(dc_loss, nan=0.0, posinf=1e6, neginf=1e6)
                    losses["dc_loss"] = dc_loss
                else:
                    losses["dc_loss"] = torch.tensor(0.0, device=device)
            else:
                losses["dc_loss"] = torch.tensor(0.0, device=device)
        else:
            # 未提供 pred_obs，无法计算 DC 损失（禁止在 Loss 中调用 H）
            losses["dc_loss"] = torch.tensor(0.0, device=device)
    else:
        losses["dc_loss"] = torch.tensor(0.0, device=device)

    # 4. 总损失
    total_loss = (
        losses["reconstruction_loss"]
        + w_spec * losses["spectral_loss"]
        + w_dc * losses["dc_loss"]
    )
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=1e6)
    losses["total_loss"] = total_loss

    return losses
