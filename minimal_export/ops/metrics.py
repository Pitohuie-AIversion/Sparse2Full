"""
PDEBench稀疏观测重建系统 - 指标计算模块

实现各种评估指标的计算，包括：
- 相对L2误差 (Rel-L2)
- 平均绝对误差 (MAE)
- 峰值信噪比 (PSNR)
- 结构相似性指数 (SSIM)
- 频域误差 (fRMSE)
- 边界误差 (bRMSE)
- 数据一致性误差 (DC Error)

遵循技术架构文档7.6节评测标准。
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def compute_rel_l2_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """计算相对L2误差

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        eps: 数值稳定性参数

    Returns:
        相对L2误差 [B, C] 或标量
    """
    # 计算每个样本每个通道的相对L2误差
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
    target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]

    # L2范数
    diff_norm = torch.norm(pred_flat - target_flat, dim=2)  # [B, C]
    target_norm = torch.norm(target_flat, dim=2)  # [B, C]

    # 相对误差
    rel_error = diff_norm / (target_norm + eps)

    return rel_error


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算平均绝对误差

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]

    Returns:
        MAE [B, C] 或标量
    """
    # 计算每个样本每个通道的MAE
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
    target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]

    mae = torch.mean(torch.abs(pred_flat - target_flat), dim=2)  # [B, C]

    return mae


def compute_psnr_batch(
    pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None
) -> torch.Tensor:
    """批量计算PSNR

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        data_range: 数据范围，如果为None则自动计算

    Returns:
        PSNR值 [B, C]
    """
    if data_range is None:
        # 使用目标的动态范围，但避免在z-score域误判，最小范围设下界
        dr = (target.max() - target.min()).detach()
        data_range = float(torch.clamp(dr, min=1e-3).item())

    mse = F.mse_loss(pred, target, reduction="none")  # [B, C, H, W]
    mse = mse.view(mse.size(0), mse.size(1), -1).mean(dim=2)  # [B, C]

    # 避免除零
    mse = torch.clamp(mse, min=1e-10)

    psnr_val = 20 * torch.log10(data_range / torch.sqrt(mse))

    return psnr_val


def compute_ssim_batch(
    pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None
) -> torch.Tensor:
    """批量计算SSIM

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        data_range: 数据范围

    Returns:
        SSIM值 [B, C]
    """
    if data_range is None:
        data_range = target.max() - target.min()

    batch_size, channels = pred.size(0), pred.size(1)
    ssim_values = torch.zeros(batch_size, channels)

    # 转换为numpy进行计算
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    for b in range(batch_size):
        for c in range(channels):
            ssim_val = ssim(
                target_np[b, c],
                pred_np[b, c],
                data_range=float(data_range),
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
            ssim_values[b, c] = ssim_val

    return ssim_values


def compute_frequency_error(
    pred: torch.Tensor, target: torch.Tensor, freq_bands: list[tuple[int, int]] = None
) -> dict[str, torch.Tensor]:
    """计算频域误差 (fRMSE)

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        freq_bands: 频率带划分 [(low_start, low_end), (mid_start, mid_end), (high_start, high_end)]

    Returns:
        各频段的RMSE误差字典
    """
    if freq_bands is None:
        # 默认频率带划分
        H, W = pred.size(2), pred.size(3)
        max_freq = min(H, W) // 2
        freq_bands = [
            (0, max_freq // 4),  # 低频
            (max_freq // 4, max_freq // 2),  # 中频
            (max_freq // 2, max_freq),  # 高频
        ]

    # FFT变换
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))

    # 计算频率坐标
    H, W = pred.size(2), pred.size(3)
    freq_y = torch.fft.fftfreq(H, d=1.0).abs()
    freq_x = torch.fft.rfftfreq(W, d=1.0).abs()
    freq_grid = torch.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)

    results = {}

    for band_name, (low_freq, high_freq) in zip(["low", "mid", "high"], freq_bands):
        # 创建频率掩码
        mask = (freq_grid >= low_freq) & (freq_grid < high_freq)
        mask = mask.to(pred.device)

        # 应用掩码
        pred_band = pred_fft * mask[None, None, :, :]
        target_band = target_fft * mask[None, None, :, :]

        # 计算RMSE
        diff = pred_band - target_band
        mse = torch.mean(torch.abs(diff) ** 2, dim=(-2, -1))  # [B, C]
        rmse = torch.sqrt(mse)

        results[f"frmse_{band_name}"] = rmse

    return results


def compute_boundary_error(
    pred: torch.Tensor, target: torch.Tensor, boundary_width: int = 16
) -> torch.Tensor:
    """计算边界误差 (bRMSE)

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        boundary_width: 边界宽度（像素）

    Returns:
        边界RMSE [B, C]
    """
    H, W = pred.size(2), pred.size(3)

    # 创建边界掩码
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[:boundary_width, :] = True  # 上边界
    mask[-boundary_width:, :] = True  # 下边界
    mask[:, :boundary_width] = True  # 左边界
    mask[:, -boundary_width:] = True  # 右边界

    mask = mask.to(pred.device)

    # 提取边界区域
    pred_boundary = pred * mask[None, None, :, :]
    target_boundary = target * mask[None, None, :, :]

    # 计算RMSE
    diff = pred_boundary - target_boundary
    mse = torch.sum(diff**2, dim=(-2, -1)) / torch.sum(mask)  # [B, C]
    rmse = torch.sqrt(mse)

    return rmse


def compute_data_consistency_error(
    pred: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """计算数据一致性误差

    Args:
        pred: 预测值 [B, C, H, W]
        observed: 观测值 [B, C, H, W]
        mask: 观测掩码 [B, 1, H, W] 或 [B, C, H, W]

    Returns:
        数据一致性误差 [B, C]
    """
    # 确保掩码维度正确
    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.expand_as(pred)

    # 在观测位置计算误差
    diff = (pred - observed) * mask

    # 计算RMSE
    mse = torch.sum(diff**2, dim=(-2, -1)) / (torch.sum(mask, dim=(-2, -1)) + 1e-8)
    rmse = torch.sqrt(mse)

    return rmse


def compute_ssim_gpu(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """GPU优化的SSIM计算

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        data_range: 数据范围，如果为None则自动计算
        window_size: 高斯窗口大小
        sigma: 高斯窗口标准差

    Returns:
        SSIM值 [B, C]
    """
    if data_range is None:
        data_range = target.max() - target.min()

    # 创建高斯窗口
    gauss = torch.FloatTensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    gauss = gauss / gauss.sum()

    # 2D高斯核
    window_1d = gauss.unsqueeze(0)
    window_2d = window_1d.t() * window_1d
    window = window_2d.unsqueeze(0).unsqueeze(0)

    # 移动到GPU
    if pred.is_cuda:
        window = window.cuda(pred.device)

    # 计算均值
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = (
        F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.size(1))
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            target * target, window, padding=window_size // 2, groups=target.size(1)
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.size(1))
        - mu1_mu2
    )

    # SSIM常数
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    # 平均SSIM值
    ssim_values = ssim_map.mean(dim=(-2, -1))  # [B, C]

    return ssim_values


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    data_range: float | None = None,
    use_gpu_ssim: bool = True,
) -> dict[str, float]:
    """计算所有评估指标

    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        observed: 观测值 [B, C, H, W]，用于计算数据一致性误差
        mask: 观测掩码 [B, 1, H, W] 或 [B, C, H, W]
        data_range: 数据范围
        use_gpu_ssim: 是否使用GPU优化的SSIM计算

    Returns:
        包含所有指标的字典（标量值）
    """
    metrics = {}

    # 基础指标
    rel_l2 = compute_rel_l2_error(pred, target)
    metrics["rel_l2"] = float(rel_l2.mean().item())

    mae = compute_mae(pred, target)
    metrics["mae"] = float(mae.mean().item())

    psnr_val = compute_psnr_batch(pred, target, data_range)
    metrics["psnr"] = float(psnr_val.mean().item())

    # GPU优化的SSIM计算
    if use_gpu_ssim and pred.is_cuda:
        ssim_val = compute_ssim_gpu(pred, target, data_range)
    else:
        ssim_val = compute_ssim_batch(pred, target, data_range)
    metrics["ssim"] = float(ssim_val.mean().item())

    # 频域指标
    freq_metrics = compute_frequency_error(pred, target)
    for key, value in freq_metrics.items():
        metrics[key] = float(value.mean().item())

    # 边界指标
    brmse = compute_boundary_error(pred, target)
    metrics["brmse"] = float(brmse.mean().item())

    # 数据一致性指标
    if observed is not None and mask is not None:
        dc_error = compute_data_consistency_error(pred, observed, mask)
        metrics["dc_error"] = float(dc_error.mean().item())

    return metrics


def aggregate_metrics(
    metrics_list: list[dict[str, torch.Tensor]], reduction: str = "mean"
) -> dict[str, float]:
    """聚合多个样本的指标

    Args:
        metrics_list: 指标字典列表
        reduction: 聚合方式 ('mean', 'std', 'median')

    Returns:
        聚合后的指标字典
    """
    if not metrics_list:
        return {}

    # 获取所有指标名称
    metric_names = set()
    for metrics in metrics_list:
        metric_names.update(metrics.keys())

    aggregated = {}

    for name in metric_names:
        # 收集所有样本的该指标值
        values = []
        for metrics in metrics_list:
            if name in metrics:
                metric_val = metrics[name]
                if isinstance(metric_val, torch.Tensor):
                    values.append(metric_val.detach().cpu().numpy())
                else:
                    values.append(metric_val)

        if values:
            values = np.concatenate([np.atleast_1d(v) for v in values])

            if reduction == "mean":
                aggregated[name] = float(np.mean(values))
            elif reduction == "std":
                aggregated[name] = float(np.std(values))
            elif reduction == "median":
                aggregated[name] = float(np.median(values))
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

    return aggregated


def compute_statistical_significance(
    metrics1: dict[str, list[float]],
    metrics2: dict[str, list[float]],
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """计算统计显著性

    Args:
        metrics1: 方法1的指标值
        metrics2: 方法2的指标值
        alpha: 显著性水平

    Returns:
        包含p值和Cohen's d的字典
    """
    from scipy import stats

    results = {}

    for metric_name in metrics1.keys():
        if metric_name in metrics2:
            values1 = np.array(metrics1[metric_name])
            values2 = np.array(metrics2[metric_name])

            # t检验
            t_stat, p_value = stats.ttest_rel(values1, values2)

            # Cohen's d
            pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std

            results[metric_name] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "significant": p_value < alpha,
            }

    return results
