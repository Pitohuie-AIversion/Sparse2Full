"""
AR训练指标计算

提供AR模型的评估指标计算功能
"""

import torch
import torch.nn as nn


class ARMetrics:
    """AR模型指标计算器"""

    def __init__(self):
        pass

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """
        计算所有指标

        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W]

        Returns:
            指标字典
        """
        metrics = {}

        # MSE
        mse = nn.MSELoss()(pred, target)
        metrics["mse"] = mse.item()

        # MAE
        mae = nn.L1Loss()(pred, target)
        metrics["mae"] = mae.item()

        # RMSE
        rmse = torch.sqrt(mse)
        metrics["rmse"] = rmse.item()

        # 相对L2误差
        target_norm = torch.norm(target.view(target.size(0), -1), p=2, dim=1)
        diff_norm = torch.norm((pred - target).view(pred.size(0), -1), p=2, dim=1)
        rel_l2 = torch.mean(diff_norm / (target_norm + 1e-8))
        metrics["rel_l2"] = rel_l2.item()

        # PSNR
        max_val = torch.max(target)
        min_val = torch.min(target)
        dynamic_range = max_val - min_val
        psnr = 20 * torch.log10(dynamic_range / (torch.sqrt(mse) + 1e-8))
        metrics["psnr"] = psnr.item()

        return metrics

    def compute_single_metric(
        self, pred: torch.Tensor, target: torch.Tensor, metric_name: str
    ) -> float:
        """
        计算单个指标

        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W]
            metric_name: 指标名称

        Returns:
            指标值
        """
        if metric_name == "mse":
            return nn.MSELoss()(pred, target).item()
        elif metric_name == "mae":
            return nn.L1Loss()(pred, target).item()
        elif metric_name == "rmse":
            mse = nn.MSELoss()(pred, target)
            return torch.sqrt(mse).item()
        elif metric_name == "rel_l2":
            target_norm = torch.norm(target.view(target.size(0), -1), p=2, dim=1)
            diff_norm = torch.norm((pred - target).view(pred.size(0), -1), p=2, dim=1)
            return torch.mean(diff_norm / (target_norm + 1e-8)).item()
        elif metric_name == "psnr":
            mse = nn.MSELoss()(pred, target)
            max_val = torch.max(target)
            min_val = torch.min(target)
            dynamic_range = max_val - min_val
            psnr = 20 * torch.log10(dynamic_range / (torch.sqrt(mse) + 1e-8))
            return psnr.item()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
