"""
AR训练损失函数

提供AR模型的损失计算功能
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def compute_ar_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model: Optional[nn.Module] = None,
    loss_config: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    计算AR训练的总损失
    
    Args:
        pred: 预测值 [B, C, H, W]
        target: 目标值 [B, C, H, W]
        model: 模型（可选）
        loss_config: 损失函数配置
        
    Returns:
        总损失值
    """
    if loss_config is None:
        loss_config = {}
    
    # 默认配置
    loss_type = loss_config.get("type", "mse")
    reduction = loss_config.get("reduction", "mean")
    
    if loss_type == "mse":
        criterion = nn.MSELoss(reduction=reduction)
    elif loss_type == "l1":
        criterion = nn.L1Loss(reduction=reduction)
    elif loss_type == "smooth_l1":
        criterion = nn.SmoothL1Loss(reduction=reduction)
    else:
        # 默认使用MSE
        criterion = nn.MSELoss(reduction=reduction)
    
    # 计算损失
    loss = criterion(pred, target)
    
    # 添加L2正则化（如果配置）
    if model is not None and loss_config.get("l2_weight", 0.0) > 0.0:
        l2_weight = loss_config["l2_weight"]
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_weight * l2_reg
    
    return loss