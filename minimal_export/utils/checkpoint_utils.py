"""
检查点工具

提供模型检查点的保存和加载功能
"""

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    checkpoint_path: str,
    additional_info: dict[str, Any] | None = None,
) -> None:
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        global_step: 全局步数
        best_val_loss: 最佳验证损失
        checkpoint_path: 检查点路径
        additional_info: 额外信息
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    # 确保目录存在
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        strict: 是否严格匹配模型参数

    Returns:
        检查点信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)

        # 加载模型状态
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info("Model state dict loaded")
        else:
            logger.warning("No model state dict found in checkpoint")

        # 加载优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state dict loaded")
        elif optimizer is not None:
            logger.warning("No optimizer state dict found in checkpoint")

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state dict loaded")
        elif scheduler is not None:
            logger.warning("No scheduler state dict found in checkpoint")

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def find_latest_checkpoint(
    checkpoint_dir: str, pattern: str = "checkpoint_*.pth"
) -> str | None:
    """
    查找最新的检查点文件

    Args:
        checkpoint_dir: 检查点目录
        pattern: 文件名模式

    Returns:
        最新检查点路径，如果没有找到则返回None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob(pattern))
    if not checkpoint_files:
        return None

    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    latest_checkpoint = str(checkpoint_files[0])
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")

    return latest_checkpoint
