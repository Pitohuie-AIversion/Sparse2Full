"""核心操作模块

包含损失函数、指标计算、H算子等核心操作。
"""

from .degradation import apply_degradation_operator
from .losses import compute_total_loss
from .metrics import compute_all_metrics

__all__ = [
    "apply_degradation_operator",
    "compute_total_loss", 
    "compute_all_metrics",
]