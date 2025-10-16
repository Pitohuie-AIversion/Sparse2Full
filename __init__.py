"""PDEBench稀疏观测重建系统

基于深度学习的偏微分方程稀疏观测重建系统，支持超分辨率(SR)和裁剪(Crop)任务。
严格按照开发手册的黄金法则实现，确保观测算子H与训练DC复用同一实现。
"""

__version__ = "1.0.0"
__author__ = "PDEBench Team"
__description__ = "Deep Learning System for PDE Sparse Observation Reconstruction"

# 导入核心模块
from . import models
from . import ops
from . import datasets
from . import utils

# 导入主要功能
from .ops import apply_degradation_operator, compute_total_loss, compute_all_metrics
from .models import SwinUNet, HybridModel, MLPModel

__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__description__",
    
    # 核心模块
    "models",
    "ops", 
    "datasets",
    "utils",
    
    # 主要功能
    "apply_degradation_operator",
    "compute_total_loss", 
    "compute_all_metrics",
    "SwinUNet",
    "HybridModel", 
    "MLPModel",
]