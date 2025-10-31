"""PDEBench稀疏观测重建系统。

提供模型、算子、数据集与工具等子模块。为降低测试环境依赖，在导入
可选依赖失败时会提供清晰的报错提示。
"""

from typing import Callable
from types import ModuleType

__version__ = "1.0.0"
__author__ = "PDEBench Team"
__description__ = "Deep Learning System for PDE Sparse Observation Reconstruction"

# 导入核心模块
from . import models

try:  # pragma: no cover - 可选依赖
    from . import utils
except Exception:  # pragma: no cover - 缺失可选依赖
    utils = None  # type: ignore[assignment]

try:  # pragma: no cover - 可选依赖
    from . import ops
except Exception as err:  # pragma: no cover - 缺失可选依赖
    def _raise_missing_ops(name: str) -> None:
        raise ImportError(
            f"`{name}` requires optional dependencies for the ops module. "
            "Install `opencv-python-headless` or the full `opencv-python` package."
        ) from err

    def _missing_ops_func(name: str) -> Callable:
        def _wrapper(*_, **__):  # pragma: no cover - 简单报错逻辑
            _raise_missing_ops(name)

        return _wrapper

    ops = ModuleType("Sparse2Full.ops")
    ops.__doc__ = (
        "Proxy module that raises informative ImportError messages when optional "
        "dependencies (e.g. OpenCV) for `Sparse2Full.ops` are not installed."
    )

    def _ops_getattr(name: str):  # pragma: no cover - 属性访问时触发
        _raise_missing_ops(name)

    setattr(ops, "__getattr__", _ops_getattr)

    apply_degradation_operator = _missing_ops_func("apply_degradation_operator")
    compute_total_loss = _missing_ops_func("compute_total_loss")
    compute_all_metrics = _missing_ops_func("compute_all_metrics")

    setattr(ops, "apply_degradation_operator", apply_degradation_operator)
    setattr(ops, "compute_total_loss", compute_total_loss)
    setattr(ops, "compute_all_metrics", compute_all_metrics)
else:
    from .ops import apply_degradation_operator, compute_total_loss, compute_all_metrics

try:  # pragma: no cover - 可选依赖
    from . import datasets
except Exception:  # pragma: no cover - 缺失数据相关依赖
    datasets = None  # type: ignore[assignment]

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