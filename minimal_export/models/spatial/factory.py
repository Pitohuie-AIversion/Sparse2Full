"""
空间预测模型工厂函数
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..registry import create_model as registry_create_model

if TYPE_CHECKING:
    from ..base import BaseModel


def create_model(model_name: str, **kwargs) -> BaseModel:
    """
    创建空间预测模型实例

    Args:
        model_name: 模型名称
        **kwargs: 模型参数

    Returns:
        BaseModel: 模型实例
    """
    if model_name is None:
        raise ValueError("model_name is None")
    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is empty")

    try:
        return registry_create_model(name, **kwargs)
    except Exception as e:
        # 不假设 registry 的异常类型，统一包装，便于日志定位
        raise RuntimeError(
            f"Failed to create model '{name}'. "
            f"Check registry name/alias and kwargs. Original error: {type(e).__name__}: {e}"
        ) from e
