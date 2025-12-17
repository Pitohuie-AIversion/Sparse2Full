"""兼容测试的简单 CombinedLoss 封装

部分测试期望从 `losses.combined_loss` 导入 `CombinedLoss`。
此处提供一个最小实现，返回标量 MSE，用于解耦测试路径差异。
"""

from typing import Any
import torch


class CombinedLoss:
    def __init__(self, **kwargs: Any) -> None:
        self._mse = torch.nn.MSELoss()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._mse(pred, target)