"""AR (Auto-Regressive) 模块

提供自回归时序预测功能，包装现有的单帧模型为多步预测模型。
支持teacher forcing训练和roll-out推理。
"""

from .wrapper import ARWrapper

__all__ = ['ARWrapper']