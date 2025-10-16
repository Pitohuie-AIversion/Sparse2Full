"""损失函数工具模块

提供损失函数的便捷导入和计算功能
"""

from ops.loss import (
    TotalLoss,
    ReconstructionLoss,
    SpectralLoss,
    DataConsistencyLoss,
    compute_total_loss,
    compute_gradient_loss,
    compute_pde_residual_loss
)

__all__ = [
    'TotalLoss',
    'ReconstructionLoss', 
    'SpectralLoss',
    'DataConsistencyLoss',
    'compute_total_loss',
    'compute_gradient_loss',
    'compute_pde_residual_loss'
]