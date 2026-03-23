"""测试兼容的训练工具导出

tests/test_e2e_training.py 期望从 `tools.train` 导入：
- create_model(config)
- create_loss_function(config)
- create_optimizer(model, optimizer_config)

本模块提供最小实现，复用项目现有模型与损失封装，满足统一接口：
forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

from typing import Dict, Any
import torch
import torch.nn as nn

from models import create_model as _create_model

try:
    from losses.combined_loss import CombinedLoss
except Exception:
    class CombinedLoss:
        def __init__(self, **kwargs: Any) -> None:
            self._mse = nn.MSELoss()
        def __call__(self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self._mse(pred, target)


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """根据测试配置创建模型实例。

    兼容配置结构：
    {
        'name': 'SwinUNet',
        'in_channels': 2,
        'out_channels': 2,
        'img_size': [64, 64],
        ...其它模型参数...
    }
    """
    name = model_config.get('name', 'SwinUNet')
    params = dict(model_config)
    params.pop('name', None)
    # 统一 img_size 为 int 或 tuple
    img_size = params.get('img_size', 64)
    if isinstance(img_size, (list, tuple)):
        # 项目模型接受单值或方形；取较大边或第一个
        params['img_size'] = img_size[0]
    return _create_model(name, **params)


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """创建损失函数。

    测试使用 CombinedLoss，内部默认MSE，支持权重字段但忽略以保持简单。
    """
    return CombinedLoss(**loss_config)


def create_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """创建优化器（AdamW）。"""
    name = optimizer_config.get('name', 'AdamW').lower()
    lr = float(optimizer_config.get('lr', 1e-3))
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))
    betas = optimizer_config.get('betas', (0.9, 0.999))
    if isinstance(betas, (list, tuple)):
        betas = (float(betas[0]), float(betas[1]))
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        # 回退到AdamW
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)