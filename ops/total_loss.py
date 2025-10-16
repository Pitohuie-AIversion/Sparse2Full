"""总损失函数类

实现三件套损失：L = L_rec + λ_s L_spec + λ_dc L_dc
严格按照开发手册要求，确保值域正确处理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

from .losses import compute_total_loss


class TotalLoss(nn.Module):
    """总损失函数类
    
    包含重建损失、频谱损失和数据一致性损失的组合
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        obs_data: Dict, 
        norm_stats: Optional[Dict[str, torch.Tensor]], 
        cfg: DictConfig
    ) -> Dict[str, torch.Tensor]:
        """前向传播计算损失
        
        Args:
            pred: 模型预测（z-score域）[B, C, H, W]
            target: 真值标签（z-score域）[B, C, H, W]
            obs_data: 观测数据字典
            norm_stats: 归一化统计量
            cfg: 配置
            
        Returns:
            损失字典
        """
        return compute_total_loss(pred, target, obs_data, norm_stats, cfg)