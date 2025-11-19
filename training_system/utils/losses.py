"""
损失函数模块

提供训练所需的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np


class CombinedLoss(nn.Module):
    """组合损失函数
    
    包含重建损失、频域损失和数据一致性损失
    """
    
    def __init__(self, 
                 rec_weight: float = 1.0,
                 spec_weight: float = 0.5, 
                 dc_weight: float = 1.0,
                 rec_loss_type: str = 'mse',
                 spec_loss_type: str = 'mse',
                 dc_loss_type: str = 'mse',
                 low_freq_modes: int = 16,
                 observation_config: Optional[Dict[str, Any]] = None,
                 normalization_stats: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.rec_weight = rec_weight
        self.spec_weight = spec_weight
        self.dc_weight = dc_weight
        
        self.rec_loss_type = rec_loss_type
        self.spec_loss_type = spec_loss_type
        self.dc_loss_type = dc_loss_type
        
        self.low_freq_modes = low_freq_modes
        self.observation_config = observation_config or {}
        self.normalization_stats = normalization_stats
        
        # 初始化损失函数
        self._init_loss_functions()
    
    def _init_loss_functions(self):
        """初始化各个损失函数"""
        # 重建损失
        if self.rec_loss_type == 'mse':
            self.rec_loss_fn = nn.MSELoss(reduction='mean')
        elif self.rec_loss_type == 'l1':
            self.rec_loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"不支持的重建损失类型: {self.rec_loss_type}")
        
        # 频域损失
        if self.spec_loss_type == 'mse':
            self.spec_loss_fn = nn.MSELoss(reduction='mean')
        elif self.spec_loss_type == 'l1':
            self.spec_loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"不支持的频域损失类型: {self.spec_loss_type}")
        
        # 数据一致性损失
        if self.dc_loss_type == 'mse':
            self.dc_loss_fn = nn.MSELoss(reduction='mean')
        elif self.dc_loss_type == 'l1':
            self.dc_loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"不支持的数据一致性损失类型: {self.dc_loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                observation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W] 
            observation: 观测值 [B, C, H', W'] (可选)
            
        Returns:
            损失字典，包含各个损失分量
        """
        losses = {}
        total_loss = 0.0
        
        # 重建损失
        if self.rec_weight > 0:
            rec_loss = self.rec_loss_fn(pred, target)
            losses['rec_loss'] = rec_loss
            total_loss += self.rec_weight * rec_loss
        
        # 频域损失
        if self.spec_weight > 0:
            spec_loss = self._compute_spectral_loss(pred, target)
            losses['spec_loss'] = spec_loss
            total_loss += self.spec_weight * spec_loss
        
        # 数据一致性损失
        if self.dc_weight > 0 and observation is not None:
            dc_loss = self._compute_data_consistency_loss(pred, observation)
            losses['dc_loss'] = dc_loss
            total_loss += self.dc_weight * dc_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频域损失
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W]
            
        Returns:
            频域损失
        """
        # 转换为频域
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # 只考虑低频分量
        H, W = pred.shape[-2:]
        center_h, center_w = H // 2, W // 2
        
        # 提取低频区域
        low_freq_h = min(self.low_freq_modes, center_h)
        low_freq_w = min(self.low_freq_modes, center_w)
        
        pred_low_freq = pred_fft[..., 
            center_h-low_freq_h:center_h+low_freq_h,
            center_w-low_freq_w:center_w+low_freq_w]
        target_low_freq = target_fft[...,
            center_h-low_freq_h:center_h+low_freq_h,
            center_w-low_freq_w:center_w+low_freq_w]
        
        # 计算频域损失
        spec_loss = self.spec_loss_fn(pred_low_freq, target_low_freq)
        return spec_loss
    
    def _compute_data_consistency_loss(self, pred: torch.Tensor, 
                                     observation: torch.Tensor) -> torch.Tensor:
        """计算数据一致性损失
        
        Args:
            pred: 预测值 [B, C, H, W]
            observation: 观测值 [B, C, H', W']
            
        Returns:
            数据一致性损失
        """
        # 这里需要根据观测算子来实现
        # 暂时使用简单的下采样作为示例
        B, C, H, W = pred.shape
        obs_H, obs_W = observation.shape[-2:]
        
        if H != obs_H or W != obs_W:
            # 对预测进行下采样以匹配观测尺寸
            pred_downsampled = F.interpolate(pred, size=(obs_H, obs_W), 
                                           mode='bilinear', align_corners=False)
        else:
            pred_downsampled = pred
        
        dc_loss = self.dc_loss_fn(pred_downsampled, observation)
        return dc_loss


class SimpleLoss(nn.Module):
    """简单的损失函数包装器"""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)