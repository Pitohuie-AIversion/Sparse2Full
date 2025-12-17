"""
增强损失函数模块 - 解决loss停滞问题
提供多种正则化损失和自适应权重调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from omegaconf import DictConfig
import numpy as np

from .degradation import apply_degradation_operator


class EnhancedReconstructionLoss(nn.Module):
    """增强重建损失 - 包含多种损失函数和正则化"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.get('loss_type', 'mse')
        self.reduction = config.get('reduction', 'mean')
        self.huber_delta = config.get('huber_delta', 1.0)
        self.smooth_l1_beta = config.get('smooth_l1_beta', 1.0)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算增强重建损失"""
        
        # 基础重建损失
        if self.loss_type == 'mse':
            base_loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'l1':
            base_loss = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            base_loss = F.huber_loss(pred, target, delta=self.huber_delta, reduction='none')
        elif self.loss_type == 'smooth_l1':
            base_loss = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # 应用掩码
        if mask is not None:
            base_loss = base_loss * mask
            if self.reduction == 'mean':
                base_loss = base_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                base_loss = base_loss.sum()
        else:
            if self.reduction == 'mean':
                base_loss = base_loss.mean()
            elif self.reduction == 'sum':
                base_loss = base_loss.sum()
        
        # 计算梯度损失（边缘保持）
        grad_loss = self._compute_gradient_loss(pred, target, mask)
        
        # 计算频谱损失（低频保持）
        spectral_loss = self._compute_low_freq_loss(pred, target)
        
        return {
            'base_loss': base_loss,
            'gradient_loss': grad_loss,
            'spectral_loss': spectral_loss,
            'total_loss': base_loss + 0.1 * grad_loss + 0.05 * spectral_loss
        }
    
    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算梯度损失 - 保持边缘信息"""
        batch_size, channels, height, width = pred.shape
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 扩展sobel算子以匹配输入通道数
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        # 对每个通道计算梯度（分组卷积）
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
        
        # 计算梯度幅度
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # 计算梯度损失
        grad_loss = F.mse_loss(pred_grad_mag, target_grad_mag, reduction=self.reduction)
        
        return grad_loss
    
    def _compute_low_freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算低频损失 - 保持整体结构"""
        # 使用平均池化提取低频信息
        kernel_size = 3
        pred_low = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        low_freq_loss = F.mse_loss(pred_low, target_low, reduction=self.reduction)
        
        return low_freq_loss


class AdaptiveSpectralLoss(nn.Module):
    """自适应频谱损失 - 动态调整频率权重"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.k_max = config.get('k_max', 16)
        self.adaptive_weight = config.get('adaptive_weight', True)
        self.frequency_weights = None
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                epoch: Optional[int] = None) -> torch.Tensor:
        """计算自适应频谱损失"""
        
        # 计算FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 获取频率幅度
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 自适应频率权重
        if self.adaptive_weight and epoch is not None:
            weights = self._compute_adaptive_weights(target_mag, epoch)
        else:
            weights = self._compute_uniform_weights()
        
        # 只比较低频部分
        pred_mag_low = pred_mag[..., :self.k_max, :self.k_max]
        target_mag_low = target_mag[..., :self.k_max, :self.k_max]
        weights_low = weights[..., :self.k_max, :self.k_max]
        
        # 加权频谱损失
        spectral_loss = F.mse_loss(pred_mag_low * weights_low, target_mag_low * weights_low)
        
        # 相位损失（可选）
        phase_loss = self._compute_phase_loss(pred_fft, target_fft)
        
        return spectral_loss + 0.1 * phase_loss
    
    def _compute_adaptive_weights(self, target_mag: torch.Tensor, epoch: int) -> torch.Tensor:
        """计算自适应频率权重"""
        # 基于训练进度调整频率重要性
        # 早期关注低频，后期关注高频
        progress = min(epoch / 200.0, 1.0)  # 假设200个epoch
        
        # 创建频率坐标
        B, C, H, W = target_mag.shape
        freq_y = torch.arange(H).float() / H
        freq_x = torch.arange(W).float() / W
        freq_yy, freq_xx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # 计算频率距离
        freq_dist = torch.sqrt(freq_yy**2 + freq_xx**2).to(target_mag.device)
        
        # 自适应权重：早期低频重要，后期高频重要
        low_freq_weight = 1.0 - 0.5 * progress
        high_freq_weight = 0.5 + 0.5 * progress
        
        # 指数衰减权重
        weights = low_freq_weight * torch.exp(-freq_dist * 5) + high_freq_weight * torch.exp(-(1 - freq_dist) * 5)
        
        return weights.unsqueeze(0).unsqueeze(0)
    
    def _compute_uniform_weights(self) -> torch.Tensor:
        """计算均匀权重"""
        return torch.ones(1, 1, self.k_max, self.k_max)
    
    def _compute_phase_loss(self, pred_fft: torch.Tensor, target_fft: torch.Tensor) -> torch.Tensor:
        """计算相位损失"""
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # 只比较低频相位
        pred_phase_low = pred_phase[..., :self.k_max, :self.k_max]
        target_phase_low = target_phase[..., :self.k_max, :self.k_max]
        
        # 相位周期性损失
        phase_diff = torch.abs(torch.sin(pred_phase_low - target_phase_low))
        phase_loss = phase_diff.mean()
        
        return phase_loss


class EnhancedDCLoss(nn.Module):
    """增强数据一致性损失 - 多尺度一致性检查"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.multi_scale = config.get('multi_scale', True)
        self.scale_factors = config.get('scale_factors', [1.0, 0.5, 0.25])
        
    def forward(self, pred_orig: torch.Tensor, obs_data: torch.Tensor, 
                h_params: Dict, epoch: Optional[int] = None) -> torch.Tensor:
        """计算增强数据一致性损失"""
        
        total_dc_loss = 0.0
        
        if self.multi_scale:
            # 多尺度一致性检查
            for scale_factor in self.scale_factors:
                if scale_factor == 1.0:
                    pred_scaled = pred_orig
                    obs_scaled = obs_data
                else:
                    # 下采样预测和观测
                    size = int(pred_orig.shape[-2] * scale_factor)
                    pred_scaled = F.interpolate(pred_orig, size=(size, size), mode='bilinear', align_corners=False)
                    obs_scaled = F.interpolate(obs_data, size=(size, size), mode='bilinear', align_corners=False)
                
                # 应用观测算子
                pred_obs = apply_degradation_operator(pred_scaled, h_params)
                
                # 计算尺度特定的DC损失
                scale_dc_loss = F.mse_loss(pred_obs, obs_scaled)
                total_dc_loss += scale_factor * scale_dc_loss  # 较大尺度权重更高
        else:
            # 单尺度一致性检查
            pred_obs = apply_degradation_operator(pred_orig, h_params)
            total_dc_loss = F.mse_loss(pred_obs, obs_data)
        
        # 添加边缘一致性损失
        edge_dc_loss = self._compute_edge_consistency(pred_orig, obs_data, h_params)
        
        return total_dc_loss + 0.1 * edge_dc_loss
    
    def _compute_edge_consistency(self, pred_orig: torch.Tensor, obs_data: torch.Tensor, 
                                 h_params: Dict) -> torch.Tensor:
        """计算边缘一致性损失"""
        batch_size, channels, height, width = pred_orig.shape
        
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred_orig.dtype, device=pred_orig.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred_orig.dtype, device=pred_orig.device).view(1, 1, 3, 3)
        
        # 扩展sobel算子以匹配输入通道数
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        # 计算预测边缘
        pred_edge_x = F.conv2d(pred_orig, sobel_x, padding=1, groups=channels)
        pred_edge_y = F.conv2d(pred_orig, sobel_y, padding=1, groups=channels)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        
        # 应用观测算子到边缘
        pred_edge_obs = apply_degradation_operator(pred_edge, h_params)
        
        # 计算观测边缘
        obs_edge_x = F.conv2d(obs_data, sobel_x, padding=1, groups=channels)
        obs_edge_y = F.conv2d(obs_data, sobel_y, padding=1, groups=channels)
        obs_edge = torch.sqrt(obs_edge_x**2 + obs_edge_y**2 + 1e-8)
        
        # 边缘一致性损失
        edge_dc_loss = F.mse_loss(pred_edge_obs, obs_edge)
        
        return edge_dc_loss


class AdaptiveLossWeights(nn.Module):
    """自适应损失权重 - 动态调整各损失分量的权重"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.adaptive_weights = config.get('adaptive_weights', True)
        self.weight_adjustment_factor = config.get('weight_adjustment_factor', 0.1)
        
        # 初始权重
        self.register_buffer('rec_weight', torch.tensor(config.get('rec_weight', 1.0)))
        self.register_buffer('spec_weight', torch.tensor(config.get('spec_weight', 0.5)))
        self.register_buffer('dc_weight', torch.tensor(config.get('dc_weight', 0.5)))
        self.register_buffer('grad_weight', torch.tensor(config.get('grad_weight', 0.1)))
        
        # 历史损失记录
        self.loss_history = []
        
    def forward(self, loss_dict: Dict[str, torch.Tensor], 
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """应用自适应权重"""
        
        if not self.adaptive_weights or epoch is None:
            # 使用固定权重
            return {
                'total_loss': (self.rec_weight * loss_dict['reconstruction'] +
                             self.spec_weight * loss_dict['spectral'] +
                             self.dc_weight * loss_dict['degradation_consistency'] +
                             self.grad_weight * loss_dict['gradient']),
                'weights': {
                    'reconstruction': self.rec_weight.item(),
                    'spectral': self.spec_weight.item(),
                    'degradation_consistency': self.dc_weight.item(),
                    'gradient': self.grad_weight.item()
                }
            }
        
        # 自适应权重调整
        self.loss_history.append({k: v.item() for k, v in loss_dict.items()})
        
        # 基于训练进度调整权重
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            
            # 计算各损失的趋势
            loss_trends = {}
            for key in ['reconstruction', 'spectral', 'degradation_consistency', 'gradient']:
                if key in recent_losses[0]:
                    values = [loss[key] for loss in recent_losses]
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    loss_trends[key] = trend
            
            # 调整权重：下降趋势的损失减小权重，上升趋势的损失增加权重
            if 'reconstruction' in loss_trends:
                if loss_trends['reconstruction'] < -0.01:  # 快速下降
                    self.rec_weight = torch.clamp(self.rec_weight * 0.95, 0.1, 2.0)
                elif loss_trends['reconstruction'] > 0.01:  # 上升或停滞
                    self.rec_weight = torch.clamp(self.rec_weight * 1.05, 0.1, 2.0)
            
            if 'spectral' in loss_trends:
                if loss_trends['spectral'] < -0.005:
                    self.spec_weight = torch.clamp(self.spec_weight * 0.95, 0.05, 1.0)
                elif loss_trends['spectral'] > 0.005:
                    self.spec_weight = torch.clamp(self.spec_weight * 1.05, 0.05, 1.0)
            
            if 'degradation_consistency' in loss_trends:
                if loss_trends['degradation_consistency'] < -0.005:
                    self.dc_weight = torch.clamp(self.dc_weight * 0.95, 0.05, 1.0)
                elif loss_trends['degradation_consistency'] > 0.005:
                    self.dc_weight = torch.clamp(self.dc_weight * 1.05, 0.05, 1.0)
        
        # 应用权重
        total_loss = (self.rec_weight * loss_dict['reconstruction'] +
                     self.spec_weight * loss_dict['spectral'] +
                     self.dc_weight * loss_dict['degradation_consistency'] +
                     self.grad_weight * loss_dict['gradient'])
        
        return {
            'total_loss': total_loss,
            'weights': {
                'reconstruction': self.rec_weight.item(),
                'spectral': self.spec_weight.item(),
                'degradation_consistency': self.dc_weight.item(),
                'gradient': self.grad_weight.item()
            }
        }


def compute_enhanced_total_loss(
    pred_z: torch.Tensor, 
    target_z: torch.Tensor, 
    obs_data: Dict, 
    norm_stats: Optional[Dict[str, torch.Tensor]], 
    config: DictConfig,
    epoch: Optional[int] = None,
    loss_weights_override: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """计算增强总损失 - 集成所有改进的损失函数"""
    
    device = pred_z.device
    
    # 获取损失权重
    if loss_weights_override is not None:
        w_rec = loss_weights_override.get('reconstruction', 1.0)
        w_spec = loss_weights_override.get('spectral', 0.5)
        w_dc = loss_weights_override.get('data_consistency', 0.5)
        w_grad = loss_weights_override.get('gradient', 0.1)
    else:
        # 从配置获取权重
        w_rec = 1.0
        w_spec = 0.5
        w_dc = 0.5
        w_grad = 0.1
        
        if hasattr(config, 'loss'):
            if hasattr(config.loss, 'reconstruction') and hasattr(config.loss.reconstruction, 'weight'):
                w_rec = float(config.loss.reconstruction.weight)
            if hasattr(config.loss, 'spectral') and hasattr(config.loss.spectral, 'weight'):
                w_spec = float(config.loss.spectral.weight)
            if hasattr(config.loss, 'degradation_consistency') and hasattr(config.loss.degradation_consistency, 'weight'):
                w_dc = float(config.loss.degradation_consistency.weight)
            if hasattr(config.loss, 'gradient_weight'):
                w_grad = float(config.loss.gradient_weight)
    
    # 重建损失（z-score域）
    rec_loss_fn = EnhancedReconstructionLoss(config.get('reconstruction', {}))
    rec_losses = rec_loss_fn(pred_z, target_z)
    
    # 频谱损失（需要反归一化到原值域）
    if w_spec > 0:
        if norm_stats is not None:
            # 确保归一化统计数据的形状正确
            sigma = norm_stats['sigma'].to(device).view(1, -1, 1, 1)  # [1, C, 1, 1]
            mu = norm_stats['mu'].to(device).view(1, -1, 1, 1)      # [1, C, 1, 1]
            pred_orig = pred_z * sigma + mu
            target_orig = target_z * sigma + mu
        else:
            pred_orig = pred_z
            target_orig = target_z
        
        spec_loss_fn = AdaptiveSpectralLoss(config.get('spectral', {}))
        spectral_loss = spec_loss_fn(pred_orig, target_orig, epoch)
    else:
        spectral_loss = torch.tensor(0.0, device=device)
    
    # 数据一致性损失
    if w_dc > 0 and 'observation' in obs_data:
        if norm_stats is not None:
            # 确保归一化统计数据的形状正确（复用之前的计算）
            if w_spec > 0:  # 如果已经计算过，直接使用
                pred_orig = pred_orig
            else:  # 否则重新计算
                sigma = norm_stats['sigma'].to(device).view(1, -1, 1, 1)  # [1, C, 1, 1]
                mu = norm_stats['mu'].to(device).view(1, -1, 1, 1)      # [1, C, 1, 1]
                pred_orig = pred_z * sigma + mu
        else:
            pred_orig = pred_z
        
        dc_loss_fn = EnhancedDCLoss(config.get('degradation_consistency', {}))
        
        # 获取观测数据
        obs_tensor = obs_data['observation']
        if isinstance(obs_tensor, dict):
            obs_tensor = obs_tensor.get('data', obs_tensor)
        
        # 获取观测参数
        h_params = obs_data.get('h_params', {
            'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'
        })
        
        dc_loss = dc_loss_fn(pred_orig, obs_tensor, h_params, epoch)
    else:
        dc_loss = torch.tensor(0.0, device=device)
    
    # 梯度损失（已在重建损失中计算）
    gradient_loss = rec_losses['gradient_loss']
    
    # 组装损失字典
    loss_dict = {
        'reconstruction': rec_losses['base_loss'],
        'spectral': spectral_loss,
        'degradation_consistency': dc_loss,
        'gradient': gradient_loss
    }
    
    # 自适应权重调整
    weight_adjuster = AdaptiveLossWeights(config.get('adaptive_weights', {}))
    weighted_losses = weight_adjuster(loss_dict, epoch)
    
    # 总损失
    total_loss = weighted_losses['total_loss']
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': rec_losses['base_loss'],
        'spectral_loss': spectral_loss,
        'dc_loss': dc_loss,
        'gradient_loss': gradient_loss,
        'weights': weighted_losses['weights']
    }