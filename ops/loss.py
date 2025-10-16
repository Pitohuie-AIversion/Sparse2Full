"""损失函数模块

实现PDEBench稀疏观测重建系统的损失函数。
按照开发手册要求，实现三件套损失：L_rec + λ_s L_spec + λ_dc L_dc。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np

from .degradation import apply_degradation_operator


class ReconstructionLoss(nn.Module):
    """重建损失 L_rec
    
    支持L1、L2、Huber等损失类型。
    """
    
    def __init__(self, loss_type: str = "l2", reduction: str = "mean"):
        """初始化重建损失
        
        Args:
            loss_type: 损失类型 ('l1', 'l2', 'huber', 'smooth_l1')
            reduction: 归约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        
        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction=reduction)
        elif self.loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算重建损失
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 目标结果 [B, C, H, W]
            
        Returns:
            重建损失
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"Warning: pred contains NaN or Inf values")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"Warning: target contains NaN or Inf values")
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        
        loss = self.loss_fn(pred, target)
        
        # 检查损失是否为NaN
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: reconstruction loss is NaN or Inf, returning zero loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss


class SpectralLoss(nn.Module):
    """频域损失 L_spec
    
    比较低频成分，支持非周期信号的镜像延拓。
    """
    
    def __init__(
        self, 
        low_freq_modes: int = 16,
        loss_type: str = "l2",
        reduction: str = "mean",
        mirror_padding: bool = True
    ):
        """初始化频域损失
        
        Args:
            low_freq_modes: 低频模数 (kx=ky=low_freq_modes)
            loss_type: 损失类型
            reduction: 归约方式
            mirror_padding: 是否使用镜像延拓处理非周期信号
        """
        super().__init__()
        self.low_freq_modes = low_freq_modes
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.mirror_padding = mirror_padding
        
        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported spectral loss type: {self.loss_type}")
    
    def _apply_mirror_padding(self, x: torch.Tensor) -> torch.Tensor:
        """应用镜像延拓
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            镜像延拓后的张量 [B, C, 2H, 2W]
        """
        # 水平镜像
        x_h_flip = torch.flip(x, dims=[-1])
        x_extended_h = torch.cat([x, x_h_flip], dim=-1)
        
        # 垂直镜像
        x_v_flip = torch.flip(x_extended_h, dims=[-2])
        x_extended = torch.cat([x_extended_h, x_v_flip], dim=-2)
        
        return x_extended
    
    def _extract_low_freq_modes(self, x_fft: torch.Tensor) -> torch.Tensor:
        """提取低频模
        
        Args:
            x_fft: FFT结果 [B, C, H, W]
            
        Returns:
            低频模 [B, C, low_freq_modes, low_freq_modes]
        """
        B, C, H, W = x_fft.shape
        
        # 确保不超过实际尺寸
        modes = min(self.low_freq_modes, H // 2, W // 2)
        
        # 提取低频模（DC分量在中心）
        h_center, w_center = H // 2, W // 2
        
        # 确保索引不会越界
        h_start = max(0, h_center - modes // 2)
        h_end = min(H, h_center + modes // 2)
        w_start = max(0, w_center - modes // 2)
        w_end = min(W, w_center + modes // 2)
        
        low_freq = x_fft[
            :, :,
            h_start:h_end,
            w_start:w_end
        ]
        
        return low_freq
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频域损失
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 目标结果 [B, C, H, W]
            
        Returns:
            频域损失
        """
        # 镜像延拓（如果启用）
        if self.mirror_padding:
            pred_padded = self._apply_mirror_padding(pred)
            target_padded = self._apply_mirror_padding(target)
        else:
            pred_padded = pred
            target_padded = target
        
        # FFT变换
        pred_fft = torch.fft.fft2(pred_padded, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_padded, dim=(-2, -1))
        
        # 移动DC分量到中心
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))
        
        # 提取低频模
        pred_low_freq = self._extract_low_freq_modes(pred_fft)
        target_low_freq = self._extract_low_freq_modes(target_fft)
        
        # 计算损失（使用幅度）
        pred_mag = torch.abs(pred_low_freq)
        target_mag = torch.abs(target_low_freq)
        
        # 添加数值稳定性检查
        if torch.isnan(pred_mag).any() or torch.isinf(pred_mag).any():
            print(f"Warning: NaN/Inf in pred_mag, shape: {pred_mag.shape}, range: [{pred_mag.min():.6f}, {pred_mag.max():.6f}]")
            pred_mag = torch.nan_to_num(pred_mag, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(target_mag).any() or torch.isinf(target_mag).any():
            print(f"Warning: NaN/Inf in target_mag, shape: {target_mag.shape}, range: [{target_mag.min():.6f}, {target_mag.max():.6f}]")
            target_mag = torch.nan_to_num(target_mag, nan=0.0, posinf=1e6, neginf=-1e6)
        
        loss = self.loss_fn(pred_mag, target_mag)
        
        # 增强数值稳定性检查
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: spectral loss is NaN or Inf, returning zero loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 额外的数值范围检查
        if loss > 1e6:
            print(f"Warning: Spectral loss too large ({loss}), clamping to 1e6")
            loss = torch.clamp(loss, max=1e6)
        
        return loss


class DataConsistencyLoss(nn.Module):
    """数据一致性损失 L_dc
    
    确保重建结果经过观测算子H后与观测数据一致。
    """
    
    def __init__(
        self, 
        loss_type: str = "l2",
        reduction: str = "mean",
        denormalize_fn: Optional[callable] = None
    ):
        """初始化数据一致性损失
        
        Args:
            loss_type: 损失类型
            reduction: 归约方式
            denormalize_fn: 反归一化函数（将z-score域转换为原值域）
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.denormalize_fn = denormalize_fn
        
        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported DC loss type: {self.loss_type}")
    
    def forward(
        self, 
        pred: torch.Tensor, 
        observation: torch.Tensor,
        task_params: Dict[str, Any]
    ) -> torch.Tensor:
        """计算数据一致性损失
        
        Args:
            pred: 预测结果 [B, C, H, W] (z-score域)
            observation: 观测数据 [B, C', H', W']
            task_params: 任务参数（用于观测算子H）
            
        Returns:
            数据一致性损失
        """
        # 反归一化到原值域（如果需要）
        if self.denormalize_fn is not None:
            pred_original = self.denormalize_fn(pred)
        else:
            pred_original = pred
        
        # 应用观测算子H
        pred_observation = apply_degradation_operator(pred_original, task_params)
        
        # 确保观测张量和预测观测张量的尺寸匹配
        if pred_observation.shape != observation.shape:
            # 如果尺寸不匹配，将观测张量调整到预测观测张量的尺寸
            observation = F.interpolate(
                observation, 
                size=pred_observation.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 计算损失
        loss = self.loss_fn(pred_observation, observation)
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(pred_observation).any() or torch.isinf(pred_observation).any():
            print(f"Warning: pred_observation contains NaN or Inf values")
            pred_observation = torch.nan_to_num(pred_observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(observation).any() or torch.isinf(observation).any():
            print(f"Warning: observation contains NaN or Inf values")
            observation = torch.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查损失是否为NaN
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: data consistency loss is NaN or Inf, returning zero loss")
            return torch.tensor(0.0, device=pred_observation.device, requires_grad=True)
        
        return loss


class TotalLoss(nn.Module):
    """总损失函数
    
    实现三件套损失：L = L_rec + λ_s L_spec + λ_dc L_dc
    """
    
    def __init__(
        self,
        # 默认权重配置 - 大幅降低频域损失权重以避免数值不稳定
        rec_weight: float = 1.0,
        spec_weight: float = 0.001,  # 进一步降低到0.001
        dc_weight: float = 1.0,
        rec_loss_type: str = "l2",
        spec_loss_type: str = "l2",
        dc_loss_type: str = "l2",
        low_freq_modes: int = 16,
        denormalize_fn: Optional[callable] = None
    ):
        """初始化总损失
        
        Args:
            rec_weight: 重建损失权重
            spec_weight: 频域损失权重
            dc_weight: 数据一致性损失权重
            rec_loss_type: 重建损失类型
            spec_loss_type: 频域损失类型
            dc_loss_type: 数据一致性损失类型
            low_freq_modes: 低频模数
            denormalize_fn: 反归一化函数
        """
        super().__init__()
        
        self.rec_weight = rec_weight
        self.spec_weight = spec_weight
        self.dc_weight = dc_weight
        
        # 初始化各个损失函数
        self.rec_loss = ReconstructionLoss(rec_loss_type)
        # 禁用镜像延拓并添加数值稳定性检查
        self.spec_loss = SpectralLoss(low_freq_modes, spec_loss_type, mirror_padding=False)
        self.dc_loss = DataConsistencyLoss(dc_loss_type, denormalize_fn=denormalize_fn)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        observation: torch.Tensor,
        task_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算总损失
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 目标结果 [B, C, H, W]
            observation: 观测数据 [B, C', H', W']
            task_params: 任务参数
            
        Returns:
            总损失和各项损失的字典
        """
        # 计算各项损失
        loss_rec = self.rec_loss(pred, target)
        loss_spec = self.spec_loss(pred, target)
        loss_dc = self.dc_loss(pred, observation, task_params)
        
        # 检查各项损失的有效性
        if torch.isnan(loss_rec) or torch.isinf(loss_rec):
            print(f"Warning: Invalid reconstruction loss: {loss_rec}")
            loss_rec = torch.tensor(0.0, device=loss_rec.device, dtype=loss_rec.dtype)
        
        if torch.isnan(loss_spec) or torch.isinf(loss_spec):
            print(f"Warning: Invalid spectral loss: {loss_spec}")
            loss_spec = torch.tensor(0.0, device=loss_spec.device, dtype=loss_spec.dtype)
        
        if torch.isnan(loss_dc) or torch.isinf(loss_dc):
            print(f"Warning: Invalid DC loss: {loss_dc}")
            loss_dc = torch.tensor(0.0, device=loss_dc.device, dtype=loss_dc.dtype)
        
        # 加权求和
        total_loss = (
            self.rec_weight * loss_rec +
            self.spec_weight * loss_spec +
            self.dc_weight * loss_dc
        )
        
        # 检查总损失
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid total loss: {total_loss}")
            print(f"  rec_loss: {loss_rec}, spec_loss: {loss_spec}, dc_loss: {loss_dc}")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
        
        # 返回详细损失信息
        loss_dict = {
            'total': total_loss,
            'reconstruction': loss_rec,
            'spectral': loss_spec,
            'data_consistency': loss_dc,
            'rec_weighted': self.rec_weight * loss_rec,
            'spec_weighted': self.spec_weight * loss_spec,
            'dc_weighted': self.dc_weight * loss_dc
        }
        
        return total_loss, loss_dict


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    observation: torch.Tensor,
    task_params: Dict[str, Any],
    loss_config: Optional[Dict[str, Any]] = None,
    denormalize_fn: Optional[callable] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算总损失的便捷函数
    
    Args:
        pred: 预测结果 [B, C, H, W]
        target: 目标结果 [B, C, H, W]
        observation: 观测数据 [B, C', H', W']
        task_params: 任务参数
        loss_config: 损失配置
        denormalize_fn: 反归一化函数
        
    Returns:
        总损失和各项损失的字典
    """
    # 默认配置
    if loss_config is None:
        loss_config = {
            'rec_weight': 1.0,
            'spec_weight': 0.5,
            'dc_weight': 1.0,
            'rec_loss_type': 'l2',
            'spec_loss_type': 'l2',
            'dc_loss_type': 'l2',
            'low_freq_modes': 16
        }
    
    # 创建损失函数
    loss_fn = TotalLoss(
        rec_weight=loss_config.get('rec_weight', 1.0),
        spec_weight=loss_config.get('spec_weight', 0.5),  # 使用配置文件中的权重
        dc_weight=loss_config.get('dc_weight', 1.0),
        rec_loss_type=loss_config.get('rec_loss_type', 'l2'),
        spec_loss_type=loss_config.get('spec_loss_type', 'l2'),
        dc_loss_type=loss_config.get('dc_loss_type', 'l2'),
        low_freq_modes=loss_config.get('low_freq_modes', 16),
        denormalize_fn=denormalize_fn
    )
    
    return loss_fn(pred, target, observation, task_params)


def compute_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l2"
) -> torch.Tensor:
    """计算梯度损失（可选）
    
    Args:
        pred: 预测结果 [B, C, H, W]
        target: 目标结果 [B, C, H, W]
        loss_type: 损失类型
        
    Returns:
        梯度损失
    """
    # 计算梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # 计算损失
    if loss_type.lower() == "l1":
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    else:  # l2
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    
    return loss_x + loss_y


def compute_pde_residual_loss(
    pred: torch.Tensor,
    pde_params: Dict[str, Any],
    loss_type: str = "l2",
    low_freq_weight: float = 2.0
) -> torch.Tensor:
    """计算PDE残差损失（可选）
    
    Args:
        pred: 预测结果 [B, C, H, W]
        pde_params: PDE参数
        loss_type: 损失类型
        low_freq_weight: 低频加权
        
    Returns:
        PDE残差损失
    """
    # 这里是一个简化的实现，实际需要根据具体PDE方程来计算
    # 例如对于扩散方程：∂u/∂t = α∇²u
    
    # 计算拉普拉斯算子
    laplacian = (
        pred[:, :, 2:, 1:-1] + pred[:, :, :-2, 1:-1] +
        pred[:, :, 1:-1, 2:] + pred[:, :, 1:-1, :-2] -
        4 * pred[:, :, 1:-1, 1:-1]
    )
    
    # 简化的残差（这里假设稳态方程）
    residual = laplacian
    
    # 低频加权
    if low_freq_weight > 1.0:
        # 对低频成分加权
        residual_fft = torch.fft.fft2(residual)
        residual_fft = torch.fft.fftshift(residual_fft)
        
        H, W = residual.shape[-2:]
        h_center, w_center = H // 2, W // 2
        low_freq_size = min(16, H // 4, W // 4)
        
        # 创建权重掩码
        weight_mask = torch.ones_like(residual_fft, dtype=torch.float32)
        weight_mask[
            :, :,
            h_center - low_freq_size:h_center + low_freq_size,
            w_center - low_freq_size:w_center + low_freq_size
        ] *= low_freq_weight
        
        residual_fft_weighted = residual_fft * weight_mask
        residual = torch.real(torch.fft.ifft2(torch.fft.ifftshift(residual_fft_weighted)))
    
    # 计算损失
    if loss_type.lower() == "l1":
        return F.l1_loss(residual, torch.zeros_like(residual))
    else:  # l2
        return F.mse_loss(residual, torch.zeros_like(residual))