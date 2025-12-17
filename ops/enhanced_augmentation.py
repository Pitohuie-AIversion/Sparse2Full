"""
增强数据增强模块 - 防止小数据集过拟合
提供多种数据增强策略和自适应增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torchvision import transforms
import random


class AdaptiveDataAugmentation(nn.Module):
    """自适应数据增强 - 根据训练进度动态调整增强强度"""
    
    def __init__(self, config: Dict, initial_intensity: float = 0.5, final_intensity: float = 0.1):
        super().__init__()
        self.config = config
        self.initial_intensity = initial_intensity
        self.final_intensity = final_intensity
        self.current_intensity = initial_intensity
        self.current_epoch = 0
        self.max_epochs = config.get('max_epochs', 200)
        
        # 增强操作
        self.spatial_aug = SpatialAugmentation(config)
        self.intensity_aug = IntensityAugmentation(config)
        self.frequency_aug = FrequencyAugmentation(config)
        
    def update_intensity(self, epoch: int):
        """根据训练进度更新增强强度"""
        self.current_epoch = epoch
        # 线性衰减增强强度
        progress = min(epoch / self.max_epochs, 1.0)
        self.current_intensity = self.initial_intensity * (1 - progress) + self.final_intensity * progress
    
    def forward(self, x: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        """应用自适应增强"""
        if mode != 'train':
            return x
        
        # 应用空间增强
        if random.random() < self.current_intensity:
            x = self.spatial_aug(x, intensity=self.current_intensity)
        
        # 应用强度增强
        if random.random() < self.current_intensity * 0.7:
            x = self.intensity_aug(x, intensity=self.current_intensity)
        
        # 应用频率增强（较少使用）
        if random.random() < self.current_intensity * 0.3:
            x = self.frequency_aug(x, intensity=self.current_intensity)
        
        return x


class SpatialAugmentation(nn.Module):
    """空间域增强 - 旋转、翻转、平移等"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """应用空间增强"""
        B, C, H, W = x.shape
        
        # 随机水平翻转
        if random.random() < 0.5 * intensity:
            x = torch.flip(x, dims=[-1])
        
        # 随机垂直翻转
        if random.random() < 0.3 * intensity:
            x = torch.flip(x, dims=[-2])
        
        # 随机旋转（90度的倍数）
        if random.random() < 0.4 * intensity:
            k = random.randint(1, 3)  # 90, 180, 270度
            x = torch.rot90(x, k=k, dims=[-2, -1])
        
        # 轻微平移（最多5%图像尺寸）
        if random.random() < 0.3 * intensity:
            max_shift = int(0.05 * min(H, W))
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            x = self._translate(x, shift_x, shift_y)
        
        return x
    
    def _translate(self, x: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
        """平移张量"""
        B, C, H, W = x.shape
        
        # 创建平移后的张量
        translated = torch.zeros_like(x)
        
        # 计算有效区域
        start_x = max(0, shift_x)
        end_x = min(W, W + shift_x)
        start_y = max(0, shift_y)
        end_y = min(H, H + shift_y)
        
        # 计算源区域
        src_start_x = max(0, -shift_x)
        src_end_x = min(W, W - shift_x)
        src_start_y = max(0, -shift_y)
        src_end_y = min(H, H - shift_y)
        
        if start_x < end_x and start_y < end_y:
            translated[:, :, start_y:end_y, start_x:end_x] = x[:, :, src_start_y:src_end_y, src_start_x:src_end_x]
        
        return translated


class IntensityAugmentation(nn.Module):
    """强度域增强 - 亮度、对比度、噪声等"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """应用强度增强"""
        # 添加高斯噪声
        if random.random() < 0.6 * intensity:
            noise_std = 0.01 * intensity
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        
        # 随机亮度调整
        if random.random() < 0.4 * intensity:
            brightness_factor = 1.0 + random.uniform(-0.1, 0.1) * intensity
            x = x * brightness_factor
        
        # 随机对比度调整
        if random.random() < 0.3 * intensity:
            contrast_factor = 1.0 + random.uniform(-0.1, 0.1) * intensity
            mean = x.mean(dim=[-2, -1], keepdim=True)
            x = (x - mean) * contrast_factor + mean
        
        # 随机gamma调整
        if random.random() < 0.2 * intensity:
            gamma = 1.0 + random.uniform(-0.1, 0.1) * intensity
            x = torch.sign(x) * torch.abs(x) ** gamma
        
        return x


class FrequencyAugmentation(nn.Module):
    """频率域增强 - 低通/高通滤波、频谱扰动等"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """应用频率增强"""
        B, C, H, W = x.shape
        
        # 转换为频域
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # 创建频率掩码
        mask = self._create_frequency_mask(H, W, intensity)
        mask = mask.to(x.device)
        
        # 应用频率掩码
        x_fft_shifted = x_fft_shifted * mask
        
        # 转换回空间域
        x_fft_ishifted = torch.fft.ifftshift(x_fft_shifted, dim=(-2, -1))
        x_augmented = torch.fft.ifft2(x_fft_ishifted, dim=(-2, -1))
        
        # 取实部
        x_augmented = x_augmented.real
        
        return x_augmented
    
    def _create_frequency_mask(self, H: int, W: int, intensity: float) -> torch.Tensor:
        """创建频率掩码"""
        # 创建中心在中心的坐标网格
        y = torch.arange(H).float() - H // 2
        x = torch.arange(W).float() - W // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 计算到中心的距离
        distance = torch.sqrt(yy**2 + xx**2)
        
        # 创建低通滤波器（保留低频成分）
        cutoff_freq = int(0.3 * min(H, W) * intensity)
        lowpass_mask = torch.exp(-distance / (2 * cutoff_freq**2))
        
        # 创建高通滤波器（保留高频成分）
        highpass_mask = 1.0 - lowpass_mask
        
        # 随机选择滤波器类型
        if random.random() < 0.5:
            return lowpass_mask
        else:
            return highpass_mask


class MixupAugmentation(nn.Module):
    """Mixup增强 - 样本混合"""
    
    def __init__(self, alpha: float = 0.2):
        super().__init__
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """应用Mixup增强"""
        if self.training and random.random() < 0.5:
            batch_size = x.size(0)
            
            # 生成混合系数
            lam = np.random.beta(self.alpha, self.alpha)
            
            # 随机打乱样本
            indices = torch.randperm(batch_size).to(x.device)
            x_shuffled = x[indices]
            
            # 混合样本
            x_mixed = lam * x + (1 - lam) * x_shuffled
            
            if y is not None:
                y_shuffled = y[indices]
                y_mixed = lam * y + (1 - lam) * y_shuffled
                return x_mixed, y_mixed
            
            return x_mixed
        
        if y is not None:
            return x, y
        return x


class CutMixAugmentation(nn.Module):
    """CutMix增强 - 区域混合"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """应用CutMix增强"""
        if self.training and random.random() < 0.3:
            batch_size = x.size(0)
            _, _, H, W = x.shape
            
            # 生成混合系数
            lam = np.random.beta(self.alpha, self.alpha)
            
            # 计算裁剪区域
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            # 随机选择裁剪位置
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # 随机选择另一个样本
            rand_idx = np.random.randint(batch_size)
            
            # 应用CutMix
            x_cutmix = x.clone()
            x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[rand_idx, :, bby1:bby2, bbx1:bbx2]
            
            # 调整混合系数
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            
            if y is not None:
                y_cutmix = y.clone()
                y_cutmix = lam * y + (1 - lam) * y[rand_idx]
                return x_cutmix, y_cutmix
            
            return x_cutmix
        
        if y is not None:
            return x, y
        return x


class AdvancedDataAugmentation(nn.Module):
    """高级数据增强组合器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 初始化各种增强器
        self.adaptive_aug = AdaptiveDataAugmentation(config)
        self.mixup_aug = MixupAugmentation(alpha=config.get('mixup_alpha', 0.2))
        self.cutmix_aug = CutMixAugmentation(alpha=config.get('cutmix_alpha', 1.0))
        
        # 增强概率
        self.mixup_prob = config.get('mixup_prob', 0.2)
        self.cutmix_prob = config.get('cutmix_prob', 0.1)
        
    def update_epoch(self, epoch: int):
        """更新训练周期"""
        self.adaptive_aug.update_intensity(epoch)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, 
                epoch: int = 0, mode: str = 'train') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """应用高级数据增强"""
        if mode != 'train':
            return (x, y) if y is not None else x
        
        # 更新周期
        self.update_epoch(epoch)
        
        # 应用自适应增强
        x = self.adaptive_aug(x, mode='train')
        
        # 应用Mixup或CutMix（二选一）
        if random.random() < self.mixup_prob:
            return self.mixup_aug(x, y)
        elif random.random() < self.cutmix_prob:
            return self.cutmix_aug(x, y)
        
        if y is not None:
            return x, y
        return x


def create_augmentation_pipeline(config: Dict) -> AdvancedDataAugmentation:
    """创建增强管道"""
    return AdvancedDataAugmentation(config)