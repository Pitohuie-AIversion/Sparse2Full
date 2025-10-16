"""数据一致性算子H

实现SR和Crop模式的退化算子，确保与训练DC复用同一实现。
严格按照开发手册要求，保证观测生成H与训练DC的H算子完全一致。
"""

from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn.functional as F
import cv2
import numpy as np


class SuperResolutionOperator:
    """超分辨率退化算子"""
    
    def __init__(self, scale: int = 2, sigma: float = 1.0, kernel_size: int = 5, boundary: str = 'mirror'):
        self.scale = scale
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.boundary = boundary
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """应用SR退化算子"""
        params = {
            'task': 'sr',
            'scale': self.scale,
            'sigma': self.sigma,
            'kernel_size': self.kernel_size,
            'boundary': self.boundary
        }
        return apply_degradation_operator(x, params)


class CropOperator:
    """裁剪退化算子"""
    
    def __init__(self, crop_size: Tuple[int, int], crop_box: Optional[Tuple[int, int, int, int]] = None, boundary: str = 'mirror'):
        self.crop_size = crop_size
        self.crop_box = crop_box
        self.boundary = boundary
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """应用Crop退化算子"""
        params = {
            'task': 'crop',
            'crop_size': self.crop_size,
            'crop_box': self.crop_box,
            'boundary': self.boundary
        }
        return apply_degradation_operator(x, params)


def apply_degradation_operator(
    pred: torch.Tensor, 
    params: Dict
) -> torch.Tensor:
    """应用退化算子H
    
    SR: H_sr(pred) = blur ∘ area_downsample
    - blur: GaussianBlur(kernel_size, sigma)，边界策略：mirror/wrap/zero
    - downsample: F.interpolate(mode='area', align_corners=False)
    - 对齐规则：输出尺寸 = input_size // scale，stride=scale
    
    Crop: H_crop(pred) = crop(pred, box)
    - box: (x1, y1, x2, y2)，基于patch_align对齐
    - 边界处理：超出部分按boundary策略填充
    
    Args:
        pred: 预测张量（原值域）[B, C, H, W]
        params: H算子参数字典
        
    Returns:
        退化后的张量
    """
    # 兼容新旧参数格式
    task = params.get('task', params.get('task_type', 'sr'))
    
    # 确保task是字符串类型
    if isinstance(task, (list, tuple)):
        task = task[0]
    if not isinstance(task, str):
        task = str(task)
    
    if task.lower() in ["sr", "super_resolution"]:
        return _apply_sr_degradation(pred, params)
    elif task.lower() in ["crop", "cropping"]:
        return _apply_crop_degradation(pred, params)
    else:
        raise ValueError(f"Unknown task: {task}")


def _apply_sr_degradation(pred: torch.Tensor, params: Dict) -> torch.Tensor:
    """应用SR退化算子：blur + downsample
    
    Args:
        pred: 输入张量 [B, C, H, W]
        params: SR参数 {scale, sigma, kernel_size, boundary}
        
    Returns:
        下采样后的张量 [B, C, H//scale, W//scale]
    """
    # 兼容新旧参数格式
    scale = params.get('scale', params.get('scale_factor', 2))
    sigma = params.get('sigma', 1.0)
    kernel_size = params.get('kernel_size', 5)
    boundary = params.get('boundary', 'mirror')
    
    # 确保参数是标量值
    if isinstance(boundary, (list, tuple)):
        boundary = boundary[0]
    if isinstance(sigma, (list, tuple)):
        sigma = sigma[0]
    
    B, C, H, W = pred.shape
    
    # 1. 高斯模糊
    blurred = _gaussian_blur(pred, kernel_size, sigma, boundary)
    
    # 2. 区域下采样
    # 确保scale是标量值
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 1:
            scale = scale.item()
        else:
            scale = scale.flatten()[0].item()
    if isinstance(scale, (list, tuple)):
        scale = scale[0]
    scale = int(scale)
    
    target_h = H // scale
    target_w = W // scale
    
    downsampled = F.interpolate(
        blurred,
        size=(target_h, target_w),
        mode='area',
        # align_corners=False  # area模式不支持align_corners
    )
    
    return downsampled


def _apply_crop_degradation(pred: torch.Tensor, params: Dict) -> torch.Tensor:
    """应用Crop退化算子：区域提取
    
    Args:
        pred: 输入张量 [B, C, H, W]
        params: Crop参数 {crop_size, crop_box, boundary}
        
    Returns:
        裁剪后的张量 [B, C, crop_h, crop_w]
    """
    crop_box = params.get('crop_box')
    crop_size = params['crop_size']
    boundary = params.get('boundary', 'mirror')
    
    B, C, H, W = pred.shape
    crop_h, crop_w = crop_size
    
    if crop_box is None:
        # 如果没有指定crop_box，使用中心裁剪
        center_y, center_x = H // 2, W // 2
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(H, y1 + crop_h)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(W, x1 + crop_w)
        
        crop_box = (x1, y1, x2, y2)
    
    x1, y1, x2, y2 = crop_box
    
    # 确保crop_box在有效范围内
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(x1, min(x2, W))
    y2 = max(y1, min(y2, H))
    
    # 直接裁剪
    cropped = pred[:, :, y1:y2, x1:x2]
    
    # 如果裁剪区域尺寸不足，需要进行边界处理
    actual_h, actual_w = cropped.shape[-2:]
    
    if actual_h != crop_h or actual_w != crop_w:
        # 需要填充到期望尺寸
        cropped = _pad_to_size(cropped, crop_h, crop_w, boundary)
    
    return cropped


def _gaussian_blur(
    x: torch.Tensor, 
    kernel_size: int, 
    sigma: float, 
    boundary: str = 'mirror'
) -> torch.Tensor:
    """高斯模糊
    
    Args:
        x: 输入张量 [B, C, H, W]
        kernel_size: 核大小
        sigma: 标准差
        boundary: 边界策略
        
    Returns:
        模糊后的张量
    """
    # 确保kernel_size为奇数
    if isinstance(kernel_size, torch.Tensor):
        if kernel_size.numel() == 1:
            kernel_size = kernel_size.item()
        else:
            # 如果是多元素张量，取第一个元素
            kernel_size = kernel_size.flatten()[0].item()
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = kernel_size[0]
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建高斯核
    kernel = _create_gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    # 确保kernel与输入张量的数据类型一致
    kernel = kernel.to(dtype=x.dtype)
    
    # 边界填充
    pad_size = kernel_size // 2
    if boundary == 'mirror' or boundary == 'reflect':
        x_padded = F.pad(x, [pad_size] * 4, mode='reflect')
    elif boundary == 'wrap' or boundary == 'circular':
        x_padded = F.pad(x, [pad_size] * 4, mode='circular')
    elif boundary == 'zero':
        x_padded = F.pad(x, [pad_size] * 4, mode='constant', value=0)
    elif boundary == 'replicate':
        x_padded = F.pad(x, [pad_size] * 4, mode='replicate')
    else:
        raise ValueError(f"Unknown boundary mode: {boundary}")
    
    # 应用卷积
    B, C, H_pad, W_pad = x_padded.shape
    x_padded = x_padded.view(B * C, 1, H_pad, W_pad)
    
    blurred = F.conv2d(x_padded, kernel, padding=0)
    blurred = blurred.view(B, C, blurred.shape[-2], blurred.shape[-1])
    
    return blurred


def _create_gaussian_kernel(
    kernel_size: int, 
    sigma: float, 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    """创建高斯卷积核
    
    Args:
        kernel_size: 核大小
        sigma: 标准差
        device: 设备
        dtype: 数据类型
        
    Returns:
        高斯核 [1, 1, kernel_size, kernel_size]
    """
    # 创建坐标网格
    # 确保kernel_size是标量值
    if isinstance(kernel_size, torch.Tensor):
        kernel_size = kernel_size.item()
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords = coords - kernel_size // 2
    
    # 计算高斯权重
    # 确保sigma在正确的设备上
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() == 1:
            sigma = sigma.item()
        else:
            # 如果是多元素张量，取第一个元素
            sigma = sigma.flatten()[0].item()
    if isinstance(sigma, (list, tuple)):
        sigma = sigma[0]
    sigma = float(sigma)
    sigma = torch.tensor(sigma, dtype=dtype, device=device)
    gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # 创建2D核
    kernel = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = kernel / kernel.sum()
    
    # 调整形状为 [1, 1, H, W]
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    return kernel


def _pad_to_size(
    x: torch.Tensor, 
    target_h: int, 
    target_w: int, 
    boundary: str = 'mirror'
) -> torch.Tensor:
    """填充张量到目标尺寸
    
    Args:
        x: 输入张量 [B, C, H, W]
        target_h: 目标高度
        target_w: 目标宽度
        boundary: 边界策略
        
    Returns:
        填充后的张量 [B, C, target_h, target_w]
    """
    B, C, H, W = x.shape
    
    if H >= target_h and W >= target_w:
        # 如果当前尺寸已经足够，直接裁剪
        return x[:, :, :target_h, :target_w]
    
    # 计算需要填充的大小
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    
    # 计算各边填充大小
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 应用填充
    if boundary == 'mirror' or boundary == 'reflect':
        padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
    elif boundary == 'wrap' or boundary == 'circular':
        padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='circular')
    elif boundary == 'zero':
        padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0)
    elif boundary == 'replicate':
        padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='replicate')
    else:
        raise ValueError(f"Unknown boundary mode: {boundary}")
    
    # 确保尺寸正确
    if padded.shape[-2:] != (target_h, target_w):
        padded = padded[:, :, :target_h, :target_w]
    
    return padded


def verify_degradation_consistency(
    target: torch.Tensor,
    observation: torch.Tensor, 
    h_params: Dict,
    tolerance: float = 1e-8
) -> Dict[str, float]:
    """验证退化算子一致性
    
    验证 MSE(H(GT), y) < tolerance，确保观测生成H与训练DC的H算子一致。
    
    Args:
        target: 真值 [B, C, H, W]
        observation: 观测数据 [B, C, H', W']
        h_params: H算子参数
        tolerance: 容忍度
        
    Returns:
        验证结果字典 {mse, max_error, passed}
    """
    try:
        # 应用H算子到真值
        h_target = apply_degradation_operator(target, h_params)
        
        # 检查尺寸是否匹配
        if h_target.shape != observation.shape:
            # 如果尺寸不匹配，尝试调整到相同尺寸
            if h_target.numel() == observation.numel():
                # 如果元素数量相同，reshape到相同形状
                h_target = h_target.view(observation.shape)
            else:
                # 如果元素数量不同，使用插值调整尺寸
                h_target = F.interpolate(
                    h_target, 
                    size=observation.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # 计算误差
        mse = F.mse_loss(h_target, observation).item()
        max_error = torch.max(torch.abs(h_target - observation)).item()
        
        passed = mse < tolerance
        
        return {
            'mse': mse,
            'max_error': max_error,
            'tolerance': tolerance,
            'passed': passed,
            'h_target_shape': list(h_target.shape),
            'observation_shape': list(observation.shape)
        }
        
    except Exception as e:
        # 如果出现任何错误，返回失败结果
        return {
            'mse': float('inf'),
            'max_error': float('inf'),
            'tolerance': tolerance,
            'passed': False,
            'error': str(e),
            'h_target_shape': list(target.shape) if 'h_target' not in locals() else list(h_target.shape),
            'observation_shape': list(observation.shape)
        }