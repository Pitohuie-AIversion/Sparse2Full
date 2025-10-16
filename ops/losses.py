"""损失函数系统

实现三件套损失：L = L_rec + λ_s L_spec + λ_dc L_dc
严格按照开发手册要求，确保值域正确处理。
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .degradation import apply_degradation_operator


def compute_total_loss(
    pred_z: torch.Tensor, 
    target_z: torch.Tensor, 
    obs_data: Dict, 
    norm_stats: Optional[Dict[str, torch.Tensor]], 
    config: DictConfig
) -> Dict[str, torch.Tensor]:
    """计算总损失，包含重建损失、频谱损失和数据一致性损失
    
    **值域说明**：
    - 模型输出默认在z-score域（标准化后）
    - DC损失和谱损失在原值域计算（需反归一化：pred_orig = pred_z * sigma + mu）
    - 重建损失可在z-score域直接计算
    
    **损失计算规则**：
    - 输入期望：pred_z, target_z（z-score域），norm_stats（归一化统计量）
    - 频域损失：默认只比较前kx=ky=16的rFFT系数，非周期边界用镜像延拓
    - DC验收：对GT调用H与生成y的MSE < 1e-8视为通过
    
    Args:
        pred_z: 模型预测（z-score域）[B, C, H, W]
        target_z: 真值标签（z-score域）[B, C, H, W]
        obs_data: 观测数据字典，包含baseline、mask、coords、h_params、observation
        norm_stats: 归一化统计量，用于反归一化到原值域
        config: 损失权重配置
    
    Returns:
        Dict包含各损失分量：reconstruction_loss, spectral_loss, dc_loss, total_loss
    """
    device = pred_z.device
    B, C, H, W = pred_z.shape
    
    # 获取损失权重
    w_rec = config.loss.rec_weight
    w_spec = config.loss.spec_weight
    w_dc = config.loss.dc_weight
    w_grad = config.loss.get('gradient_weight', 0.0)
    
    losses = {}
    
    # 1. 重建损失（在z-score域计算）
    reconstruction_loss = _compute_reconstruction_loss(pred_z, target_z, obs_data)
    losses['reconstruction_loss'] = reconstruction_loss
    
    # 2. 频谱损失（在原值域计算）
    if w_spec > 0:
        pred_orig = _denormalize_tensor(pred_z, norm_stats, config.data['keys'])
        target_orig = _denormalize_tensor(target_z, norm_stats, config.data['keys'])
        spectral_loss = _compute_spectral_loss(pred_orig, target_orig, config)
        losses['spectral_loss'] = spectral_loss
    else:
        losses['spectral_loss'] = torch.tensor(0.0, device=device)
    
    # 3. 数据一致性损失（在原值域计算）
    if w_dc > 0:
        pred_orig = _denormalize_tensor(pred_z, norm_stats, config.data['keys'])
        dc_loss = _compute_data_consistency_loss(pred_orig, obs_data, norm_stats, config.data['keys'])
        losses['dc_loss'] = dc_loss
    else:
        losses['dc_loss'] = torch.tensor(0.0, device=device)
    
    # 4. 梯度损失（可选，在z-score域计算）
    if w_grad > 0:
        gradient_loss = _compute_gradient_loss(pred_z, target_z)
        losses['gradient_loss'] = gradient_loss
    else:
        losses['gradient_loss'] = torch.tensor(0.0, device=device)
    
    # 5. 总损失
    total_loss = (
        w_rec * losses['reconstruction_loss'] +
        w_spec * losses['spectral_loss'] +
        w_dc * losses['dc_loss'] +
        w_grad * losses['gradient_loss']
    )
    losses['total_loss'] = total_loss
    
    return losses


def _compute_reconstruction_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    obs_data: Dict
) -> torch.Tensor:
    """计算重建损失
    
    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]
        obs_data: 观测数据
        
    Returns:
        重建损失
    """
    # 使用相对L2损失作为主要重建损失
    rel_l2 = _compute_relative_l2_loss(pred, target)
    
    # 可选：添加MAE损失
    mae = F.l1_loss(pred, target)
    
    # 组合损失（主要使用Rel-L2）
    reconstruction_loss = rel_l2 + 0.1 * mae
    
    return reconstruction_loss


def _compute_spectral_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    config: DictConfig
) -> torch.Tensor:
    """计算频谱损失
    
    仅比较前kx=ky=16的rFFT系数，非周期边界用镜像延拓
    
    Args:
        pred: 预测（原值域）[B, C, H, W]
        target: 真值（原值域）[B, C, H, W]
        config: 配置
        
    Returns:
        频谱损失
    """
    low_freq_modes = getattr(config.loss, 'low_freq_modes', 16)
    use_rfft = getattr(config.loss, 'use_rfft', False)
    normalize = getattr(config.loss, 'normalize', False)
    
    B, C, H, W = pred.shape
    
    # 镜像延拓（用于非周期边界）
    pred_extended = _mirror_extend(pred)
    target_extended = _mirror_extend(target)
    
    spectral_losses = []
    
    for c in range(C):
        pred_c = pred_extended[:, c]  # [B, H_ext, W_ext]
        target_c = target_extended[:, c]
        
        if use_rfft:
            # 使用实数FFT
            pred_fft = torch.fft.rfft2(pred_c, norm='ortho' if normalize else None)
            target_fft = torch.fft.rfft2(target_c, norm='ortho' if normalize else None)
        else:
            # 使用复数FFT
            pred_fft = torch.fft.fft2(pred_c, norm='ortho' if normalize else None)
            target_fft = torch.fft.fft2(target_c, norm='ortho' if normalize else None)
        
        # 只比较低频部分
        low_freq_modes_int = int(low_freq_modes)
        pred_fft_low = pred_fft[:, :low_freq_modes_int, :low_freq_modes_int]
        target_fft_low = target_fft[:, :low_freq_modes_int, :low_freq_modes_int]
        
        # 计算频谱损失（使用L2损失）
        spectral_loss_c = F.mse_loss(pred_fft_low.real, target_fft_low.real) + \
                         F.mse_loss(pred_fft_low.imag, target_fft_low.imag)
        
        spectral_losses.append(spectral_loss_c)
    
    # 多通道平均
    spectral_loss = torch.stack(spectral_losses).mean()
    
    return spectral_loss


def _compute_data_consistency_loss(
    pred: torch.Tensor, 
    obs_data: Dict,
    norm_stats: Optional[Dict[str, torch.Tensor]],
    keys: list
) -> torch.Tensor:
    """计算数据一致性损失
    
    DC损失：‖H(ŷ)−y‖₂
    
    Args:
        pred: 预测（原值域）[B, C, H, W]
        obs_data: 观测数据字典
        norm_stats: 归一化统计量
        keys: 数据键名列表
        
    Returns:
        数据一致性损失
    """
    h_params = obs_data['h_params']
    
    # 应用H算子到预测
    h_pred = apply_degradation_operator(pred, h_params)
    
    # 获取对应的观测数据（原值域）
    observation = obs_data.get('observation')
    
    if observation is None:
        # 如果没有直接的观测数据，从baseline生成
        baseline_z = obs_data.get('baseline')  # z-score域
        if baseline_z is not None and norm_stats is not None:
            # 反归一化baseline到原值域
            baseline_orig = _denormalize_tensor(baseline_z, norm_stats, keys)
            # 应用H算子生成观测
            observation = apply_degradation_operator(baseline_orig, h_params)
        else:
            # 无法获取观测数据，返回零损失
            return torch.tensor(0.0, device=pred.device)
    
    # 确保observation在原值域且维度匹配
    if observation.shape != h_pred.shape:
        # 调整尺寸
        observation = F.interpolate(observation, size=h_pred.shape[-2:], mode='bilinear', align_corners=False)
    
    # 计算DC损失
    dc_loss = F.mse_loss(h_pred, observation)
    
    return dc_loss


def _compute_gradient_loss(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """计算梯度损失
    
    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]
        
    Returns:
        梯度损失
    """
    # 计算梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # 计算梯度损失
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    
    gradient_loss = grad_loss_x + grad_loss_y
    
    return gradient_loss


def _compute_relative_l2_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """计算相对L2损失
    
    Rel-L2 = ‖pred - target‖₂ / (‖target‖₂ + eps)
    
    Args:
        pred: 预测 [B, C, H, W]
        target: 真值 [B, C, H, W]
        eps: 数值稳定性常数
        
    Returns:
        相对L2损失
    """
    # 计算每个样本的相对L2损失
    diff_norm = torch.norm(pred - target, p=2, dim=(1, 2, 3))  # [B]
    target_norm = torch.norm(target, p=2, dim=(1, 2, 3))  # [B]
    
    rel_l2 = diff_norm / (target_norm + eps)
    
    # 返回批次平均
    return rel_l2.mean()


def _denormalize_tensor(
    tensor_z: torch.Tensor, 
    norm_stats: Optional[Dict[str, torch.Tensor]], 
    keys: list
) -> torch.Tensor:
    """反归一化张量到原值域
    
    Args:
        tensor_z: z-score域张量 [B, C, H, W]
        norm_stats: 归一化统计量
        keys: 数据键名列表
        
    Returns:
        原值域张量
    """
    if norm_stats is None:
        return tensor_z
    
    tensor_orig = tensor_z.clone()
    
    for i, key in enumerate(keys):
        if i >= tensor_z.size(1):
            break
            
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        
        if mean_key in norm_stats and std_key in norm_stats:
            mean = norm_stats[mean_key].to(tensor_z.device)
            std = norm_stats[std_key].to(tensor_z.device)
            
            # 确保mean和std的形状正确
            if mean.dim() == 0:
                mean = mean.unsqueeze(0)
            if std.dim() == 0:
                std = std.unsqueeze(0)
            
            # 反归一化：x_orig = x_z * std + mean
            tensor_orig[:, i:i+1] = tensor_z[:, i:i+1] * std.view(1, 1, 1, 1) + mean.view(1, 1, 1, 1)
        else:
            # 如果没有找到对应的归一化统计量，保持原值
            print(f"Warning: No normalization stats found for key '{key}', keeping original values")
    
    return tensor_orig


def _mirror_extend(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """镜像延拓张量（用于非周期边界的FFT）
    
    Args:
        x: 输入张量 [B, C, H, W]
        factor: 延拓倍数
        
    Returns:
        延拓后的张量 [B, C, H*factor, W*factor]
    """
    B, C, H, W = x.shape
    
    # 水平镜像
    x_h_mirror = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)  # [B, C, H, 2W]
    
    # 垂直镜像
    x_extended = torch.cat([x_h_mirror, torch.flip(x_h_mirror, dims=[-2])], dim=-2)  # [B, C, 2H, 2W]
    
    return x_extended


def compute_loss_weights_schedule(
    epoch: int, 
    total_epochs: int, 
    base_weights: Dict[str, float]
) -> Dict[str, float]:
    """计算损失权重调度
    
    可以实现课程学习，例如：
    - 早期阶段重点关注重建损失
    - 后期阶段增加数据一致性损失权重
    
    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        base_weights: 基础权重
        
    Returns:
        调度后的权重
    """
    progress = epoch / total_epochs
    
    # 简单的线性调度示例
    weights = {}
    
    # 处理每个权重，确保从DictConfig中提取数值
    for key, value in base_weights.items():
        # 如果value是DictConfig，提取其中的weight字段
        if hasattr(value, 'weight'):
            base_weight = float(value.weight)
        elif hasattr(value, '_content') and isinstance(value._content, dict):
            # 处理嵌套的DictConfig
            if 'weight' in value._content:
                base_weight = float(value._content['weight'])
            else:
                base_weight = 1.0  # 默认权重
        elif isinstance(value, str):
            # 如果是字符串，跳过或设置默认值
            if key == 'rec_loss_type' or key == 'spec_loss_type' or key == 'dc_loss_type':
                continue  # 跳过损失类型配置
            else:
                base_weight = 1.0  # 默认权重
        elif hasattr(value, '__dict__'):
            # 处理其他类型的配置对象
            try:
                base_weight = float(value)
            except (TypeError, ValueError):
                base_weight = 1.0  # 默认权重
        else:
            try:
                base_weight = float(value)
            except (TypeError, ValueError):
                base_weight = 1.0  # 默认权重
        
        weights[key] = base_weight
    
    # DC损失权重随训练进度增加
    if 'data_consistency' in weights:
        weights['data_consistency'] = weights['data_consistency'] * (0.1 + 0.9 * progress)
    
    # 频谱损失权重在中期达到峰值
    if 'spectral' in weights:
        spectral_factor = 4 * progress * (1 - progress)  # 在0.5处达到峰值1.0
        weights['spectral'] = weights['spectral'] * (0.5 + 0.5 * spectral_factor)
    
    return weights