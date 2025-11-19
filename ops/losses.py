"""损失函数系统

实现三件套损失：L = L_rec + λ_s L_spec + λ_dc L_dc
严格按照开发手册要求，确保值域正确处理。

同时支持自回归(AR)时序预测的损失函数。
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .degradation import apply_degradation_operator

# 为测试兼容提供CombinedLoss别名，复用utils.losses.TotalLoss实现
try:
    from utils.losses import TotalLoss as CombinedLoss  # noqa: F401
except Exception:
    CombinedLoss = None  # 在缺少依赖时保持可导入但不可用


class ARLoss(torch.nn.Module):
    """自回归时序预测损失函数
    
    支持多步预测的损失计算，包括：
    - 逐步损失：每个时间步的预测损失
    - 累积损失：整个序列的总损失
    - 教师强制损失：使用真实值作为输入的损失
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.get('loss_type', 'mse')
        self.step_weights = config.get('step_weights', None)  # 每步的权重
        self.accumulate_loss = config.get('accumulate_loss', True)
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: [B, T_out, C, H, W] 或 [B, C, H, W] 预测序列
            targets: [B, T_out, C, H, W] 或 [B, C, H, W] 目标序列
            mask: [B, T_out, C, H, W] 可选的掩码
            
        Returns:
            Dict包含step_losses, total_loss等
        """
        # 处理不同维度的输入
        if len(predictions.shape) == 4:
            # 如果是4D，添加时间维度
            predictions = predictions.unsqueeze(1)  # [B, 1, C, H, W]
        if len(targets.shape) == 4:
            targets = targets.unsqueeze(1)  # [B, 1, C, H, W]
            
        B, T_out, C, H, W = predictions.shape
        
        # 计算每个时间步的损失
        step_losses = []
        for t in range(T_out):
            pred_t = predictions[:, t]  # [B, C, H, W]
            target_t = targets[:, t]    # [B, C, H, W]
            
            if self.loss_type == 'mse':
                loss_t = F.mse_loss(pred_t, target_t, reduction='none')
            elif self.loss_type == 'l1':
                loss_t = F.l1_loss(pred_t, target_t, reduction='none')
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
            # 应用掩码
            if mask is not None:
                mask_t = mask[:, t]
                loss_t = loss_t * mask_t
                loss_t = loss_t.sum() / (mask_t.sum() + 1e-8)
            else:
                loss_t = loss_t.mean()
            
            step_losses.append(loss_t)
        
        # 计算加权总损失
        if self.step_weights is not None:
            weights = torch.tensor(self.step_weights, device=predictions.device)
            total_loss = sum(w * loss for w, loss in zip(weights, step_losses))
        else:
            total_loss = sum(step_losses) / len(step_losses)
        
        return {
            'step_losses': step_losses,
            'total_loss': total_loss,
            'ar_loss': total_loss
        }


class SpectralLoss(torch.nn.Module):
    """频谱损失函数"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.k_max = config.get('k_max', 16)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频谱损失"""
        # 计算FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 只比较低频部分
        pred_fft_low = pred_fft[..., :self.k_max, :self.k_max]
        target_fft_low = target_fft[..., :self.k_max, :self.k_max]
        
        # 计算损失
        loss = F.mse_loss(pred_fft_low.real, target_fft_low.real) + \
               F.mse_loss(pred_fft_low.imag, target_fft_low.imag)
        
        return loss


class DCLoss(torch.nn.Module):
    """数据一致性损失函数"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
    def forward(self, 
                pred_orig: torch.Tensor, 
                obs_data: torch.Tensor, 
                h_params: Dict) -> torch.Tensor:
        """计算数据一致性损失"""
        # 应用观测算子H到预测结果
        pred_obs = apply_degradation_operator(
            pred_orig, 
            h_params
        )
        
        # 与观测数据比较
        target_obs = obs_data  # obs_data直接是张量，不是字典
        
        # 确保pred_obs和target_obs的尺寸匹配
        if pred_obs.shape != target_obs.shape:
            # 如果尺寸不匹配，将target_obs调整到pred_obs的尺寸
            target_obs = F.interpolate(
                target_obs, 
                size=pred_obs.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        loss = F.mse_loss(pred_obs, target_obs)
        
        return loss


def compute_total_loss(
    pred_z: torch.Tensor, 
    target_z: torch.Tensor, 
    obs_data: Dict, 
    norm_stats: Optional[Dict[str, torch.Tensor]], 
    config: DictConfig,
    loss_weights_override: Optional[Dict[str, float]] = None
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
    
    # 获取损失权重（健壮兼容：支持 train.loss_weights 与 loss.* 两种结构）
    w_rec = 1.0
    w_spec = 0.0
    w_dc = 0.0
    w_grad = 0.0

    # 优先使用 loss_weights_override 参数
    if loss_weights_override is not None:
        w_rec = loss_weights_override.get('reconstruction', w_rec)
        w_spec = loss_weights_override.get('spectral', w_spec)
        w_dc = loss_weights_override.get('data_consistency', w_dc)
        w_grad = loss_weights_override.get('gradient', w_grad)
    else:
        # 优先使用 train.loss_weights 结构
        has_train_loss_weights = hasattr(config, 'train') and hasattr(config.train, 'loss_weights')
        if has_train_loss_weights:
            try:
                if hasattr(config.train.loss_weights, 'reconstruction'):
                    w_rec = float(config.train.loss_weights.reconstruction)
                if hasattr(config.train.loss_weights, 'spectral'):
                    w_spec = float(config.train.loss_weights.spectral)
                if hasattr(config.train.loss_weights, 'data_consistency'):
                    w_dc = float(config.train.loss_weights.data_consistency)
                # 可选：梯度项
                if hasattr(config.train.loss_weights, 'gradient'):
                    w_grad = float(getattr(config.train.loss_weights, 'gradient', 0.0))
            except Exception:
                # 若读取失败，回退到默认值
                pass

        # 兼容旧版 loss.* 结构
        if hasattr(config, 'loss') and not has_train_loss_weights:
            # reconstruction
            if hasattr(config.loss, 'reconstruction') and hasattr(config.loss.reconstruction, 'weight'):
                try:
                    w_rec = float(config.loss.reconstruction.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, 'reconstruction') and isinstance(config.loss.reconstruction, (int, float)):
                w_rec = float(config.loss.reconstruction)
            # spectral
            if hasattr(config.loss, 'spectral') and hasattr(config.loss.spectral, 'weight'):
                try:
                    w_spec = float(config.loss.spectral.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, 'spectral') and isinstance(config.loss.spectral, (int, float)):
                w_spec = float(config.loss.spectral)
            # data consistency aliases
            if hasattr(config.loss, 'data_consistency') and hasattr(config.loss.data_consistency, 'weight'):
                try:
                    w_dc = float(config.loss.data_consistency.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, 'degradation_consistency') and hasattr(config.loss.degradation_consistency, 'weight'):
                try:
                    w_dc = float(config.loss.degradation_consistency.weight)
                except Exception:
                    pass
            elif hasattr(config.loss, 'data_consistency') and isinstance(config.loss.data_consistency, (int, float)):
                w_dc = float(config.loss.data_consistency)
            elif hasattr(config.loss, 'degradation_consistency') and isinstance(config.loss.degradation_consistency, (int, float)):
                w_dc = float(config.loss.degradation_consistency)
            # 梯度项
            w_grad = getattr(config.loss, 'gradient_weight', w_grad)
    
    losses = {}
    
    # 1. 重建损失（在z-score域计算）
    reconstruction_loss = _compute_reconstruction_loss(pred_z, target_z, obs_data)
    losses['reconstruction_loss'] = reconstruction_loss
    
    # 2. 频谱损失（在原值域计算）
    if w_spec > 0:
        # 获取数据键，支持不同的配置结构
        data_keys = config.data.get('keys', None) if hasattr(config, 'data') else None
        if data_keys is None:
            # 如果没有keys，使用默认的反归一化
            pred_orig = _denormalize_tensor(pred_z, norm_stats, None)
            target_orig = _denormalize_tensor(target_z, norm_stats, None)
        else:
            pred_orig = _denormalize_tensor(pred_z, norm_stats, data_keys)
            target_orig = _denormalize_tensor(target_z, norm_stats, data_keys)
        spectral_loss = _compute_spectral_loss(pred_orig, target_orig, config)
        losses['spectral_loss'] = spectral_loss
    else:
        losses['spectral_loss'] = torch.tensor(0.0, device=device)
    
    # 3. 数据一致性损失（在原值域计算）
    if w_dc > 0:
        # 获取数据键，支持不同的配置结构
        data_keys = config.data.get('keys', None) if hasattr(config, 'data') else None
        if data_keys is None:
            # 如果没有keys，使用默认的反归一化
            pred_orig = _denormalize_tensor(pred_z, norm_stats, None)
            dc_loss = _compute_data_consistency_loss(pred_orig, obs_data, norm_stats, None)
        else:
            pred_orig = _denormalize_tensor(pred_z, norm_stats, data_keys)
            dc_loss = _compute_data_consistency_loss(pred_orig, obs_data, norm_stats, data_keys)
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


def compute_temporal_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    obs_data: Optional[Any],
    norm_stats: Optional[Dict[str, torch.Tensor]],
    config: DictConfig
) -> Dict[str, torch.Tensor]:
    """时序损失（供测试与真实任务使用）

    - 输入为 z-score 域；频谱与 DC 在原值域计算。
    - 支持 4D [B,C,H,W]（单步）与 5D [B,T_out,C,H,W]（多步）。
    - 多步时对各分量按时间维平均，保持与黄金法则一致。
    """
    if pred_z.dim() == 4 and target_z.dim() == 4:
        return compute_total_loss(pred_z, target_z, obs_data or {}, norm_stats, config)

    if pred_z.dim() != 5 or target_z.dim() != 5:
        raise ValueError("compute_temporal_loss expects 4D or 5D tensors")

    device = pred_z.device
    B, T_out, C, H, W = pred_z.shape

    rec_list = []
    spec_list = []
    dc_list = []
    grad_list = []

    for t in range(T_out):
        pred_t = pred_z[:, t]
        target_t = target_z[:, t]

        # 构造步级 obs_data
        obs_t: Dict[str, Any] = {}
        if isinstance(obs_data, dict):
            # 观测序列
            if 'observation' in obs_data:
                obs_val = obs_data['observation']
                obs_t['observation'] = obs_val[:, t] if isinstance(obs_val, torch.Tensor) and obs_val.dim() == 5 else obs_val
            # baseline/mask 可能带时间维
            if 'baseline' in obs_data:
                base_val = obs_data['baseline']
                obs_t['baseline'] = base_val[:, t] if isinstance(base_val, torch.Tensor) and base_val.dim() == 5 else base_val
            if 'mask' in obs_data:
                mask_val = obs_data['mask']
                obs_t['mask'] = mask_val[:, t] if isinstance(mask_val, torch.Tensor) and mask_val.dim() == 5 else mask_val
            # H 参数直接复用
            if 'h_params' in obs_data:
                obs_t['h_params'] = obs_data['h_params']
        elif isinstance(obs_data, torch.Tensor):
            # 直接给出观测序列张量
            if obs_data.dim() == 5:
                obs_t['observation'] = obs_data[:, t]
        else:
            obs_t = {}

        losses_t = compute_total_loss(pred_t, target_t, obs_t, norm_stats, config)
        rec_list.append(losses_t.get('reconstruction_loss', torch.tensor(0.0, device=device)))
        spec_list.append(losses_t.get('spectral_loss', torch.tensor(0.0, device=device)))
        dc_list.append(losses_t.get('dc_loss', torch.tensor(0.0, device=device)))
        grad_list.append(losses_t.get('gradient_loss', torch.tensor(0.0, device=device)))

    def _mean_stack(lst: list[torch.Tensor]) -> torch.Tensor:
        if not lst:
            return torch.tensor(0.0, device=device)
        return torch.stack(lst).mean()

    reconstruction_loss = _mean_stack(rec_list)
    spectral_loss = _mean_stack(spec_list)
    dc_loss = _mean_stack(dc_list)
    gradient_loss = _mean_stack(grad_list)

    # 读取权重，遵循 compute_total_loss 的约定
    w_rec = 1.0
    w_spec = 0.0
    w_dc = 0.0
    w_grad = 0.0

    has_train_loss_weights = hasattr(config, 'train') and hasattr(config.train, 'loss_weights')
    if has_train_loss_weights:
        try:
            if hasattr(config.train.loss_weights, 'reconstruction'):
                w_rec = float(config.train.loss_weights.reconstruction)
            if hasattr(config.train.loss_weights, 'spectral'):
                w_spec = float(config.train.loss_weights.spectral)
            if hasattr(config.train.loss_weights, 'data_consistency'):
                w_dc = float(config.train.loss_weights.data_consistency)
            if hasattr(config.train.loss_weights, 'gradient'):
                w_grad = float(getattr(config.train.loss_weights, 'gradient', 0.0))
        except Exception:
            pass
    elif hasattr(config, 'loss'):
        if hasattr(config.loss, 'reconstruction') and hasattr(config.loss.reconstruction, 'weight'):
            try:
                w_rec = float(config.loss.reconstruction.weight)
            except Exception:
                pass
        elif hasattr(config.loss, 'reconstruction') and isinstance(config.loss.reconstruction, (int, float)):
            w_rec = float(config.loss.reconstruction)
        if hasattr(config.loss, 'spectral') and hasattr(config.loss.spectral, 'weight'):
            try:
                w_spec = float(config.loss.spectral.weight)
            except Exception:
                pass
        elif hasattr(config.loss, 'spectral') and isinstance(config.loss.spectral, (int, float)):
            w_spec = float(config.loss.spectral)
        if hasattr(config.loss, 'data_consistency') and hasattr(config.loss.data_consistency, 'weight'):
            try:
                w_dc = float(config.loss.data_consistency.weight)
            except Exception:
                pass
        elif hasattr(config.loss, 'degradation_consistency') and hasattr(config.loss.degradation_consistency, 'weight'):
            try:
                w_dc = float(config.loss.degradation_consistency.weight)
            except Exception:
                pass
        elif hasattr(config.loss, 'data_consistency') and isinstance(config.loss.data_consistency, (int, float)):
            w_dc = float(config.loss.data_consistency)
        elif hasattr(config.loss, 'degradation_consistency') and isinstance(config.loss.degradation_consistency, (int, float)):
            w_dc = float(config.loss.degradation_consistency)
        w_grad = getattr(config.loss, 'gradient_weight', w_grad)

    total_loss = (
        w_rec * reconstruction_loss +
        w_spec * spectral_loss +
        w_dc * dc_loss +
        w_grad * gradient_loss
    )

    return {
        'reconstruction_loss': reconstruction_loss,
        'spectral_loss': spectral_loss,
        'dc_loss': dc_loss,
        'gradient_loss': gradient_loss,
        'total_loss': total_loss
    }


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
    # 兼容读取：优先 train.spectral_loss.*，其次 loss.*
    low_freq_modes = 16
    use_rfft = False
    normalize = False
    boundary_mode = None

    if hasattr(config, 'train') and hasattr(config.train, 'spectral_loss'):
        low_freq_modes = getattr(config.train.spectral_loss, 'low_freq_modes', low_freq_modes)
        use_rfft = getattr(config.train.spectral_loss, 'use_rfft', use_rfft)
        normalize = getattr(config.train.spectral_loss, 'normalize', normalize)
        boundary_mode = getattr(config.train.spectral_loss, 'boundary_mode', boundary_mode)

    if hasattr(config, 'loss'):
        low_freq_modes = getattr(config.loss, 'low_freq_modes', low_freq_modes)
        use_rfft = getattr(config.loss, 'use_rfft', use_rfft)
        normalize = getattr(config.loss, 'normalize', normalize)
        # 统一边界策略：mirror/zero/none（none表示周期）
        if hasattr(config.loss, 'spectral') and hasattr(config.loss.spectral, 'boundary_mode'):
            boundary_mode = getattr(config.loss.spectral, 'boundary_mode', boundary_mode)
        elif hasattr(config.loss, 'boundary_mode'):
            boundary_mode = getattr(config.loss, 'boundary_mode', boundary_mode)
    boundary_mode = boundary_mode or 'mirror'
    
    B, C, H, W = pred.shape
    
    # 边界处理：镜像延拓 / 零填充 / 无处理（周期）
    if (boundary_mode or 'mirror') == 'mirror':
        pred_extended = _mirror_extend(pred)
        target_extended = _mirror_extend(target)
    elif boundary_mode == 'zero':
        pred_extended = _zero_extend(pred)
        target_extended = _zero_extend(target)
    else:
        # none/periodic: 不进行延拓，直接FFT
        pred_extended = pred
        target_extended = target
    
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
    norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    keys: Optional[list] = None,
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
    print(f"DEBUG: obs_data keys = {list(obs_data.keys())}")
    print(f"DEBUG: pred shape = {pred.shape}")
    
    h_params = obs_data['h_params']
    print(f"DEBUG: In _compute_data_consistency_loss, h_params = {h_params}")
    print(f"DEBUG: h_params type: {type(h_params)}")
    
    # 应用H算子到预测
    h_pred = apply_degradation_operator(pred, h_params)
    
    # 获取对应的观测数据（原值域），兼容多种键名
    observation = obs_data.get('observation')
    if observation is None:
        for alt_key in ['lr_observation', 'degraded', 'observation_lr', 'obs', 'y']:
            if alt_key in obs_data:
                observation = obs_data[alt_key]
                break
    
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
        # 检查维度是否匹配
        if observation.dim() != h_pred.dim():
            print(f"WARNING: observation dim {observation.dim()} != h_pred dim {h_pred.dim()}")
            return torch.tensor(0.0, device=pred.device)
        
        # 检查通道数是否匹配
        if observation.shape[1] != h_pred.shape[1]:
            # 调整通道数
            if observation.shape[1] > h_pred.shape[1]:
                observation = observation[:, :h_pred.shape[1]]
            else:
                print(f"WARNING: observation channels {observation.shape[1]} < h_pred channels {h_pred.shape[1]}")
                return torch.tensor(0.0, device=pred.device)
        
        # 调整空间尺寸
        if observation.shape[-2:] != h_pred.shape[-2:]:
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
    # 差分近似梯度；针对小尺寸（H<2或W<2）避免空张量导致的NaN
    B, C, H, W = pred.shape

    components = []
    if W > 1:
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        components.append(F.l1_loss(pred_grad_x, target_grad_x))
    if H > 1:
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        components.append(F.l1_loss(pred_grad_y, target_grad_y))

    if not components:
        # 单像素或无法计算梯度的情况，返回0
        return torch.tensor(0.0, device=pred.device)

    gradient_loss = sum(components)
    
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
    
    # 回退策略：当未提供keys时，尝试使用通道级 mean/std 或全局 mean/std
    if keys is None:
        # 支持字典形式：{'mean': Tensor[C], 'std': Tensor[C]} 或标量
        if isinstance(norm_stats, dict) and ('mean' in norm_stats and 'std' in norm_stats):
            mean = norm_stats['mean'].to(tensor_z.device)
            std = norm_stats['std'].to(tensor_z.device)
            if mean.dim() == 0:
                mean = mean.repeat(tensor_z.size(1))
            if std.dim() == 0:
                std = std.repeat(tensor_z.size(1))
            tensor_orig = tensor_z * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
            return tensor_orig
        # 支持列表/元组 (mean, std)
        if isinstance(norm_stats, (tuple, list)) and len(norm_stats) == 2:
            mean = norm_stats[0].to(tensor_z.device)
            std = norm_stats[1].to(tensor_z.device)
            if mean.dim() == 0:
                mean = mean.repeat(tensor_z.size(1))
            if std.dim() == 0:
                std = std.repeat(tensor_z.size(1))
            tensor_orig = tensor_z * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
            return tensor_orig
        # 若无法推断，直接返回原张量（保持z-score域）
        return tensor_z

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
            tensor_orig[:, i:i+1] = tensor_z[:, i:i+1] * std.reshape(1, 1, 1, 1) + mean.reshape(1, 1, 1, 1)
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


def _zero_extend(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """零填充延拓张量到更大尺寸
    
    Args:
        x: 输入张量 [B, C, H, W]
        factor: 延拓倍数（默认为2）
    Returns:
        延拓后的张量 [B, C, H*factor, W*factor]
    """
    B, C, H, W = x.shape
    pad_w = W * (factor - 1)
    pad_h = H * (factor - 1)
    # 仅在右侧和下方进行零填充，保持原图在左上角
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    return x_padded


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


def l1_mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算L1 MAE损失
    
    Args:
        x: 预测张量
        y: 目标张量
        
    Returns:
        L1 MAE损失
    """
    return (x - y).abs().mean()


def rel_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算相对L2损失（适用于时序数据）
    
    Args:
        x: 预测张量 [B, T, C, H, W]
        y: 目标张量 [B, T, C, H, W]
        eps: 数值稳定性常数
        
    Returns:
        相对L2损失
    """
    num = torch.sqrt(((x-y)**2).sum(dim=(2,3,4)))  # [B, T]
    den = torch.sqrt((y**2).sum(dim=(2,3,4))) + eps  # [B, T]
    return (num/den).mean()


def compute_ar_loss(
    pred_seq: torch.Tensor, 
    gt_seq: torch.Tensor, 
    cfg_loss: Dict[str, Any]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算自回归模型的损失
    
    Args:
        pred_seq: 预测序列 [B, T_out, C, H, W]
        gt_seq: 真值序列 [B, T_out, C, H, W]
        cfg_loss: 损失配置
        
    Returns:
        总损失和损失项字典
    """
    assert gt_seq is not None, "AR训练需要teacher（target_seq）"
    
    # 获取损失权重
    w_rel2 = cfg_loss.get("rel2_weight", 1.0)
    w_mae = cfg_loss.get("mae_weight", 0.1)
    
    # 计算损失
    rel2_loss = rel_l2(pred_seq, gt_seq)
    mae_loss = l1_mae(pred_seq, gt_seq)
    
    # 总损失
    loss = w_rel2 * rel2_loss + w_mae * mae_loss
    
    # 返回总损失和损失项
    loss_items = {
        "rel2": rel2_loss.item(),
        "mae": mae_loss.item()
    }
    
    return loss, loss_items


def compute_ar_total_loss(
    pred_seq: torch.Tensor, 
    gt_seq: torch.Tensor, 
    obs_data: Dict, 
    norm_stats: Optional[Dict[str, torch.Tensor]], 
    config: DictConfig
) -> Dict[str, torch.Tensor]:
    """计算自回归模型的总损失，包含重建损失、频谱损失和数据一致性损失
    
    Args:
        pred_seq: 预测序列（z-score域）[B, T_out, C, H, W]
        gt_seq: 真值序列（z-score域）[B, T_out, C, H, W]
        obs_data: 观测数据字典
        norm_stats: 归一化统计量
        config: 损失配置
        
    Returns:
        损失字典
    """
    device = pred_seq.device
    B, T, C, H, W = pred_seq.shape
    
    # 获取损失权重
    if hasattr(config.loss, 'rel2_weight'):
        w_rel2 = config.loss.rel2_weight
    else:
        w_rel2 = 1.0
    
    if hasattr(config.loss, 'mae_weight'):
        w_mae = config.loss.mae_weight
    else:
        w_mae = 0.1
    
    # 频谱损失权重
    if hasattr(config, 'loss'):
        if hasattr(config.loss, 'spectral') and hasattr(config.loss.spectral, 'weight'):
            w_spec = float(config.loss.spectral.weight)
        elif hasattr(config.loss, 'spectral') and isinstance(config.loss.spectral, (int, float)):
            w_spec = float(config.loss.spectral)
        else:
            w_spec = 0.0
    else:
        w_spec = 0.0

    # DC损失权重（兼容 data_consistency 与 degradation_consistency 两种键名）
    w_dc = 0.0
    if hasattr(config, 'loss'):
        if hasattr(config.loss, 'data_consistency') and hasattr(config.loss.data_consistency, 'weight'):
            try:
                w_dc = float(config.loss.data_consistency.weight)
            except Exception:
                w_dc = 0.0
        elif hasattr(config.loss, 'degradation_consistency') and hasattr(config.loss.degradation_consistency, 'weight'):
            try:
                w_dc = float(config.loss.degradation_consistency.weight)
            except Exception:
                w_dc = 0.0
        elif hasattr(config.loss, 'data_consistency') and isinstance(config.loss.data_consistency, (int, float)):
            w_dc = float(config.loss.data_consistency)
        elif hasattr(config.loss, 'degradation_consistency') and isinstance(config.loss.degradation_consistency, (int, float)):
            w_dc = float(config.loss.degradation_consistency)
    
    losses = {}
    
    # 1. 重建损失（在z-score域计算）
    rel2_loss = rel_l2(pred_seq, gt_seq)
    mae_loss = l1_mae(pred_seq, gt_seq)
    reconstruction_loss = w_rel2 * rel2_loss + w_mae * mae_loss
    losses['reconstruction_loss'] = reconstruction_loss
    losses['rel2_loss'] = rel2_loss
    losses['mae_loss'] = mae_loss
    
    # 2. 频谱损失（在原值域计算）- 可选
    if w_spec > 0:
        # 将序列转换为批次处理
        pred_flat = pred_seq.reshape(B*T, C, H, W).contiguous()
        gt_flat = gt_seq.reshape(B*T, C, H, W).contiguous()

        # 安全解析 keys：若未提供或类型异常则置为 None，启用通道级 mean/std
        data_keys = None
        try:
            if hasattr(config, 'data'):
                maybe_keys = getattr(config.data, 'keys', None)
                # 仅当 keys 是通道名（纯字母字符串）时接受；否则视为误配置
                if isinstance(maybe_keys, (list, tuple)) and all(isinstance(k, str) and k.isalpha() for k in maybe_keys):
                    data_keys = list(maybe_keys)
        except Exception:
            data_keys = None

        # 反归一化到原值域
        pred_orig = _denormalize_tensor(pred_flat, norm_stats, data_keys)
        target_orig = _denormalize_tensor(gt_flat, norm_stats, data_keys)

        # 计算频谱损失
        spectral_loss = _compute_spectral_loss(pred_orig, target_orig, config)
        losses['spectral_loss'] = spectral_loss
    else:
        losses['spectral_loss'] = torch.tensor(0.0, device=device)
    
    # 3. 数据一致性损失（在原值域计算）- 可选
    if w_dc > 0 and 'observation_seq' in obs_data:
        # 将序列转换为批次处理 - 添加安全检查确保维度匹配
        try:
            pred_flat = pred_seq.reshape(B*T, C, H, W).contiguous()
        except RuntimeError as e:
            # 如果reshape失败，检查实际维度并调整
            actual_elements = pred_seq.numel()
            expected_elements = B * T * C * H * W
            if actual_elements != expected_elements:
                print(f"WARNING: Dimension mismatch in pred_seq reshape: expected {expected_elements}, got {actual_elements}")
                print(f"pred_seq shape: {pred_seq.shape}, expected: [{B}, {T}, {C}, {H}, {W}]")
                # 尝试修复维度 - 可能是T_out维度不匹配
                if pred_seq.dim() == 4:
                    # FNO2D输出是4D [B, C, H, W]，需要扩展到5D
                    pred_seq = pred_seq.unsqueeze(1).expand(-1, T, -1, -1, -1)
                elif pred_seq.shape[1] != T:
                    # T维度不匹配，调整T值
                    actual_T = pred_seq.shape[1]
                    T = actual_T
                    pred_flat = pred_seq.reshape(B*T, C, H, W).contiguous()
                else:
                    raise e
            else:
                raise e

        # 安全解析 keys：若未提供或类型异常则置为 None，启用通道级 mean/std
        data_keys = None
        try:
            if hasattr(config, 'data'):
                maybe_keys = getattr(config.data, 'keys', None)
                if isinstance(maybe_keys, (list, tuple)) and all(isinstance(k, str) and k.isalpha() for k in maybe_keys):
                    data_keys = list(maybe_keys)
        except Exception:
            data_keys = None

        # 反归一化到原值域
        pred_orig = _denormalize_tensor(pred_flat, norm_stats, data_keys)

        # 准备观测数据
        obs_data_flat = {}
        for key, value in obs_data.items():
            if key.endswith('_seq') and isinstance(value, torch.Tensor):
                # 处理序列数据 - 添加安全检查
                try:
                    obs_data_flat[key.replace('_seq', '')] = value.reshape(B*T, *value.shape[2:]).contiguous()
                except RuntimeError:
                    # 如果观测数据也有维度问题，跳过或调整
                    print(f"WARNING: Skipping reshape for obs_data key {key}, shape {value.shape}")
                    continue
            else:
                obs_data_flat[key] = value

        # 计算DC损失
        dc_loss = _compute_data_consistency_loss(pred_orig, obs_data_flat, norm_stats, data_keys)
        losses['dc_loss'] = dc_loss
    else:
        losses['dc_loss'] = torch.tensor(0.0, device=device)
    
    # 4. 总损失
    total_loss = (
        losses['reconstruction_loss'] +
        w_spec * losses['spectral_loss'] +
        w_dc * losses['dc_loss']
    )
    losses['total_loss'] = total_loss
    
    return losses
