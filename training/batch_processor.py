"""张量批次处理器模块 (Batch Processor)

负责从 DataLoader 批次数据中构建统一的模型输入与目标真值张量，
并进行严格的通道对齐校验与物理场空间尺寸 Fail-Fast 守恒拦截。
"""

import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class BatchProcessor:
    """训练批次张量构建与物理守恒校验处理器"""

    def __init__(self, config: Optional[DictConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._logged_channel_warning = False

    def build_model_input(
        self, 
        batch: Dict[str, torch.Tensor], 
        model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """统一构建模型输入：[baseline, coords?, mask?]
        
        - 支持 `use_lowres_input`：直接提取未经插值的低分辨率原始物理场 (如 32x32)
        - 否则默认处理时序 baseline（取最后一个时间步或添加 batch 维）并追加 `coords` 与 `mask` 通道
        - 依据模型的 `in_channels` 对最终通道进行裁剪/填充对齐
        """
        # 1. 检查是否启用直接低维输入 (兼容 DataParallel module)
        model_unwrapped = getattr(model, 'module', model) if model is not None else None
        use_lowres = False
        if model_unwrapped is not None:
            use_lowres = getattr(model_unwrapped, 'use_lowres_input', False)
        
        if not use_lowres and self.config is not None:
            model_cfg = getattr(self.config, 'model', None)
            if model_cfg is not None and hasattr(model_cfg, 'params'):
                use_lowres = getattr(model_cfg.params, 'use_lowres_input', False)

        # 2. 提取输入张量
        if use_lowres:
            raw_obs = batch.get('lr_observation', batch.get('original_observation', None))
            if raw_obs is not None:
                if raw_obs.dim() == 5:
                    raw_obs = raw_obs[:, -1]
                elif raw_obs.dim() == 3:
                    raw_obs = raw_obs.unsqueeze(0)
                self.logger.debug(f"Using direct low-resolution input with shape: {raw_obs.shape}")
                model_input = raw_obs
            else:
                model_input = batch.get('baseline', batch.get('observation', batch.get('target', None)))
                if model_input is None:
                    raise KeyError(f"Batch missing direct observation, baseline or target. Available keys: {list(batch.keys())}")
        else:
            baseline = batch.get('baseline', batch.get('observation', batch.get('target', None)))
            if baseline is None:
                raise KeyError(f"Batch missing baseline, observation or target tensors. Available keys: {list(batch.keys())}")

            # 处理时序维度
            if baseline.dim() == 5:  # [B, T, C, H, W]
                self.logger.debug(f"baseline shape before time select: {baseline.shape}")
                baseline = baseline[:, -1]  # 取最后一个时间步 [B, C, H, W]
                self.logger.debug(f"baseline shape after time select: {baseline.shape}")
            elif baseline.dim() == 3:  # [C, H, W] 或 [T_in*C, H, W]
                self.logger.debug(f"baseline shape before add batch: {baseline.shape}")
                baseline = baseline.unsqueeze(0)  # [1, C, H, W]
                self.logger.debug(f"baseline shape after add batch: {baseline.shape}")
            else:
                self.logger.debug(f"baseline shape: {baseline.shape}")

            model_input = baseline

            # 追加物理坐标通道
            if 'coords' in batch:
                coords = batch['coords']
                self.logger.debug(f"coords shape: {coords.shape}")
                model_input = torch.cat([model_input, coords], dim=1)
                self.logger.debug(f"model_input after coords: {model_input.shape}")

            # 追加掩码通道
            if 'mask' in batch:
                mask = batch['mask']
                self.logger.debug(f"mask shape: {mask.shape}")
                model_input = torch.cat([model_input, mask], dim=1)
                self.logger.debug(f"model_input after mask: {model_input.shape}")

        # 3. 获取模型实际期望的总输入通道数 (兼容 DataParallel)
        expected_in = None
        if model_unwrapped is not None:
            expected_in = getattr(model_unwrapped, 'in_channels', None)
        
        if expected_in is None and self.config is not None:
            model_cfg = getattr(self.config, 'model', None)
            if model_cfg is not None:
                expected_in = getattr(model_cfg, 'in_channels', None)
                if expected_in is None and hasattr(model_cfg, 'params'):
                    expected_in = getattr(model_cfg.params, 'in_channels', None)

        # 4. 通道对齐与严格防御
        if expected_in is not None:
            expected_in = int(expected_in)
            if model_input.shape[1] != expected_in:
                tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', None)) if self.config else None
                allow_ch_mismatch = getattr(tr_cfg, 'allow_channel_mismatch', True) if tr_cfg else True
                
                if not allow_ch_mismatch:
                    raise ValueError(
                        f"Model input channel mismatch! Actual input has {model_input.shape[1]} channels, "
                        f"but model expects {expected_in} channels. "
                        "Check your observation mode, coords and mask configuration."
                    )
                if model_input.shape[1] > expected_in:
                    if not self._logged_channel_warning:
                        self.logger.warning(
                            f"model_input channels ({model_input.shape[1]}) trimmed to expected {expected_in} (will suppress subsequent warnings)"
                        )
                        self._logged_channel_warning = True
                    model_input = model_input[:, :expected_in]
                else:
                    pad_ch = expected_in - model_input.shape[1]
                    if not self._logged_channel_warning:
                        self.logger.warning(
                            f"model_input channels ({model_input.shape[1]}) padded with {pad_ch} zeros to reach expected {expected_in} (will suppress subsequent warnings)"
                        )
                        self._logged_channel_warning = True
                    pad = torch.zeros(
                        model_input.shape[0], pad_ch, model_input.shape[2], model_input.shape[3],
                        device=model_input.device, dtype=model_input.dtype
                    )
                    model_input = torch.cat([model_input, pad], dim=1)

        return model_input

    def prepare_target(
        self, 
        target: torch.Tensor, 
        pred_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """统一处理目标物理场：时序选择、通道校验与空间尺寸 Fail-Fast 校验"""
        # 1. 处理时序数据
        if target.dim() == 5:  # [B, T, C, H, W]
            target = target[:, -1]  # 取最后一个时间步 [B, C, H, W]
            self.logger.debug(f"target shape after time select: {target.shape}")

        # 2. 通道对齐与校验
        if target.shape[1] != pred_shape[1]:
            tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', None)) if self.config else None
            allow_ch_mismatch = getattr(tr_cfg, 'allow_channel_mismatch', False) if tr_cfg else False
            
            if not allow_ch_mismatch and target.shape[1] < pred_shape[1]:
                raise ValueError(
                    f"Target channel mismatch! Target has {target.shape[1]} channels, but prediction has {pred_shape[1]} channels."
                )
            self.logger.warning(
                f"target channels ({target.shape[1]}) trimmed to match prediction channels ({pred_shape[1]})"
            )
            target = target[:, :pred_shape[1]]

        # 3. 空间尺寸对齐与物理完整性校验 (Fail-Fast: 严禁静默重采样 Ground Truth)
        if target.shape[-2:] != pred_shape[-2:]:
            tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', None)) if self.config else None
            allow_resample = getattr(tr_cfg, 'allow_target_resampling', False) if tr_cfg else False
            
            if not allow_resample:
                raise ValueError(
                    f"Physical field spatial dimension mismatch! Target shape {tuple(target.shape[-2:])} != Prediction shape {tuple(pred_shape[-2:])}. "
                    "Silently interpolating ground-truth violates physical conservation laws and corrupts evaluation metrics. "
                    "If intentional, set training.allow_target_resampling=true in your config."
                )
            target = F.interpolate(target, size=pred_shape[-2:], mode='bilinear', align_corners=False)
            self.logger.warning(
                f"Target physical field resized from {target.shape[-2:]} to {pred_shape[-2:]} via bilinear interpolation."
            )

        return target
