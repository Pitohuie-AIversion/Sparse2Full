"""Swin时序包装器

将时序功能接入SwinUNet，保持主干架构不变。
支持单时序模块和双头(AR+NAR)架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import logging

from ..temporal_block import create_temporal_module
from ..decoder.query_head import create_query_head
from ..swin_unet import SwinUNet
from ..ar.wrapper import ARWrapper

logger = logging.getLogger(__name__)


class SwinTemporal(nn.Module):
    """Swin时序模块
    
    在SwinUNet前端添加时序处理，聚合历史信息。
    保持SwinUNet接口不变：forward(B,C,H,W) -> (B,C_out,H,W)
    
    Args:
        base_kwargs: SwinUNet基础参数
        temporal_cfg: 时序模块配置
    """
    
    def __init__(
        self,
        base_kwargs: Dict[str, Any],
        temporal_cfg: Dict[str, Any]
    ):
        super().__init__()
        
        self.temporal_cfg = temporal_cfg
        
        # 创建SwinUNet主干（保持不变）
        self.backbone = SwinUNet(**base_kwargs)
        
        # 继承基础模型属性
        self.in_channels = base_kwargs.get('in_channels', 1)
        self.out_channels = base_kwargs.get('out_channels', 1)
        self.img_size = base_kwargs.get('img_size', 256)
        
        # 创建时序模块
        self.temporal = None
        if temporal_cfg.get('enabled', False):
            self.temporal = create_temporal_module(
                temporal_type=temporal_cfg.get('type', 'conv1d'),
                c_in=self.in_channels,
                c_out=temporal_cfg.get('c_out', self.in_channels),
                k=temporal_cfg.get('k', 3),
                causal=temporal_cfg.get('causal', True),
                dropout=temporal_cfg.get('dropout', 0.0)
            )
            
            # 更新输入通道数（如果时序模块改变了通道数）
            if hasattr(self.temporal, 'get_output_channels'):
                temporal_out_channels = self.temporal.get_output_channels()
                if temporal_out_channels != self.in_channels:
                    # 需要调整backbone的输入通道数
                    logger.warning(f"Temporal module changes channels: {self.in_channels} -> {temporal_out_channels}")
        
        logger.info(f"SwinTemporal: temporal_enabled={temporal_cfg.get('enabled', False)}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，支持 (B,C,H,W) 或 (B,T,C,H,W)
            return_features: 是否返回中间特征
            
        Returns:
            输出张量 (B,C_out,H,W) 或 (output, features)
        """
        # 处理输入维度
        if x.dim() == 5:  # (B,T,C,H,W)
            if self.temporal is not None:
                # 使用时序模块聚合
                x = self.temporal(x)  # (B,C,H,W)
            else:
                # 没有时序模块，使用最后一帧
                x = x[:, -1]  # (B,C,H,W)
        elif x.dim() == 4:  # (B,C,H,W)
            # 单帧输入，直接处理
            pass
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # SwinUNet主干处理
        if return_features:
            # 如果需要返回特征，需要修改backbone
            output = self.backbone(x)
            # 简单起见，这里返回patch_embed特征作为memory
            features = self._extract_features(x)
            return output, features
        else:
            output = self.backbone(x)
            return output
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取中间特征作为memory
        
        Args:
            x: 输入张量 (B,C,H,W)
            
        Returns:
            特征张量 (B,D,H,W) - 保持原始空间分辨率
        """
        # 使用patch_embed输出作为特征
        features = self.backbone.patch_embed(x)  # (B, N, D)
        B, N, D = features.shape
        
        # 计算patch grid尺寸
        patch_size = self.backbone.patch_embed.patch_size[0]
        H_patches = W_patches = int(N ** 0.5)
        
        # 重塑为特征图
        features = features.transpose(1, 2).view(B, D, H_patches, W_patches)
        
        # 上采样到原始分辨率
        H_orig, W_orig = x.shape[-2:]
        if features.shape[-2:] != (H_orig, W_orig):
            features = F.interpolate(
                features,
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )
        
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        base_info = self.backbone.get_model_info() if hasattr(self.backbone, 'get_model_info') else {}
        
        temporal_info = {}
        if self.temporal is not None:
            temporal_info = self.temporal.get_model_info()
        
        return {
            'model_type': 'SwinTemporal',
            'backbone': base_info,
            'temporal': temporal_info,
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }


class SwinTemporalNAR(nn.Module):
    """Swin时序NAR双头模块
    
    结合时序处理、AR预测和NAR预测的完整架构。
    支持双头并行训练和单头推理。
    
    Args:
        base_kwargs: SwinUNet基础参数
        temporal_cfg: 时序模块配置
        nar_cfg: NAR头配置
        ar_cfg: AR配置（可选）
        use_ar: 是否启用AR头
        use_nar: 是否启用NAR头
    """
    
    def __init__(
        self,
        base_kwargs: Dict[str, Any],
        temporal_cfg: Dict[str, Any],
        nar_cfg: Dict[str, Any],
        ar_cfg: Optional[Dict[str, Any]] = None,
        use_ar: bool = True,
        use_nar: bool = True
    ):
        super().__init__()
        
        if not (use_ar or use_nar):
            raise ValueError("At least one of AR or NAR must be enabled")
        
        self.use_ar = use_ar
        self.use_nar = use_nar
        self.temporal_cfg = temporal_cfg
        self.nar_cfg = nar_cfg
        self.ar_cfg = ar_cfg or {}
        
        # 继承基础模型属性
        self.in_channels = base_kwargs.get('in_channels', 1)
        self.out_channels = base_kwargs.get('out_channels', 1)
        self.img_size = base_kwargs.get('img_size', 256)
        
        # 创建时序增强的SwinUNet
        self.swin_temporal = SwinTemporal(base_kwargs, temporal_cfg)
        
        # 创建AR包装器（如果启用）
        self.ar_wrapper = None
        if use_ar:
            # 使用现有的ARWrapper包装SwinTemporal
            self.ar_wrapper = ARWrapper(
                single_frame_model=self.swin_temporal,
                detach_rollout=self.ar_cfg.get('detach_rollout', True),
                scheduled_sampling=self.ar_cfg.get('scheduled_sampling', False),
                sampling_schedule=self.ar_cfg.get('sampling_schedule', None)
            )
        
        # 创建NAR查询头（如果启用）
        self.nar_head = None
        if use_nar:
            # 获取特征维度（对齐SwinUNet的embed_dim）
            d_model = nar_cfg.get('d_model', getattr(self.swin_temporal.backbone, 'embed_dim', 96))
            
            # 根据head_type决定传递的参数
            head_kwargs = {
                'd_model': d_model,
                'c_out': self.out_channels,
                'max_timesteps': nar_cfg.get('max_timesteps', 32),
                'dropout': nar_cfg.get('dropout', 0.1)
            }
            
            # 只有cross_attention类型才需要num_heads参数
            if nar_cfg.get('head_type', 'simple') == 'cross_attention':
                head_kwargs['num_heads'] = nar_cfg.get('num_heads', 8)
            
            self.nar_head = create_query_head(
                head_type=nar_cfg.get('head_type', 'simple'),
                **head_kwargs
            )
        
        logger.info(f"SwinTemporalNAR: AR={use_ar}, NAR={use_nar}")
    
    def forward(
        self,
        x_seq: torch.Tensor,
        T_out: int = 1,
        teacher_seq: Optional[torch.Tensor] = None,
        train_mode: Optional[bool] = None,
        return_both: bool = True
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """前向传播
        
        Args:
            x_seq: 输入序列 (B,T_in,C,H,W) 或 (B,C,H,W)
            T_out: 输出时间步数
            teacher_seq: 教师信号 (B,T_out,C,H,W)，训练时使用
            train_mode: 训练模式，None时使用self.training
            return_both: 是否返回双头结果
            
        Returns:
            如果return_both=True: (ar_output, nar_output)
            否则返回主要输出（优先NAR）
        """
        if train_mode is None:
            train_mode = self.training
        
        ar_output = None
        nar_output = None
        
        # AR预测
        if self.use_ar and self.ar_wrapper is not None:
            ar_output = self.ar_wrapper(
                x_in=x_seq,
                T_out=T_out,
                teacher=teacher_seq,
                train_mode=train_mode
            )
        
        # NAR预测
        if self.use_nar and self.nar_head is not None:
            # 1. 时序聚合得到单帧
            if x_seq.dim() == 5:  # (B,T_in,C,H,W)
                if self.swin_temporal.temporal is not None:
                    x_single = self.swin_temporal.temporal(x_seq)
                else:
                    x_single = x_seq[:, -1] if x_seq.size(1) > 0 else x_seq.squeeze(1)
            else:  # (B,C,H,W)
                x_single = x_seq
            
            # 2. 提取memory特征
            memory = self.swin_temporal._extract_features(x_single)
            
            # 3. NAR多步预测
            nar_output = self.nar_head(memory, T_out)
        
        # 返回结果
        if return_both:
            return ar_output, nar_output
        else:
            # 优先返回NAR，其次AR
            return nar_output if nar_output is not None else ar_output
    
    def set_epoch(self, epoch: int, total_epochs: int = None):
        """设置训练epoch（用于scheduled sampling）"""
        if self.ar_wrapper is not None:
            self.ar_wrapper.set_epoch(epoch, total_epochs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': 'SwinTemporalNAR',
            'use_ar': self.use_ar,
            'use_nar': self.use_nar,
            'temporal_config': self.temporal_cfg,
            'nar_config': self.nar_cfg,
            'ar_config': self.ar_cfg,
        }
        
        # 添加子模块信息
        if hasattr(self.swin_temporal, 'get_model_info'):
            info['swin_temporal'] = self.swin_temporal.get_model_info()
        
        if self.ar_wrapper is not None and hasattr(self.ar_wrapper, 'get_model_info'):
            info['ar_wrapper'] = self.ar_wrapper.get_model_info()
        
        if self.nar_head is not None and hasattr(self.nar_head, 'get_model_info'):
            info['nar_head'] = self.nar_head.get_model_info()
        
        # 参数统计
        info['total_parameters'] = sum(p.numel() for p in self.parameters())
        
        return info
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> Dict[str, int]:
        """计算FLOPs"""
        flops = {}
        
        if self.ar_wrapper is not None:
            flops['ar'] = self.ar_wrapper.compute_flops(input_shape)
        
        if self.nar_head is not None:
            # NAR的FLOPs主要来自查询头
            # 简单估计：基于参数量
            nar_params = sum(p.numel() for p in self.nar_head.parameters())
            if input_shape is not None:
                B, C, H, W = input_shape[-4:]
                flops['nar'] = nar_params * B * H * W
            else:
                flops['nar'] = nar_params * 256 * 256  # 默认估计
        
        flops['total'] = sum(flops.values())
        return flops
    
    def get_memory_usage(self, batch_size: int = 1, T_out: int = 3) -> Dict[str, float]:
        """估算显存使用量"""
        memory_info = {}
        
        # 基础模型显存
        base_memory = sum(p.numel() * p.element_size() for p in self.swin_temporal.parameters()) / 1024**2
        memory_info['base_MB'] = base_memory
        
        # AR显存（如果启用）
        if self.ar_wrapper is not None:
            # AR需要存储中间状态
            ar_memory = base_memory * T_out * 0.5  # 估计
            memory_info['ar_MB'] = ar_memory
        
        # NAR显存（如果启用）
        if self.nar_head is not None:
            # NAR并行计算，显存相对固定
            nar_params = sum(p.numel() * p.element_size() for p in self.nar_head.parameters()) / 1024**2
            memory_info['nar_MB'] = nar_params
        
        memory_info['total_MB'] = sum(v for k, v in memory_info.items() if k.endswith('_MB'))
        return memory_info


# 导出接口
__all__ = [
    'SwinTemporal',
    'SwinTemporalNAR'
]