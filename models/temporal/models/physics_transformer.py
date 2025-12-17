"""
物理感知Transformer时序模型
专门为PDE求解设计的完整时序预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from abc import ABC, abstractmethod

from models.temporal.base_temporal import BaseTemporalModel
from models.temporal.components.multi_scale_attn import (
    MultiScaleTemporalAttention, AdaptiveTemporalMixer, PhysicsAwareAttention
)
from models.temporal.components.physics_constraints import PhysicsConstraints, CausalConv1d, PhysicsConsistencyChecker


class PhysicsTransformerTemporal(BaseTemporalModel):
    """物理感知Transformer时序模型
    
    专门为PDE求解设计的Transformer架构，特点：
    1. 物理信息位置编码
    2. 多尺度时序注意力
    3. 物理约束机制
    4. 因果性保证
    5. 与空间模型解耦
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数  
        img_size: 图像尺寸（H, W）
        T_in: 输入时间步数
        T_out: 输出时间步数
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
        num_layers: Transformer层数
        pde_type: PDE类型 ('heat', 'wave', 'navier_stokes', 'reaction_diffusion')
        constraint_weights: 物理约束权重
        use_frequency_encoding: 是否使用频率编码
        mode: 预测模式 ('ar' for autoregressive, 'single' for single-step)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[int, Tuple[int, int]],
        T_in: int = 1,
        T_out: int = 1,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        pde_type: str = 'heat',
        constraint_weights: Optional[Dict[str, float]] = None,
        use_frequency_encoding: bool = True,
        dropout: float = 0.1,
        mode: str = 'ar',
        physics_weight: float = 0.1,
        causal_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, T_in, T_out, mode)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pde_type = pde_type
        self.use_frequency_encoding = use_frequency_encoding
        self.physics_weight = physics_weight
        self.causal_weight = causal_weight
        
        # 处理img_size参数
        if isinstance(img_size, int):
            self.img_height = self.img_width = img_size
        else:
            self.img_height, self.img_width = img_size
        
        # 计算空间维度
        self.spatial_dim = self.img_height * self.img_width
        
        # 输入投影（从输入通道到隐藏维度）
        # 注意：输入是 [B, T, C*H*W]，我们需要投影到隐藏维度
        self.input_projection = nn.Linear(in_channels * self.spatial_dim, hidden_dim)
        
        # 物理信息位置编码
        self.physics_pos_encoding = PhysicsPositionalEncoding(
            hidden_dim, max_len=1024, pde_type=pde_type
        )
        
        # 频率编码
        if use_frequency_encoding:
            self.frequency_encoding = FrequencyPositionalEncoding(hidden_dim)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            PhysicsTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                pde_type=pde_type,
                constraint_weights=constraint_weights,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影（从隐藏维度到输出通道*空间维度）
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels * self.spatial_dim)
        )
        
        # 物理一致性检查器
        self.physics_checker = PhysicsConsistencyChecker()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def encode_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """编码时序特征"""
        B, T, C = x.shape
        
        # 输入已经是 [B, T, C*H*W] 格式，直接投影
        x_reshaped = x.reshape(B * T, -1)  # [B*T, C*H*W]
        
        # 输入投影
        x_proj = self.input_projection(x_reshaped)  # [B*T, hidden_dim]
        x_proj = x_proj.reshape(B, T, -1)  # [B, T, hidden_dim]
        
        # 添加位置编码
        x_pos = self.physics_pos_encoding(x_proj)
        
        # 添加频率编码
        if self.use_frequency_encoding:
            x_freq = self.frequency_encoding(x_proj)
            x_encoded = x_pos + x_freq
        else:
            x_encoded = x_pos
        
        return x_encoded
    
    def decode_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """解码时序特征"""
        B, T, C = x.shape
        
        # 重塑输入以匹配输出投影
        x_reshaped = x.reshape(B * T, C)  # [B*T, hidden_dim]
        
        # 输出投影
        x_proj = self.output_projection(x_reshaped)  # [B*T, out_channels*H*W]
        x_proj = x_proj.reshape(B, T, -1)  # [B, T, out_channels*H*W]
        
        return x_proj
    
    def forward_single_step(self, x: torch.Tensor, 
                           physical_info: Optional[Dict] = None) -> torch.Tensor:
        """单步前向传播"""
        # 编码输入
        x_encoded = self.encode_temporal_features(x)
        
        # 通过Transformer层
        hidden_states = x_encoded
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, physical_info)
        
        # 层归一化
        normalized_features = self.layer_norm(hidden_states)
        
        # 解码输出
        output = self.decode_temporal_features(normalized_features)
        
        return output
    
    def forward_autoregressive(self, x: torch.Tensor, T_out: int,
                              physical_info: Optional[Dict] = None,
                              teacher_forcing: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自回归前向传播"""
        B, T_in, C = x.shape
        
        # 初始化输出序列
        outputs = []
        current_input = x
        
        for t in range(T_out):
            # 单步预测
            step_output = self.forward_single_step(current_input, physical_info)
            
            # 取最后一个时间步作为当前输出
            current_output = step_output[:, -1:]  # [B, 1, C]
            outputs.append(current_output)
            
            # 更新输入序列
            if teacher_forcing is not None and t < teacher_forcing.size(1):
                # 使用教师强制
                next_input = teacher_forcing[:, t:t+1]
            else:
                # 使用自回归输出
                next_input = current_output
            
            # 滑动窗口更新
            current_input = torch.cat([current_input[:, 1:], next_input], dim=1)
        
        # 合并所有输出
        output_sequence = torch.cat(outputs, dim=1)  # [B, T_out, C]
        
        return output_sequence
    
    def forward(self, x: torch.Tensor, T_out: Optional[int] = None,
                teacher_forcing: Optional[torch.Tensor] = None,
                physical_info: Optional[Dict] = None,
                return_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入序列 [B, T_in, C, H, W] 或 [B, T_in, C] 或 [B, T_in, H, W]
            T_out: 输出时间步数（自回归模式）
            teacher_forcing: 教师强制信号
            physical_info: 物理信息字典
            return_dict: 是否返回字典格式
            
        Returns:
            预测结果或包含详细信息的字典
        """
        # 保存原始形状信息
        original_shape = x.shape
        
        # 处理输入形状
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            # 重塑为 [B, T, C*H*W] 以便时序处理
            x = x.reshape(B, T, -1)
            self.original_channels = C
            self.original_height = H
            self.original_width = W
        elif x.dim() == 4:  # [B, T, H, W]
            B, T, H, W = x.shape
            x = x.reshape(B, T, -1)  # [B, T, H*W]
            self.original_channels = 1
            self.original_height = H
            self.original_width = W
        elif x.dim() == 3:  # [B, T, C]
            B, T, C = x.shape
            self.original_channels = C
            self.original_height = int(np.sqrt(C))
            self.original_width = int(np.sqrt(C))
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
        
        # 根据模式选择前向传播方式
        if self.mode == 'nar':  # 非自回归单步模式
            output = self.forward_single_step(x, physical_info)
        else:  # autoregressive
            T_out = T_out or self.T_out
            output = self.forward_autoregressive(x, T_out, physical_info, teacher_forcing)
        
        # 恢复原始形状 [B, T, C, H, W]
        if hasattr(self, 'original_channels') and hasattr(self, 'original_height') and hasattr(self, 'original_width'):
            output = output.reshape(
                output.size(0), output.size(1), 
                self.original_channels, self.original_height, self.original_width
            )
        
        if return_dict:
            # 物理一致性检查
            physics_check = self.physics_checker.comprehensive_check(output)
            
            return {
                'prediction': output,
                'physics_valid': physics_check,
                'model_type': 'physics_transformer',
                'pde_type': self.pde_type
            }
        
        return output


class PhysicsPositionalEncoding(nn.Module):
    """物理信息位置编码"""
    
    def __init__(self, hidden_dim: int, max_len: int = 1024, pde_type: str = 'heat'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pde_type = pde_type
        
        # 标准正弦位置编码
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # PDE特定的物理编码
        if pde_type == 'heat':
            # 热扩散的时间衰减特性
            diffusion_coeff = 0.1
            time_decay = torch.exp(-diffusion_coeff * torch.arange(max_len, dtype=torch.float) / 100.0)
            pe *= time_decay.unsqueeze(1)
        elif pde_type == 'wave':
            # 波动的周期性特性
            wave_freq = 0.1
            time_positions = torch.arange(max_len, dtype=torch.float)
            wave_modulation = torch.sin(wave_freq * time_positions)
            pe *= (1 + 0.1 * wave_modulation).unsqueeze(1)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FrequencyPositionalEncoding(nn.Module):
    """频率位置编码"""
    
    def __init__(self, hidden_dim: int, max_freq: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_freq = max_freq
        
        # 频率基函数
        self.freq_bases = nn.Parameter(torch.randn(max_freq, hidden_dim) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # 时间频率编码
        time_freqs = torch.arange(T, device=x.device).float() / T
        freq_encoding = torch.zeros(B, T, C, device=x.device)
        
        for i in range(self.max_freq):
            freq_component = torch.sin(2 * np.pi * time_freqs * (i + 1))
            freq_encoding += freq_component.unsqueeze(0).unsqueeze(-1) * self.freq_bases[i]
        
        return freq_encoding


class PhysicsTransformerLayer(nn.Module):
    """物理感知Transformer层"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pde_type: str = 'heat',
        constraint_weights: Optional[Dict[str, float]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 物理感知注意力
        self.physics_attention = PhysicsAwareAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            pde_constraint_weight=constraint_weights.get('pde_residual', 0.1) 
                                if constraint_weights else 0.1
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, physical_info: Optional[Dict] = None) -> torch.Tensor:
        """前向传播"""
        # 物理感知注意力
        attn_out, constraint_losses = self.physics_attention(x, physical_info)
        
        # 残差连接和归一化
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        
        # 残差连接和归一化
        x = self.norm2(x + self.dropout(ff_out))
        
        # 存储约束损失（用于训练时的总损失计算）
        if hasattr(self, 'constraint_losses'):
            self.constraint_losses = constraint_losses
        
        return x


# 辅助函数
import math
import numpy as np