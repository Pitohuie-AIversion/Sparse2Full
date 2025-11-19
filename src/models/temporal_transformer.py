"""
时间编码Transformer
基于技术方案实现时间编码Transformer，支持多种时间编码方式和Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TemporalEncodingConfig:
    """时间编码配置"""
    encoding_type: str = "sincos"  # "sincos", "learnable", "relative", "rope"
    max_sequence_length: int = 1000
    embedding_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    learnable_positional: bool = True
    time_scale: float = 10000.0
    base_frequency: float = 1.0
    normalize_time: bool = True
    time_unit: str = "second"  # "second", "minute", "hour", "day"

class SinusoidalTemporalEncoding(nn.Module):
    """
    正弦时间编码
    基于Transformer原始论文的时间编码实现
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.max_length = config.max_sequence_length
        self.dropout = nn.Dropout(config.dropout)
        
        # 预计算正弦编码
        self.register_buffer('temporal_encoding', self._get_sinusoidal_encoding())
        logger.info(f"正弦时间编码初始化完成，维度: {self.embedding_dim}")
    
    def _get_sinusoidal_encoding(self) -> torch.Tensor:
        """生成正弦时间编码"""
        encoding = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        
        # 计算频率
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * 
                           (-math.log(self.config.time_scale) / self.embedding_dim))
        
        # 正弦和余弦编码
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        return encoding.unsqueeze(0)  # 添加批次维度
    
    def forward(self, time_steps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time_steps: 时间步长张量 [batch_size, seq_len]
            mask: 可选的掩码
            
        Returns:
            时间编码 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = time_steps.shape
        
        # 获取时间编码
        temporal_enc = self.temporal_encoding[:, :seq_len, :]
        
        # 扩展为批次大小
        if batch_size > 1:
            temporal_enc = temporal_enc.expand(batch_size, -1, -1)
        
        # 应用掩码
        if mask is not None:
            temporal_enc = temporal_enc.masked_fill(mask.unsqueeze(-1) == 0, 0)
        
        return self.dropout(temporal_enc)

class LearnableTemporalEncoding(nn.Module):
    """
    可学习时间编码
    可学习的时间嵌入向量
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.max_length = config.max_sequence_length
        self.dropout = nn.Dropout(config.dropout)
        
        # 可学习的时间嵌入
        self.time_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # 初始化权重
        self._init_weights()
        logger.info(f"可学习时间编码初始化完成，维度: {self.embedding_dim}")
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, time_steps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time_steps: 时间步长张量 [batch_size, seq_len]
            mask: 可选的掩码
            
        Returns:
            时间编码 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = time_steps.shape
        
        # 确保时间步长在有效范围内
        time_steps = torch.clamp(time_steps, 0, self.max_length - 1)
        
        # 获取时间嵌入
        temporal_enc = self.time_embedding(time_steps)  # [batch_size, seq_len, embedding_dim]
        
        # 层归一化
        temporal_enc = self.layer_norm(temporal_enc)
        
        # 应用掩码
        if mask is not None:
            temporal_enc = temporal_enc.masked_fill(mask.unsqueeze(-1) == 0, 0)
        
        return self.dropout(temporal_enc)

class RelativeTemporalEncoding(nn.Module):
    """
    相对时间编码
    基于相对位置的时间编码
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.dropout = nn.Dropout(config.dropout)
        
        # 相对时间编码参数
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.max_length, self.max_length)
        )
        
        # 时间尺度参数
        self.time_scale = nn.Parameter(torch.tensor(config.time_scale))
        
        logger.info(f"相对时间编码初始化完成，维度: {self.embedding_dim}")
    
    @property
    def max_length(self):
        return self.config.max_sequence_length
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """获取相对位置矩阵"""
        # 创建相对位置矩阵
        positions = torch.arange(seq_len, dtype=torch.float32)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # 归一化相对位置
        relative_positions = relative_positions / self.time_scale
        
        return relative_positions
    
    def forward(self, time_steps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time_steps: 时间步长张量 [batch_size, seq_len]
            mask: 可选的掩码
            
        Returns:
            相对时间编码 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = time_steps.shape
        
        # 获取相对位置
        relative_positions = self._get_relative_positions(seq_len)
        
        # 应用相对位置偏置
        relative_encoding = relative_positions + self.relative_position_bias[:seq_len, :seq_len]
        
        # 扩展到多头
        relative_encoding = relative_encoding.unsqueeze(0).unsqueeze(0)
        relative_encoding = relative_encoding.expand(batch_size, self.num_heads, -1, -1)
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码到注意力矩阵形状
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
            relative_encoding = relative_encoding.masked_fill(attention_mask == 0, float('-inf'))
        
        return relative_encoding

class RoPETemporalEncoding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 时间编码
    旋转位置编码，支持更长的序列
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.max_length = config.max_sequence_length
        self.base_frequency = config.base_frequency
        
        # 预计算旋转矩阵
        self.register_buffer('cos_cached', None)
        self.register_buffer('sin_cached', None)
        self._cached_seq_len = 0
        
        logger.info(f"RoPE时间编码初始化完成，维度: {self.embedding_dim}")
    
    def _precompute_freqs_cis(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算频率和复数"""
        # 计算频率
        freqs = 1.0 / (self.base_frequency ** (torch.arange(0, self.embedding_dim, 2).float() / self.embedding_dim))
        
        # 创建位置
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        
        # 计算复数
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis
    
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            time_steps: 可选的时间步长
            
        Returns:
            RoPE编码后的张量 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # 预计算频率（如果序列长度变化）
        if seq_len > self._cached_seq_len:
            freqs_cis = self._precompute_freqs_cis(seq_len)
            self.register_buffer('cos_cached', freqs_cis.real)
            self.register_buffer('sin_cached', freqs_cis.imag)
            self._cached_seq_len = seq_len
        
        # 应用RoPE
        x_complex = torch.view_as_complex(x.float().reshape(batch_size, seq_len, -1, 2))
        freqs_cis = torch.view_as_complex(torch.stack([
            self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        ], dim=-1))
        
        # 旋转
        x_rotated = x_complex * freqs_cis.unsqueeze(0)
        
        # 转换回实数
        x_out = torch.view_as_real(x_rotated).reshape(batch_size, seq_len, embedding_dim)
        
        return x_out.type_as(x)

class TemporalMultiHeadAttention(nn.Module):
    """
    时间多头注意力机制
    集成时间编码的多头注意力
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.dropout = nn.Dropout(config.dropout)
        
        assert self.embedding_dim % self.num_heads == 0, "embedding_dim必须能被num_heads整除"
        
        # 线性变换层
        self.q_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # 时间编码
        self.temporal_encoding = self._create_temporal_encoding(config)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
        
        logger.info(f"时间多头注意力初始化完成，头数: {self.num_heads}, 维度: {self.embedding_dim}")
    
    def _create_temporal_encoding(self, config: TemporalEncodingConfig):
        """创建时间编码"""
        if config.encoding_type == "sincos":
            return SinusoidalTemporalEncoding(config)
        elif config.encoding_type == "learnable":
            return LearnableTemporalEncoding(config)
        elif config.encoding_type == "relative":
            return RelativeTemporalEncoding(config)
        elif config.encoding_type == "rope":
            return RoPETemporalEncoding(config)
        else:
            raise ValueError(f"不支持的时间编码类型: {config.encoding_type}")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, embedding_dim]
            key: 键张量 [batch_size, seq_len, embedding_dim]
            value: 值张量 [batch_size, seq_len, embedding_dim]
            mask: 可选的注意力掩码
            time_steps: 可选的时间步长
            
        Returns:
            注意力输出 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # 应用线性变换
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 应用时间编码
        if time_steps is not None:
            if isinstance(self.temporal_encoding, (SinusoidalTemporalEncoding, LearnableTemporalEncoding)):
                # 添加式时间编码
                time_enc = self.temporal_encoding(time_steps, mask)
                Q = Q + time_enc
                K = K + time_enc
            elif isinstance(self.temporal_encoding, RoPETemporalEncoding):
                # RoPE编码
                Q = self.temporal_encoding(Q, time_steps)
                K = self.temporal_encoding(K, time_steps)
        
        # 重塑为多头形式
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用相对时间编码（如果适用）
        if isinstance(self.temporal_encoding, RelativeTemporalEncoding) and time_steps is not None:
            relative_bias = self.temporal_encoding(time_steps, mask)
            scores = scores + relative_bias
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)
        
        # 重塑回原始形状
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embedding_dim
        )
        
        # 最终线性变换
        output = self.out_linear(attention_output)
        
        return output

class TemporalTransformerEncoder(nn.Module):
    """
    时间编码Transformer编码器
    集成时间编码的多层Transformer编码器
    """
    
    def __init__(self, config: TemporalEncodingConfig, num_layers: int = 6):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.embedding_dim = config.embedding_dim
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(config) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        logger.info(f"时间编码Transformer编码器初始化完成，层数: {num_layers}, 维度: {self.embedding_dim}")
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            mask: 可选的掩码
            time_steps: 可选的时间步长
            
        Returns:
            编码输出 [batch_size, seq_len, embedding_dim]
        """
        # 通过所有Transformer层
        for layer in self.layers:
            x = layer(x, mask, time_steps)
        
        # 最终层归一化
        x = self.layer_norm(x)
        
        return x

class TemporalTransformerLayer(nn.Module):
    """
    时间Transformer层
    单个Transformer层，包含时间编码注意力
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.dropout = nn.Dropout(config.dropout)
        
        # 多头注意力
        self.self_attention = TemporalMultiHeadAttention(config)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * self.embedding_dim, self.embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # 层归一化
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        self.ff_norm = nn.LayerNorm(self.embedding_dim)
        
        logger.info(f"时间Transformer层初始化完成，维度: {self.embedding_dim}")
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            mask: 可选的掩码
            time_steps: 可选的时间步长
            
        Returns:
            输出张量 [batch_size, seq_len, embedding_dim]
        """
        # 自注意力
        attention_output = self.self_attention(x, x, x, mask, time_steps)
        x = self.attention_norm(x + attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        return x

class TemporalTransformerEncoderWrapper:
    """
    时间编码Transformer包装器
    提供简化的API和配置管理
    """
    
    def __init__(self, config: TemporalEncodingConfig, num_layers: int = 6):
        self.config = config
        self.model = TemporalTransformerEncoder(config, num_layers)
        
        logger.info(f"时间编码Transformer包装器初始化完成")
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        return self.model(x, mask, time_steps)
    
    def get_model(self) -> nn.Module:
        """获取模型实例"""
        return self.model
    
    def get_config(self) -> TemporalEncodingConfig:
        """获取配置"""
        return self.config
    
    def save_model(self, filepath: Union[str, Path]):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_class': self.model.__class__.__name__
            }, filepath)
            logger.info(f"模型已保存到: {filepath}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def load_model(self, filepath: Union[str, Path], map_location: Optional[str] = None):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")

# 时间编码类型映射
TEMPORAL_ENCODING_TYPES = {
    "sincos": SinusoidalTemporalEncoding,
    "learnable": LearnableTemporalEncoding,
    "relative": RelativeTemporalEncoding,
    "rope": RoPETemporalEncoding,
}

def create_temporal_encoder(config: TemporalEncodingConfig, num_layers: int = 6) -> TemporalTransformerEncoderWrapper:
    """
    创建时间编码Transformer
    
    Args:
        config: 时间编码配置
        num_layers: Transformer层数
        
    Returns:
        时间编码Transformer包装器
    """
    return TemporalTransformerEncoderWrapper(config, num_layers)