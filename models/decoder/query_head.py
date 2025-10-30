"""NAR查询头实现

时间查询头，用于一次性并行预测多个时间步。
支持轻量级时间条件调制和更强的交叉注意力机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def sinusoid_time_embed(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """正弦位置编码
    
    Args:
        timesteps: 时间步张量 (T,)
        dim: 编码维度
        
    Returns:
        时间编码 (T, dim)
    """
    half_dim = dim // 2
    
    # 处理dim=1或dim=2的边界情况
    if half_dim == 0:
        # dim=1的情况，直接返回零向量
        return torch.zeros(len(timesteps), dim, device=timesteps.device)
    elif half_dim == 1:
        # dim=2的情况，使用简化的编码
        emb = timesteps[:, None]  # (T, 1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (T, 2)
    else:
        # 标准情况
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    if dim % 2 == 1:  # 如果维度是奇数，补零
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb


class TimeQueryHead(nn.Module):
    """时间查询头（轻量版）
    
    使用时间条件调制进行NAR多步预测。
    设计原则：轻量、稳定、并行高效。
    
    Args:
        d_model: 输入特征维度
        c_out: 输出通道数
        max_timesteps: 最大时间步数，默认32
        use_layer_norm: 是否使用LayerNorm，默认True
        dropout: dropout概率，默认0.0
    """
    
    def __init__(
        self, 
        d_model: int, 
        c_out: int,
        max_timesteps: int = 64,  # 扩展到64以支持T_out=10
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.c_out = c_out
        self.max_timesteps = max_timesteps
        
        # Key-Value生成器
        self.to_kv = nn.Conv2d(d_model, 2 * d_model, kernel_size=1, bias=False)
        
        # 时间条件投影层
        self.time_proj = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.output_proj = nn.Conv2d(d_model, c_out, kernel_size=1)
        
        # 可选的归一化和dropout
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"TimeQueryHead: d_model={d_model}, c_out={c_out}, max_T={max_timesteps}")
    
    def _init_weights(self):
        """初始化权重"""
        # Xavier初始化
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # 偏置初始化为0
        if self.time_proj.bias is not None:
            nn.init.constant_(self.time_proj.bias, 0)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(self, memory: torch.Tensor, T_out: int) -> torch.Tensor:
        """前向传播
        
        Args:
            memory: 记忆特征 (B, D, H, W)
            T_out: 输出时间步数
            
        Returns:
            预测序列 (B, T_out, C, H, W)
        """
        B, D, H, W = memory.shape
        
        if T_out > self.max_timesteps:
            logger.warning(f"T_out ({T_out}) > max_timesteps ({self.max_timesteps})")
        
        # 生成Key和Value
        kv = self.to_kv(memory)  # (B, 2*D, H, W)
        k, v = kv.split(D, dim=1)  # 各自 (B, D, H, W)
        
        # 生成时间查询
        timesteps = torch.arange(1, T_out + 1, device=memory.device, dtype=torch.float32)
        time_embed = sinusoid_time_embed(timesteps, D)  # (T_out, D)
        
        # 时间条件投影
        time_queries = self.time_proj(time_embed)  # (T_out, D)
        
        if self.layer_norm is not None:
            time_queries = self.layer_norm(time_queries)
        
        if self.dropout is not None:
            time_queries = self.dropout(time_queries)
        
        # 优化的并行时间条件调制
        # 将所有时间查询重塑为 (T_out, D, 1, 1) 以支持批量处理
        time_queries = time_queries.view(T_out, D, 1, 1)  # (T_out, D, 1, 1)
        
        # 扩展V以匹配时间维度: (B, D, H, W) -> (B, T_out, D, H, W)
        v_expanded = v.unsqueeze(1).expand(B, T_out, D, H, W)  # (B, T_out, D, H, W)
        
        # 扩展时间查询以匹配批次维度: (T_out, D, 1, 1) -> (B, T_out, D, H, W)
        time_queries_expanded = time_queries.unsqueeze(0).expand(B, T_out, D, H, W)
        
        # 并行时间条件调制
        conditioned_v = v_expanded * time_queries_expanded  # (B, T_out, D, H, W)
        
        # 重塑为批量处理格式: (B, T_out, D, H, W) -> (B*T_out, D, H, W)
        conditioned_v = conditioned_v.view(B * T_out, D, H, W)
        
        # 批量输出投影
        output = self.output_proj(conditioned_v)  # (B*T_out, C, H, W)
        
        # 重塑回时序格式: (B*T_out, C, H, W) -> (B, T_out, C, H, W)
        output = output.view(B, T_out, self.c_out, H, W)
        
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'module_type': 'TimeQueryHead',
            'd_model': self.d_model,
            'c_out': self.c_out,
            'max_timesteps': self.max_timesteps,
            'parameters': sum(p.numel() for p in self.parameters()),
        }


class CrossAttentionQueryHead(nn.Module):
    """交叉注意力查询头（增强版）
    
    使用交叉注意力机制进行NAR多步预测。
    相比TimeQueryHead更强但计算开销更大。
    
    Args:
        d_model: 输入特征维度
        c_out: 输出通道数
        num_heads: 注意力头数，默认8
        max_timesteps: 最大时间步数，默认32
        dropout: dropout概率，默认0.1
    """
    
    def __init__(
        self, 
        d_model: int, 
        c_out: int,
        num_heads: int = 8,
        max_timesteps: int = 64,  # 扩展到64以支持T_out=10
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.c_out = c_out
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_timesteps = max_timesteps
        self.scale = self.head_dim ** -0.5
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        
        # 输出投影
        self.out_proj = nn.Conv2d(d_model, c_out, kernel_size=1)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"CrossAttentionQueryHead: d_model={d_model}, heads={num_heads}, c_out={c_out}")
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, memory: torch.Tensor, T_out: int) -> torch.Tensor:
        """前向传播
        
        Args:
            memory: 记忆特征 (B, D, H, W)
            T_out: 输出时间步数
            
        Returns:
            预测序列 (B, T_out, C, H, W)
        """
        B, D, H, W = memory.shape
        
        if T_out > self.max_timesteps:
            logger.warning(f"T_out ({T_out}) > max_timesteps ({self.max_timesteps})")
        
        # 生成时间查询
        timesteps = torch.arange(1, T_out + 1, device=memory.device, dtype=torch.float32)
        time_embed = sinusoid_time_embed(timesteps, D)  # (T_out, D)
        time_embed = self.norm(time_embed)
        
        # 投影查询、键、值
        Q = self.q_proj(time_embed)  # (T_out, D)
        K = self.k_proj(memory)      # (B, D, H, W)
        V = self.v_proj(memory)      # (B, D, H, W)
        
        # 重塑为多头注意力格式
        Q = Q.view(T_out, self.num_heads, self.head_dim)  # (T_out, num_heads, head_dim)
        K = K.view(B, self.num_heads, self.head_dim, H * W)  # (B, num_heads, head_dim, H*W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)  # (B, num_heads, head_dim, H*W)
        
        # 计算注意力
        outputs = []
        for t in range(T_out):
            q_t = Q[t]  # (num_heads, head_dim)
            q_t = q_t.unsqueeze(0).expand(B, -1, -1)  # (B, num_heads, head_dim)
            
            # 注意力分数：Q @ K^T
            # 修复维度匹配问题：使用矩阵乘法而不是einsum
            attn_scores = torch.matmul(q_t.unsqueeze(-2), K.transpose(-2, -1))  # (B, num_heads, 1, H*W)
            attn_scores = attn_scores.squeeze(-2)  # (B, num_heads, H*W)
            attn_scores = attn_scores * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # 加权求和：Attn @ V
            # 修复维度匹配问题：使用矩阵乘法而不是einsum
            out_t = torch.matmul(attn_weights.unsqueeze(-2), V.transpose(-2, -1))  # (B, num_heads, 1, head_dim)
            out_t = out_t.squeeze(-2)  # (B, num_heads, head_dim)
            out_t = out_t.contiguous().view(B, D)  # (B, D)
            out_t = out_t.view(B, D, 1, 1).expand(B, D, H, W)  # (B, D, H, W)
            
            # 输出投影
            out_t = self.out_proj(out_t)  # (B, C, H, W)
            out_t = self.proj_dropout(out_t)
            
            outputs.append(out_t.unsqueeze(1))  # (B, 1, C, H, W)
        
        # 拼接所有时间步
        output = torch.cat(outputs, dim=1)  # (B, T_out, C, H, W)
        
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'module_type': 'CrossAttentionQueryHead',
            'd_model': self.d_model,
            'c_out': self.c_out,
            'num_heads': self.num_heads,
            'max_timesteps': self.max_timesteps,
            'parameters': sum(p.numel() for p in self.parameters()),
        }


def create_query_head(
    head_type: str,
    d_model: int,
    c_out: int,
    **kwargs
) -> nn.Module:
    """查询头工厂函数
    
    Args:
        head_type: 查询头类型 ('simple' | 'cross_attention')
        d_model: 输入特征维度
        c_out: 输出通道数
        **kwargs: 其他参数
        
    Returns:
        查询头实例
    """
    if head_type == 'simple':
        return TimeQueryHead(d_model=d_model, c_out=c_out, **kwargs)
    elif head_type == 'cross_attention':
        return CrossAttentionQueryHead(d_model=d_model, c_out=c_out, **kwargs)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


# 导出接口
__all__ = [
    'TimeQueryHead',
    'CrossAttentionQueryHead',
    'create_query_head',
    'sinusoid_time_embed'
]