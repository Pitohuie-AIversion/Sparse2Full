"""经典Transformer模型实现

基于"Attention is All You Need"论文的标准Transformer架构，
适配PDEBench稀疏观测重建任务的2D图像输入输出。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .base import BaseModel


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads, dropout: float = 0.1):
        super().__init__()
        # 处理Hydra配置中的列表格式
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]

        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # 最终线性变换
        output = self.w_o(attention_output)
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, num_heads, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 处理Hydra配置中的列表格式
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model: int, num_heads, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 处理Hydra配置中的列表格式
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
            src_mask: [batch_size, tgt_seq_len, src_seq_len]
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
        # 自注意力 + 残差连接 + 层归一化
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力 + 残差连接 + 层归一化
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class PatchEmbedding(nn.Module):
    """图像块嵌入层"""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, img_size, img_size]
        Returns:
            patches: [batch_size, num_patches, d_model]
        """
        x = self.proj(x)  # [B, d_model, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        return x


class PatchReconstruction(nn.Module):
    """图像块重构层"""
    
    def __init__(self, d_model: int, patch_size: int, out_channels: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Linear(d_model, out_channels * patch_size * patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_patches, d_model]
        Returns:
            img: [batch_size, out_channels, img_size, img_size]
        """
        B, N, D = x.shape
        H = W = self.img_size // self.patch_size
        
        x = self.proj(x)  # [B, N, out_channels * patch_size^2]
        x = x.view(B, H, W, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, self.img_size, self.img_size)
        
        return x


class Transformer(BaseModel):
    """经典Transformer模型
    
    基于"Attention is All You Need"论文的标准Transformer架构，
    适配2D图像输入输出的PDE重建任务。
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        patch_size: int = 16,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        """初始化Transformer模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            img_size: 图像尺寸
            patch_size: 图像块大小
            d_model: 模型维度
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        # 处理Hydra配置中的列表格式
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        
        # 图像块嵌入
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Transformer解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出嵌入（用于解码器输入）
        self.output_embedding = nn.Embedding(self.num_patches, d_model)
        
        # 图像重构
        self.patch_reconstruction = PatchReconstruction(d_model, patch_size, out_channels, img_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """编码器前向传播
        
        Args:
            src: [batch_size, in_channels, img_size, img_size]
        Returns:
            encoder_output: [batch_size, num_patches, d_model]
        """
        # 图像块嵌入
        src_patches = self.patch_embedding(src)  # [B, N, d_model]
        
        # 转换为序列格式 [seq_len, batch_size, d_model]
        src_patches = src_patches.transpose(0, 1)
        
        # 添加位置编码
        src_patches = self.pos_encoding(src_patches)
        src_patches = self.dropout(src_patches)
        
        # 通过编码器层
        encoder_output = src_patches
        for layer in self.encoder_layers:
            # 转换回 [batch_size, seq_len, d_model] 格式
            encoder_output = encoder_output.transpose(0, 1)
            encoder_output = layer(encoder_output)
            encoder_output = encoder_output.transpose(0, 1)
        
        # 转换回 [batch_size, seq_len, d_model]
        encoder_output = encoder_output.transpose(0, 1)
        return encoder_output
    
    def decode(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """解码器前向传播
        
        Args:
            encoder_output: [batch_size, num_patches, d_model]
        Returns:
            decoder_output: [batch_size, num_patches, d_model]
        """
        batch_size = encoder_output.size(0)
        
        # 创建目标序列（使用学习的嵌入）
        tgt_indices = torch.arange(self.num_patches, device=encoder_output.device)
        tgt_indices = tgt_indices.unsqueeze(0).expand(batch_size, -1)
        tgt = self.output_embedding(tgt_indices)  # [B, N, d_model]
        
        # 转换为序列格式 [seq_len, batch_size, d_model]
        tgt = tgt.transpose(0, 1)
        
        # 添加位置编码
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # 通过解码器层
        decoder_output = tgt
        for layer in self.decoder_layers:
            # 转换回 [batch_size, seq_len, d_model] 格式
            decoder_output = decoder_output.transpose(0, 1)
            encoder_output_layer = encoder_output
            decoder_output = layer(decoder_output, encoder_output_layer)
            decoder_output = decoder_output.transpose(0, 1)
        
        # 转换回 [batch_size, seq_len, d_model]
        decoder_output = decoder_output.transpose(0, 1)
        return decoder_output
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（未使用）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 编码
        encoder_output = self.encode(x)
        
        # 解码
        decoder_output = self.decode(encoder_output)
        
        # 重构图像
        output = self.patch_reconstruction(decoder_output)
        
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'patch_size': self.patch_size,
            'd_model': self.d_model,
            'num_patches': self.num_patches,
            'encoder_layers': len(self.encoder_layers),
            'decoder_layers': len(self.decoder_layers),
        })
        return info


def create_transformer(**kwargs) -> Transformer:
    """创建Transformer模型的工厂函数"""
    return Transformer(**kwargs)