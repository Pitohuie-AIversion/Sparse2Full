"""Vision Transformer (ViT) 模型实现

基于标准Vision Transformer架构，适配PDEBench稀疏观测重建任务。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .base import BaseModel


class PatchEmbedding(nn.Module):
    """图像块嵌入层"""
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 图像块投影
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 可选的归一化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            patches: [B, N, D] where N = (H//P)*(W//P), D = embed_dim
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}, {W}) doesn't match model ({self.img_size}, {self.img_size})"
        
        # 投影到patch embeddings
        x = self.proj(x)  # [B, embed_dim, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """前馈网络"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # 随机深度（Stochastic Depth）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP + 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(BaseModel):
    """Vision Transformer模型
    
    基于标准ViT架构，适配PDEBench稀疏观测重建任务。
    严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        # 解码器参数
        decoder_embed_dim: Optional[int] = None,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # 最终激活函数
        final_activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # 解码器参数
        self.decoder_embed_dim = decoder_embed_dim or embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer编码器
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # 编码器到解码器的投影
        if self.decoder_embed_dim != embed_dim:
            self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        else:
            self.decoder_embed = nn.Identity()
        
        # 解码器位置编码
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.decoder_embed_dim)
        )
        
        # Transformer解码器
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_embed_dim, num_heads=self.decoder_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=0.0,  # 解码器不使用drop_path
                norm_layer=norm_layer, act_layer=act_layer
            )
            for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        
        # 输出投影
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim, 
            patch_size**2 * out_channels, 
            bias=True
        )
        
        # 最终激活函数
        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 位置编码使用截断正态分布
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # 线性层权重初始化
        def _init_linear(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(_init_linear)
        
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """编码器前向传播"""
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # 添加cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, D]
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer编码器
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 移除cls token，只保留patch tokens
        x = x[:, 1:, :]  # [B, N, D]
        
        return x
        
    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """解码器前向传播"""
        # 编码器到解码器投影
        x = self.decoder_embed(x)  # [B, N, decoder_embed_dim]
        
        # 添加位置编码
        x = x + self.decoder_pos_embed
        
        # Transformer解码器
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 预测patch像素值
        x = self.decoder_pred(x)  # [B, N, patch_size^2 * out_channels]
        
        return x
        
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """将patch tokens重构为图像"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        
        return imgs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
        Returns:
            y: 输出张量 [B, C_out, H, W]
        """
        # 编码器
        latent = self.forward_encoder(x)  # [B, N, D]
        
        # 解码器
        pred = self.forward_decoder(latent)  # [B, N, patch_size^2 * out_channels]
        
        # 重构图像
        y = self.unpatchify(pred)  # [B, C_out, H, W]
        
        # 应用最终激活函数
        y = self.final_activation(y)
        
        return y
        
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_flops(self, input_shape: Tuple[int, int, int, int]) -> int:
        """估算FLOPs（简化版本）"""
        B, C, H, W = input_shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # Patch embedding FLOPs
        patch_flops = C * (self.patch_size ** 2) * self.embed_dim * num_patches
        
        # Transformer encoder FLOPs (简化估算)
        encoder_flops = self.depth * num_patches * self.embed_dim * (
            4 * self.embed_dim +  # QKV projection + output projection
            2 * num_patches * self.embed_dim +  # Attention computation
            8 * self.embed_dim  # MLP
        )
        
        # Transformer decoder FLOPs
        decoder_flops = self.decoder_depth * num_patches * self.decoder_embed_dim * (
            4 * self.decoder_embed_dim +
            2 * num_patches * self.decoder_embed_dim +
            8 * self.decoder_embed_dim
        )
        
        # Output projection FLOPs
        output_flops = num_patches * self.decoder_embed_dim * (self.patch_size ** 2) * self.out_channels
        
        total_flops = patch_flops + encoder_flops + decoder_flops + output_flops
        return int(total_flops * B)


# 别名，保持一致性
ViT = VisionTransformer