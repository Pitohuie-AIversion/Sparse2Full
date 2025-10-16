"""SegFormer模型

SegFormer是一个简单高效的语义分割Transformer模型，这里适配用于图像重建任务。
使用分层Transformer编码器和轻量级MLP解码器。

Reference:
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    https://arxiv.org/abs/2105.15203
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .base import BaseModel


class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        
        # 确保patch_size不会超过输入尺寸
        if isinstance(patch_size, int):
            patch_size = min(patch_size, img_size)
        
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size


class Attention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP模块"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SegFormer(BaseModel):
    """SegFormer模型
    
    基于Transformer的分层编码器，用于图像重建任务
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        embed_dims: list = [64, 128, 320, 512],
        num_heads: list = [1, 2, 5, 8],
        mlp_ratios: list = [4, 4, 4, 4],
        depths: list = [3, 4, 6, 3],
        sr_ratios: list = [8, 4, 2, 1],
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        

        self.embed_dims = embed_dims
        
        # Patch嵌入层 - 动态调整patch sizes以适应小尺寸输入
        # 对于128x128输入，使用更保守的patch sizes
        if img_size <= 128:
            patch_sizes = [4, 2, 2, 2]  # 更小的patch sizes
        else:
            patch_sizes = [7, 3, 3, 3]
        
        # 计算每个阶段的输入尺寸并调整patch size
        current_size = img_size
        adjusted_patch_sizes = []
        
        for i, base_size in enumerate(patch_sizes):
            # 确保patch size不超过当前特征图尺寸，且至少为1
            patch_size = min(base_size, max(1, current_size))
            adjusted_patch_sizes.append(patch_size)
            current_size = max(1, current_size // patch_size)
        
        self.patch_sizes = adjusted_patch_sizes
        self.patch_embed1 = PatchEmbed(img_size, adjusted_patch_sizes[0], in_channels, embed_dims[0])
        
        # 计算后续阶段的输入尺寸
        size_after_1 = img_size // adjusted_patch_sizes[0]
        self.patch_embed2 = PatchEmbed(size_after_1, adjusted_patch_sizes[1], embed_dims[0], embed_dims[1])
        
        size_after_2 = size_after_1 // adjusted_patch_sizes[1]
        self.patch_embed3 = PatchEmbed(size_after_2, adjusted_patch_sizes[2], embed_dims[1], embed_dims[2])
        
        size_after_3 = size_after_2 // adjusted_patch_sizes[2]
        self.patch_embed4 = PatchEmbed(size_after_3, adjusted_patch_sizes[3], embed_dims[2], embed_dims[3])
        
        # Transformer块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            drop=drop_rate, attn_drop=0.0, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            drop=drop_rate, attn_drop=0.0, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            drop=drop_rate, attn_drop=0.0, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            drop=drop_rate, attn_drop=0.0, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        
        # MLP解码器
        self.decode_head = nn.Sequential(
            nn.Conv2d(sum(embed_dims), 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 4 - 检查输入尺寸是否足够
        if x.shape[2] >= 3 and x.shape[3] >= 3:  # 确保输入尺寸至少为3x3
            x, H, W = self.patch_embed4(x)
            for blk in self.block4:
                x = blk(x, H, W)
            x = self.norm4(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        else:
            # 如果输入太小，跳过stage 4或使用1x1卷积
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.expand(-1, -1, 1, 1)  # 保持维度
            outs.append(x)
        
        return outs
    
    def forward(self, x):
        features = self.forward_features(x)
        
        # 上采样所有特征到相同尺寸
        upsampled_features = []
        target_size = features[0].shape[2:]
        
        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 特征融合
        x = torch.cat(upsampled_features, dim=1)
        
        # 解码
        x = self.decode_head(x)
        
        # 上采样到输入尺寸
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return x
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'name': 'SegFormer',
            'type': 'Transformer',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'img_size': self.img_size,
            'embed_dims': self.embed_dims
        }