"""UNetFormer模型

UNetFormer是在U-Net架构中集成Transformer块的混合模型，
结合了CNN的局部特征提取能力和Transformer的全局建模能力。

Reference:
    UNetFormer: A UNet-like transformer for efficient semantic segmentation
    https://arxiv.org/abs/2109.08417
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .base import BaseModel


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 确保num_heads是整数且能被dim整除
        if hasattr(num_heads, '__iter__'):
            num_heads = num_heads[0] if len(num_heads) > 0 else 8
        
        # 确保dim能被num_heads整除
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
            
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 添加数值稳定性保护
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 防止注意力权重过大
        attn = torch.clamp(attn, min=-10, max=10)
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
        # 添加数值稳定性保护
        x = torch.clamp(x, min=-100, max=100)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 确保num_heads是整数且能被dim整除
        if hasattr(num_heads, '__iter__'):
            num_heads = num_heads[0] if len(num_heads) > 0 else 8
        
        # 确保dim能被num_heads整除
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    """卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class TransformerConvBlock(nn.Module):
    """Transformer-Conv混合块"""
    
    def __init__(self, channels, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.channels = channels
        self.conv = ConvBlock(channels, channels)
        self.transformer = TransformerBlock(channels, num_heads, mlp_ratio, drop=drop, attn_drop=attn_drop)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # 卷积分支
        conv_out = self.conv(x)
        
        # Transformer分支
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
        trans_out = self.transformer(x_flat)
        trans_out = trans_out.transpose(1, 2).reshape(B, C, H, W)
        
        # 残差连接
        return conv_out + trans_out


class UNetFormer(BaseModel):
    """UNetFormer模型
    
    结合CNN和Transformer的U-Net架构，用于图像重建任务
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        base_channels: int = 64,
        num_stages: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.base_channels = base_channels
        self.num_stages = num_stages
        
        # 定义每个阶段的Transformer块数量 - 根据num_stages调整
        if num_stages == 2:
            self.depths = [1, 1]  # 2阶段配置
        elif num_stages == 3:
            self.depths = [1, 1, 2]  # 3阶段配置
        else:
            self.depths = [2, 2, 6, 2]  # 4阶段配置
        
        # 为了向后兼容，也创建局部变量
        depths = self.depths
        
        # 编码器 - 根据num_stages动态构建
        if num_stages >= 1:
            self.encoder1 = nn.Sequential(
                ConvBlock(in_channels, base_channels),
                *[TransformerConvBlock(base_channels, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[0])]
            )
        
        if num_stages >= 2:
            self.encoder2 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels, base_channels * 2),
                *[TransformerConvBlock(base_channels * 2, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[1])]
            )
        
        if num_stages >= 3:
            self.encoder3 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels * 2, base_channels * 4),
                *[TransformerConvBlock(base_channels * 4, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[2])]
            )
        
        if num_stages >= 4:
            self.encoder4 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels * 4, base_channels * 8),
                *[TransformerConvBlock(base_channels * 8, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[3])]
            )
        
        # 瓶颈层 - 根据num_stages调整
        if num_stages == 2:
            self.bottleneck = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels * 2, base_channels * 4),
                TransformerConvBlock(base_channels * 4, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            )
        elif num_stages == 3:
            self.bottleneck = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels * 4, base_channels * 8),
                TransformerConvBlock(base_channels * 8, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(base_channels * 8, base_channels * 16),
                TransformerConvBlock(base_channels * 16, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            )
        
        # 解码器 - 根据num_stages动态构建
        if num_stages == 2:
            # 2阶段配置
            self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
            self.decoder2 = nn.Sequential(
                ConvBlock(base_channels * 4, base_channels * 2),
                *[TransformerConvBlock(base_channels * 2, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[1])]
            )
            
            self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
            self.decoder1 = nn.Sequential(
                ConvBlock(base_channels * 2, base_channels),
                *[TransformerConvBlock(base_channels, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[0])]
            )
        elif num_stages == 3:
            # 3阶段配置
            self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
            self.decoder3 = nn.Sequential(
                ConvBlock(base_channels * 8, base_channels * 4),
                *[TransformerConvBlock(base_channels * 4, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[2])]
            )
            
            self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
            self.decoder2 = nn.Sequential(
                ConvBlock(base_channels * 4, base_channels * 2),
                *[TransformerConvBlock(base_channels * 2, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[1])]
            )
            
            self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
            self.decoder1 = nn.Sequential(
                ConvBlock(base_channels * 2, base_channels),
                *[TransformerConvBlock(base_channels, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[0])]
            )
        else:
            # 4阶段配置
            self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
            self.decoder4 = nn.Sequential(
                ConvBlock(base_channels * 16, base_channels * 8),
                *[TransformerConvBlock(base_channels * 8, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[3])]
            )
            
            self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
            self.decoder3 = nn.Sequential(
                ConvBlock(base_channels * 8, base_channels * 4),
                *[TransformerConvBlock(base_channels * 4, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[2])]
            )
            
            self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
            self.decoder2 = nn.Sequential(
                ConvBlock(base_channels * 4, base_channels * 2),
                *[TransformerConvBlock(base_channels * 2, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[1])]
            )
            
            self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
            self.decoder1 = nn.Sequential(
                ConvBlock(base_channels * 2, base_channels),
                *[TransformerConvBlock(base_channels, num_heads, mlp_ratio, drop_rate, attn_drop_rate) 
                  for _ in range(depths[0])]
            )
        
        # 输出层
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        self.apply(self._init_weights)
        
        # 添加数值稳定性保护
        self.register_buffer('eps', torch.tensor(1e-8))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用更小的初始化标准差
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # 使用Xavier初始化而不是Kaiming
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # 添加输入数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 编码器 - 添加更严格的数值稳定性检查
        enc1 = self.encoder1(x)
        enc1 = torch.clamp(enc1, min=-5, max=5)
        enc1 = torch.nan_to_num(enc1, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if self.num_stages >= 2:
            enc2 = self.encoder2(enc1)
            enc2 = torch.clamp(enc2, min=-5, max=5)
            enc2 = torch.nan_to_num(enc2, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if self.num_stages >= 3:
            enc3 = self.encoder3(enc2)
            enc3 = torch.clamp(enc3, min=-5, max=5)
            enc3 = torch.nan_to_num(enc3, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if self.num_stages >= 4:
            enc4 = self.encoder4(enc3)
            enc4 = torch.clamp(enc4, min=-5, max=5)
            enc4 = torch.nan_to_num(enc4, nan=0.0, posinf=5.0, neginf=-5.0)
            # 瓶颈层
            bottleneck = self.bottleneck(enc4)
        elif self.num_stages == 3:
            # 3阶段情况
            bottleneck = self.bottleneck(enc3)
        else:
            # 2阶段情况
            bottleneck = self.bottleneck(enc2)
        
        bottleneck = torch.clamp(bottleneck, min=-5, max=5)
        bottleneck = torch.nan_to_num(bottleneck, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # 解码器 - 根据阶段数动态解码，添加更严格的数值稳定性检查
        if self.num_stages == 2:
            # 2阶段解码
            dec2 = self.upconv2(bottleneck)
            dec2 = torch.clamp(dec2, min=-5, max=5)
            dec2 = torch.nan_to_num(dec2, nan=0.0, posinf=5.0, neginf=-5.0)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.decoder2(dec2)
            dec2 = torch.clamp(dec2, min=-5, max=5)
            dec2 = torch.nan_to_num(dec2, nan=0.0, posinf=5.0, neginf=-5.0)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.clamp(dec1, min=-5, max=5)
            dec1 = torch.nan_to_num(dec1, nan=0.0, posinf=5.0, neginf=-5.0)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.decoder1(dec1)
            dec1 = torch.clamp(dec1, min=-5, max=5)
            dec1 = torch.nan_to_num(dec1, nan=0.0, posinf=5.0, neginf=-5.0)
        elif self.num_stages == 3:
            # 3阶段解码
            dec3 = self.upconv3(bottleneck)
            dec3 = torch.clamp(dec3, min=-10, max=10)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.decoder3(dec3)
            dec3 = torch.clamp(dec3, min=-10, max=10)
            
            dec2 = self.upconv2(dec3)
            dec2 = torch.clamp(dec2, min=-10, max=10)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.decoder2(dec2)
            dec2 = torch.clamp(dec2, min=-10, max=10)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.clamp(dec1, min=-10, max=10)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.decoder1(dec1)
            dec1 = torch.clamp(dec1, min=-10, max=10)
        else:
            # 4阶段解码
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.clamp(dec4, min=-10, max=10)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.decoder4(dec4)
            dec4 = torch.clamp(dec4, min=-10, max=10)
            
            dec3 = self.upconv3(dec4)
            dec3 = torch.clamp(dec3, min=-10, max=10)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.decoder3(dec3)
            dec3 = torch.clamp(dec3, min=-10, max=10)
            
            dec2 = self.upconv2(dec3)
            dec2 = torch.clamp(dec2, min=-10, max=10)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.decoder2(dec2)
            dec2 = torch.clamp(dec2, min=-10, max=10)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.clamp(dec1, min=-10, max=10)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.decoder1(dec1)
            dec1 = torch.clamp(dec1, min=-10, max=10)
        
        # 输出 - 添加更严格的数值稳定性检查
        output = self.final_conv(dec1)
        output = torch.clamp(output, min=-2, max=2)  # 更严格的输出范围
        output = torch.nan_to_num(output, nan=0.0, posinf=2.0, neginf=-2.0)
        
        # 最终输出检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Output contains NaN or Inf values")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'name': 'UNetFormer',
            'type': 'Hybrid',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'img_size': self.img_size,
            'base_channels': self.base_channels
        }