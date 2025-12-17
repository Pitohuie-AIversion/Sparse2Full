"""
稀疏注意力编码器 - 基于Senseiver的传感器注意力机制

用于编码稀疏观测数据，通过注意力机制增强传感器位置的特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SparseAttentionEncoder(nn.Module):
    """稀疏注意力编码器
    
    基于Senseiver的注意力机制，将稀疏观测数据与坐标、掩码信息融合，
    生成增强的特征表示供后续模型使用。
    
    Args:
        in_channels: 输入通道数（通常为baseline + coords + mask的通道数）
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        sensor_dim: 传感器位置编码维度
        coord_dim: 坐标编码维度  
        mask_dim: 掩码编码维度
        dropout: dropout率
        use_sparse_bias: 是否使用稀疏偏置（只在观测点计算注意力）
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        sensor_dim: int = 128,
        coord_dim: int = 64,
        mask_dim: int = 32,
        dropout: float = 0.1,
        use_sparse_bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_sparse_bias = use_sparse_bias
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 输入投影层
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # 传感器位置编码（可学习的位置嵌入）
        self.sensor_embedding = nn.Sequential(
            nn.Conv2d(1, sensor_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=sensor_dim),  # 使用GroupNorm替代LayerNorm
            nn.GELU(),
            nn.Conv2d(sensor_dim, sensor_dim, kernel_size=1)
        )
        
        # 坐标编码
        self.coord_embedding = nn.Sequential(
            nn.Conv2d(2, coord_dim, kernel_size=1),  # x, y坐标
            nn.GroupNorm(num_groups=8, num_channels=coord_dim),
            nn.GELU(),
            nn.Conv2d(coord_dim, coord_dim, kernel_size=1)
        )
        
        # 掩码编码
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(1, mask_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=mask_dim),
            nn.GELU(),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1)
        )
        
        # 组合特征投影
        total_feature_dim = embed_dim + sensor_dim + coord_dim + mask_dim
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_feature_dim, embed_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),  # 使用GroupNorm替代LayerNorm
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多头自注意力
        self.qkv_proj = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=embed_dim)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=embed_dim)
        
        # 输出投影（适配SwinUNet的输入要求）
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _create_sparse_attention_mask(self, mask: torch.Tensor, window_size: int = 7) -> torch.Tensor:
        """创建稀疏注意力掩码
        
        只在观测点（mask > 0）及其邻域内计算注意力，减少计算量
        
        Args:
            mask: 观测掩码 [B, 1, H, W]
            window_size: 注意力窗口大小
            
        Returns:
            稀疏注意力掩码 [B, H*W, H*W]
        """
        B, _, H, W = mask.shape
        device = mask.device
        
        # 创建观测点掩码
        obs_mask = (mask > 0.5).float()  # [B, 1, H, W]
        
        # 膨胀操作，扩大观测点邻域
        if window_size > 1:
            kernel = torch.ones(1, 1, window_size, window_size, device=device)
            obs_mask = F.conv2d(obs_mask, kernel, padding=window_size//2)
            obs_mask = (obs_mask > 0).float()
        
        # 转换为序列格式
        obs_mask_flat = obs_mask.view(B, H * W)  # [B, H*W]
        
        # 创建稀疏注意力掩码：只在观测点之间计算注意力
        sparse_mask = obs_mask_flat.unsqueeze(2) * obs_mask_flat.unsqueeze(1)  # [B, H*W, H*W]
        
        # 将未观测位置的注意力权重设为负无穷，但避免全负无穷
        attention_mask = torch.zeros_like(sparse_mask)
        attention_mask[sparse_mask == 0] = -1e4  # 使用较大的负数而不是负无穷
        
        return attention_mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]（包含baseline观测）
            coords: 坐标信息 [B, 2, H, W]
            mask: 观测掩码 [B, 1, H, W]
            return_attention: 是否返回注意力权重
            
        Returns:
            增强特征 [B, embed_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # 输入投影
        x_proj = self.input_proj(x)  # [B, embed_dim, H, W]
        
        # 提取baseline观测（假设第一个通道是观测数据）
        if C > 0:
            baseline_obs = x[:, :1, :, :]  # [B, 1, H, W]
        else:
            baseline_obs = torch.zeros(B, 1, H, W, device=x.device)
        
        # 如果没有提供coords和mask，从输入中提取
        if coords is None and C >= 3:
            coords = x[:, 1:3, :, :]  # 假设第2-3通道是坐标
        if mask is None and C >= 4:
            mask = x[:, 3:4, :, :]   # 假设第4通道是掩码
        
        # 特征编码
        features = [x_proj]
        
        # 传感器位置编码
        if baseline_obs is not None:
            sensor_feat = self.sensor_embedding(baseline_obs)
            features.append(sensor_feat)
        
        # 坐标编码
        if coords is not None:
            coord_feat = self.coord_embedding(coords)
            features.append(coord_feat)
        
        # 掩码编码
        if mask is not None:
            mask_feat = self.mask_embedding(mask)
            features.append(mask_feat)
        
        # 特征融合
        fused_features = torch.cat(features, dim=1)  # [B, total_dim, H, W]
        fused_features = self.feature_fusion(fused_features)  # [B, embed_dim, H, W]
        
        # 稀疏自注意力
        residual = fused_features
        
        # 归一化
        norm_features = self.norm1(fused_features)
        
        # QKV投影
        qkv = self.qkv_proj(norm_features)  # [B, embed_dim*3, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # [B, embed_dim, H, W]
        
        # 转换为多头格式 [B, num_heads, head_dim, H, W]
        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k = k.view(B, self.num_heads, self.head_dim, H, W)
        v = v.view(B, self.num_heads, self.head_dim, H, W)
        
        # 计算注意力分数
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # 使用全局注意力（简化实现）
        q_flat = q.view(B, self.num_heads, self.head_dim, H * W)
        k_flat = k.view(B, self.num_heads, self.head_dim, H * W)
        v_flat = v.view(B, self.num_heads, self.head_dim, H * W)
        
        # 计算注意力
        attn = torch.einsum('bhdi,bhdj->bhij', q_flat, k_flat) * scale  # [B, num_heads, H*W, H*W]
        
        # 应用稀疏掩码
        if self.use_sparse_bias and mask is not None:
            sparse_mask = self._create_sparse_attention_mask(mask)
            attn = attn + sparse_mask.unsqueeze(1)  # [B, num_heads, H*W, H*W]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力到值
        out_flat = torch.einsum('bhij,bhdj->bhdi', attn, v_flat)  # [B, num_heads, head_dim, H*W]
        attn_out = out_flat.reshape(B, self.num_heads * self.head_dim, H, W)  # [B, embed_dim, H, W]
        
        # 输出投影
        attn_out = self.out_proj(attn_out)
        
        # 残差连接
        x = residual + attn_out
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        # 输出投影
        output = self.output_proj(x)
        
        if return_attention:
            return output, attn if 'attn' in locals() else None
        
        return output
    
    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """窗口分割"""
        B, num_heads, head_dim, H, W = x.shape
        x = x.view(B, num_heads, head_dim, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous().reshape(
            B * (H // window_size) * (W // window_size), num_heads, window_size * window_size, head_dim
        )
        return windows
    
    def _window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """窗口合并"""
        B_num_windows, num_heads, window_size_sq, head_dim = windows.shape
        B = B_num_windows // ((H // window_size) * (W // window_size))
        
        windows = windows.view(B, H // window_size, W // window_size, num_heads, window_size, window_size, head_dim)
        x = windows.permute(0, 3, 1, 4, 2, 5, 6).contiguous().reshape(B, num_heads, head_dim, H, W)
        return x
    
    def _compute_window_attention(
        self, 
        q_windows: torch.Tensor, 
        k_windows: torch.Tensor, 
        v_windows: torch.Tensor,
        scale: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算窗口内注意力"""
        B_windows, num_heads, window_size_sq, head_dim = q_windows.shape
        
        # 计算注意力分数
        attn = torch.einsum('bhni,bhnj->bhnj', q_windows, k_windows) * scale  # [B_windows, num_heads, head_dim, head_dim]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到值
        out = torch.einsum('bhnj,bhnk->bhnk', attn, v_windows)  # [B_windows, num_heads, head_dim, head_dim]
        
        return out


class SparseSwinUNet(nn.Module):
    """集成稀疏注意力编码的Swin-UNet
    
    在标准SwinUNet前添加稀疏注意力编码头，专门处理稀疏观测数据
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int = 256,
        embed_dim: int = 96,
        sparse_encoder_config: Optional[dict] = None,
        swin_unet_config: Optional[dict] = None
    ):
        super().__init__()
        
        # 稀疏注意力编码器配置
        if sparse_encoder_config is None:
            sparse_encoder_config = {
                'embed_dim': 256,
                'num_heads': 8,
                'sensor_dim': 128,
                'coord_dim': 64,
                'mask_dim': 32,
                'dropout': 0.1,
                'use_sparse_bias': True
            }
        
        # SwinUNet配置
        if swin_unet_config is None:
            swin_unet_config = {
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 8,
                'mlp_ratio': 4.0
            }
        
        # 稀疏注意力编码头
        self.sparse_encoder = SparseAttentionEncoder(
            in_channels=in_channels,
            **sparse_encoder_config
        )
        
        # SwinUNet主体（使用编码后的特征）
        from .swin_unet import SwinUNet
        self.swin_unet = SwinUNet(
            in_channels=sparse_encoder_config['embed_dim'],
            out_channels=out_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            **swin_unet_config
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]（包含baseline + coords + mask）
            **kwargs: 额外参数（coords, mask等）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 稀疏注意力编码
        sparse_features = self.sparse_encoder(x, **kwargs)
        
        # SwinUNet处理
        output = self.swin_unet(sparse_features)
        
        return output