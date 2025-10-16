"""混合模型架构

实现Hybrid模型：Attention ∥ FNO ∥ UNet的混合架构
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseModel


class HybridModel(BaseModel):
    """混合模型架构
    
    结合Attention、FNO和UNet的优势：
    - Attention分支：处理长程依赖
    - FNO分支：处理频域特征
    - UNet分支：处理局部特征
    - 特征融合：多分支特征融合
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 256,
        # 分支配置
        use_attention_branch: bool = True,
        use_fno_branch: bool = True,
        use_unet_branch: bool = True,
        # Attention分支参数
        attn_embed_dim: int = 256,
        attn_num_heads: int = 8,
        attn_num_layers: int = 6,
        attn_window_size: int = 8,
        # FNO分支参数
        fno_modes: int = 16,
        fno_width: int = 64,
        fno_num_layers: int = 4,
        # UNet分支参数
        unet_base_channels: int = 64,
        unet_num_layers: int = 4,
        # 融合参数
        fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
        fusion_channels: int = 256,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.use_attention_branch = use_attention_branch
        self.use_fno_branch = use_fno_branch
        self.use_unet_branch = use_unet_branch
        self.fusion_method = fusion_method
        
        # 输入投影层，修复通道数不匹配问题
        self.input_proj = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
        
        # 分支网络
        self.branches = nn.ModuleDict()
        branch_outputs = []
        
        if use_attention_branch:
            self.branches['attention'] = AttentionBranch(
                in_channels=fusion_channels,
                embed_dim=attn_embed_dim,
                num_heads=attn_num_heads,
                num_layers=attn_num_layers,
                window_size=attn_window_size,
                img_size=img_size
            )
            branch_outputs.append(attn_embed_dim)
        
        if use_fno_branch:
            self.branches['fno'] = FNOBranch(
                in_channels=fusion_channels,
                modes=fno_modes,
                width=fno_width,
                num_layers=fno_num_layers
            )
            branch_outputs.append(fno_width)
        
        if use_unet_branch:
            self.branches['unet'] = UNetBranch(
                in_channels=fusion_channels,
                base_channels=unet_base_channels,
                num_layers=unet_num_layers
            )
            branch_outputs.append(unet_base_channels)
        
        # 特征融合
        if fusion_method == 'concat':
            fusion_in_channels = sum(branch_outputs)
        elif fusion_method == 'add':
            # 确保所有分支输出通道数相同
            fusion_in_channels = branch_outputs[0]
            self.branch_align = nn.ModuleDict()
            for i, (name, out_ch) in enumerate(zip(self.branches.keys(), branch_outputs)):
                if out_ch != fusion_in_channels:
                    self.branch_align[name] = nn.Conv2d(out_ch, fusion_in_channels, kernel_size=1)
        elif fusion_method == 'attention':
            fusion_in_channels = fusion_channels
            self.fusion_attention = CrossBranchAttention(
                branch_channels=branch_outputs,
                fusion_channels=fusion_channels
            )
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv2d(fusion_in_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 输入投影
        x = self.input_proj(x)  # [B, fusion_channels, H, W]
        
        # 分支前向传播
        branch_outputs = {}
        for name, branch in self.branches.items():
            branch_outputs[name] = branch(x)
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused = torch.cat(list(branch_outputs.values()), dim=1)
        elif self.fusion_method == 'add':
            aligned_outputs = []
            for name, output in branch_outputs.items():
                if name in self.branch_align:
                    output = self.branch_align[name](output)
                aligned_outputs.append(output)
            fused = torch.stack(aligned_outputs, dim=0).sum(dim=0)
        elif self.fusion_method == 'attention':
            fused = self.fusion_attention(branch_outputs)
        
        # 输出
        output = self.output_head(fused)
        
        return output


class AttentionBranch(nn.Module):
    """注意力分支"""
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        window_size: int = 8,
        img_size: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.img_size = img_size
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, img_size, img_size) * 0.02)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            WindowAttentionLayer(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2
            )
            for i in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 输入投影和位置编码
        x = self.input_proj(x)
        
        # 调整位置编码尺寸
        if (H, W) != (self.img_size, self.img_size):
            pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


class WindowAttentionLayer(nn.Module):
    """窗口注意力层"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # 注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMultiHeadAttention(dim, num_heads, window_size)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 转换为序列格式
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 窗口移位
        if self.shift_size > 0:
            x = x.view(B, H, W, C)
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x.view(B, H * W, C)
        
        # 注意力
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = shortcut + x
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        # 反向窗口移位
        if self.shift_size > 0:
            x = x.view(B, H, W, C)
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x.view(B, H * W, C)
        
        # 转换回图像格式
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


class WindowMultiHeadAttention(nn.Module):
    """窗口多头注意力"""
    
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # 分割为窗口
        x = x.view(B, H, W, C)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        x = x.view(B, H_pad // self.window_size, self.window_size, 
                  W_pad // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        
        # 多头注意力
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        
        # 合并窗口
        x = x.view(B, H_pad // self.window_size, W_pad // self.window_size,
                  self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, C)
        
        # 移除padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        
        x = x.view(B, H * W, C)
        
        return x


class FNOBranch(nn.Module):
    """FNO分支"""
    
    def __init__(
        self,
        in_channels: int,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        
        # FNO层
        self.fno_layers = nn.ModuleList([
            FNOLayer(width, modes) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(width, width, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入投影
        x = self.input_proj(x)
        
        # FNO层
        for layer in self.fno_layers:
            x = layer(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


class FNOLayer(nn.Module):
    """FNO层"""
    
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = channels
        self.modes = modes
        
        # 傅里叶权重
        self.weights1 = nn.Parameter(torch.view_as_complex(
            torch.randn(channels, channels, modes, modes, 2) * 0.02
        ))
        self.weights2 = nn.Parameter(torch.view_as_complex(
            torch.randn(channels, channels, modes, modes, 2) * 0.02
        ))
        
        # 局部卷积
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 激活函数
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x)
        
        # 频域卷积 - 确保使用ComplexFloat类型
        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64)
        weights_complex = self.weights1.to(dtype=torch.complex64)
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes, :self.modes].to(dtype=torch.complex64), 
            weights_complex
        )
        
        # 逆傅里叶变换
        x1 = torch.fft.irfft2(out_ft, s=(H, W))
        
        # 局部卷积
        x2 = self.conv(x)
        
        # 残差连接和激活
        x = self.activation(x1 + x2 + x)
        
        return x


class UNetBranch(nn.Module):
    """UNet分支"""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_layers)]
        
        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
            if i < num_layers - 1:
                self.pools.append(nn.MaxPool2d(2))
        
        # 解码器
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[num_layers-i], channels[num_layers-i-1], 
                                 kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(channels[num_layers-i-1] * 2, channels[num_layers-i-1], 
                             kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[num_layers-i-1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[num_layers-i-1], channels[num_layers-i-1], 
                             kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[num_layers-i-1]),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.pools):
                skip_connections.append(x)
                x = self.pools[i](x)
        
        # 解码器
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = skip_connections[-(i+1)]
            
            # 调整尺寸
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return x


class CrossBranchAttention(nn.Module):
    """跨分支注意力融合"""
    
    def __init__(self, branch_channels: List[int], fusion_channels: int):
        super().__init__()
        self.branch_channels = branch_channels
        self.fusion_channels = fusion_channels
        
        # 分支投影
        self.branch_projs = nn.ModuleList([
            nn.Conv2d(ch, fusion_channels, kernel_size=1) 
            for ch in branch_channels
        ])
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Conv2d(fusion_channels * len(branch_channels), fusion_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, len(branch_channels), kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, branch_outputs: dict) -> torch.Tensor:
        # 投影到相同维度
        projected = []
        for i, (name, output) in enumerate(branch_outputs.items()):
            projected.append(self.branch_projs[i](output))
        
        # 计算注意力权重
        concat_features = torch.cat(projected, dim=1)
        attention_weights = self.attention(concat_features)  # [B, num_branches, H, W]
        
        # 加权融合
        fused = torch.zeros_like(projected[0])
        for i, proj in enumerate(projected):
            weight = attention_weights[:, i:i+1]  # [B, 1, H, W]
            fused = fused + weight * proj
        
        return fused