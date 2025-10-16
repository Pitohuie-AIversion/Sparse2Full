"""SegFormer/UNetFormer模型

结合Transformer和U-Net的混合架构，使用自注意力机制进行特征建模。
SegFormer使用分层Transformer编码器，UNetFormer在U-Net中集成Transformer块。

Reference:
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    https://arxiv.org/abs/2105.15203
    
    UNetFormer: A UNet-like transformer for efficient semantic segmentation
    https://arxiv.org/abs/2109.08417
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class PatchEmbed(nn.Module):
    """图像到补丁嵌入
    
    将输入图像分割成补丁并嵌入到高维空间
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class MLP(nn.Module):
    """多层感知机"""
    
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


class Attention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        qk_scale: Optional[float] = None, 
        attn_drop: float = 0.0, 
        proj_drop: float = 0.0,
        sr_ratio: int = 1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 空间缩减（用于降低计算复杂度）
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key和Value（可能经过空间缩减）
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = False, 
        qk_scale: Optional[float] = None, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        drop_path: float = 0.0, 
        act_layer: nn.Module = nn.GELU, 
        norm_layer: nn.Module = nn.LayerNorm,
        sr_ratio: int = 1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop, 
            sr_ratio=sr_ratio
        )
        
        # 随机深度（Stochastic Depth）
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """重叠补丁嵌入
    
    使用重叠的卷积进行补丁嵌入，保留更多空间信息
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SegFormerEncoder(nn.Module):
    """SegFormer编码器
    
    分层Transformer编码器，逐步降低分辨率并增加通道数
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dims: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1]
    ):
        super().__init__()
        
        self.depths = depths
        self.embed_dims = embed_dims
        
        # 补丁嵌入层
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        # Transformer块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则
        cur = 0
        
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        
        cur += depths[1]
        # 修复注意力头数问题：确保embed_dims[2]能被num_heads[2]整除
        adjusted_num_heads_2 = 4 if embed_dims[2] % num_heads[2] != 0 else num_heads[2]
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=adjusted_num_heads_2, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(-1, H, W, self.embed_dims[0]).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(-1, H, W, self.embed_dims[1]).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(-1, H, W, self.embed_dims[2]).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(-1, H, W, self.embed_dims[3]).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        return outs


class MLPDecoder(nn.Module):
    """MLP解码器
    
    使用MLP进行特征融合和上采样
    """
    
    def __init__(self, in_channels: List[int], embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # 线性投影层
        self.linear_c4 = nn.Linear(in_channels[3], embedding_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embedding_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embedding_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embedding_dim)
        
        self.linear_fuse = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4.permute(0, 2, 3, 1).reshape(n, -1, c4.shape[1])).permute(0, 2, 1).reshape(n, -1, h, w)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.permute(0, 2, 3, 1).reshape(n, -1, c3.shape[1])).permute(0, 2, 1).reshape(n, -1, *c3.shape[2:])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.permute(0, 2, 3, 1).reshape(n, -1, c2.shape[1])).permute(0, 2, 1).reshape(n, -1, *c2.shape[2:])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.permute(0, 2, 3, 1).reshape(n, -1, c1.shape[1])).permute(0, 2, 1).reshape(n, -1, *c1.shape[2:])
        
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1).permute(0, 2, 3, 1).reshape(n, -1, _c1.shape[1] * 4)).permute(0, 2, 1).reshape(n, -1, *c1.shape[2:])
        x = self.dropout(_c)
        
        return x


class SegFormerUNetFormer(BaseModel):
    """SegFormer/UNetFormer模型
    
    结合Transformer和U-Net的混合架构：
    - 编码器：分层Transformer编码器（SegFormer风格）
    - 解码器：MLP解码器进行特征融合
    
    这种设计结合了Transformer的全局建模能力和高效的解码策略。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        embed_dims: 各阶段的嵌入维度，默认[64, 128, 256, 512]
        num_heads: 各阶段的注意力头数，默认[1, 2, 4, 8]
        mlp_ratios: MLP扩展比例，默认[4, 4, 4, 4]
        depths: 各阶段的Transformer块数量，默认[3, 4, 6, 3]
        sr_ratios: 空间缩减比例，默认[8, 4, 2, 1]
        decoder_dim: 解码器嵌入维度，默认256
        dropout: Dropout概率，默认0.1
        **kwargs: 其他参数
    
    Examples:
        >>> model = SegFormerUNetFormer(in_channels=3, out_channels=1, img_size=256)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        embed_dims: List[int] = None,
        num_heads: List[int] = None,
        mlp_ratios: List[float] = None,
        depths: List[int] = None,
        sr_ratios: List[int] = None,
        decoder_dim: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        # 默认配置
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depths is None:
            depths = [3, 4, 6, 3]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.decoder_dim = decoder_dim
        self.dropout = dropout
        
        # SegFormer编码器
        self.encoder = SegFormerEncoder(
            img_size=img_size,
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.1
        )
        
        # MLP解码器
        self.decoder = MLPDecoder(
            in_channels=embed_dims,
            embedding_dim=decoder_dim,
            dropout=dropout
        )
        
        # 输出头
        self.head = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（忽略，保持接口一致性）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 保存原始输入尺寸
        original_size = x.shape[2:]
        
        # 编码器
        features = self.encoder(x)
        
        # 解码器
        x = self.decoder(features)
        
        # 上采样到原始尺寸
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # 输出头
        x = self.head(x)
        
        return x
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs（简化估算）
        
        Args:
            input_shape: 输入形状，默认为(1, in_channels, img_size, img_size)
            
        Returns:
            FLOPs数量
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        batch_size, _, height, width = input_shape
        
        # 编码器FLOPs（简化估算）
        encoder_flops = 0
        h, w = height, width
        
        for i, (embed_dim, depth, num_head) in enumerate(zip(self.embed_dims, self.depths, self.num_heads)):
            if i > 0:
                h, w = h // 2, w // 2
            
            # 补丁嵌入
            if i == 0:
                patch_flops = self.in_channels * embed_dim * 7 * 7 * h * w
            else:
                patch_flops = self.embed_dims[i-1] * embed_dim * 3 * 3 * h * w
            
            # Transformer块
            seq_len = h * w
            attn_flops = depth * (
                # QKV投影
                3 * embed_dim * embed_dim * seq_len +
                # 注意力计算
                num_head * seq_len * seq_len * (embed_dim // num_head) +
                # 输出投影
                embed_dim * embed_dim * seq_len +
                # MLP
                2 * embed_dim * (embed_dim * 4) * seq_len
            )
            
            encoder_flops += patch_flops + attn_flops
        
        # 解码器FLOPs
        decoder_flops = 0
        for embed_dim in self.embed_dims:
            decoder_flops += embed_dim * self.decoder_dim * (height // 4) * (width // 4)
        
        # 特征融合
        fusion_flops = self.decoder_dim * 4 * self.decoder_dim * height * width
        
        # 输出头
        head_flops = self.decoder_dim * self.out_channels * height * width
        
        total_flops = encoder_flops + decoder_flops + fusion_flops + head_flops
        self._flops = total_flops * batch_size
        return self._flops
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取注意力图（用于可视化）
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            各阶段的注意力图列表
        """
        # 这里简化实现，实际需要修改Attention类来返回注意力权重
        attention_maps = []
        
        # 通过编码器获取特征
        features = self.encoder(x)
        
        # 返回特征图作为注意力的代理（简化版本）
        for feat in features:
            # 计算通道注意力作为代理
            attn = torch.mean(feat, dim=1, keepdim=True)  # [B, 1, H, W]
            attention_maps.append(attn)
        
        return attention_maps


# 别名，保持向后兼容
SegFormer = SegFormerUNetFormer
UNetFormer = SegFormerUNetFormer