"""Swin-UNet模型实现

基于Swin Transformer的UNet架构，支持可选的FNO瓶颈层。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from .base import BaseModel


class WindowAttention(nn.Module):
    """窗口多头自注意力机制"""
    
    def __init__(
        self, 
        dim: int, 
        window_size: Tuple[int, int], 
        num_heads: int, 
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None, 
        attn_drop: float = 0., 
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 获取相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        num_heads: int,
        window_size: int = 7, 
        shift_size: int = 0, 
        mlp_ratio: float = 4.,
        qkv_bias: bool = True, 
        qk_scale: Optional[float] = None, 
        drop: float = 0., 
        attn_drop: float = 0.,
        drop_path: float = 0., 
        act_layer: nn.Module = nn.GELU, 
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch合并层"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """Patch扩展层 - 对称于PatchMerging的上采样操作"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        B, L, C = x.shape
        assert C == 2 * self.dim, "expand dimension wrong"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """基础Swin Transformer层"""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        depth: int, 
        num_heads: int,
        window_size: int, 
        mlp_ratio: float = 4., 
        qkv_bias: bool = True, 
        qk_scale: Optional[float] = None,
        drop: float = 0., 
        attn_drop: float = 0., 
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm, 
        downsample: Optional[nn.Module] = None, 
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # patch合并层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP层"""
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.
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


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将特征图分割为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """将窗口合并回特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinUNetDecoder(nn.Module):
    """对称的Swin-UNet解码器 - 使用PatchExpanding和Swin Transformer块"""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        depths: List[int],
        num_heads: List[int],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_connections: bool = True,
        patches_resolution: Tuple[int, int] = (64, 64),
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.depths = depths
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.patches_resolution = patches_resolution
        
        # 计算解码器的分辨率序列（从小到大）
        num_layers = len(depths)
        self.decoder_resolutions = []
        for i in range(num_layers):
            # 解码器分辨率：从最小开始逐步增大
            scale = 2 ** (num_layers - 1 - i)
            res = (patches_resolution[0] // scale, patches_resolution[1] // scale)
            self.decoder_resolutions.append(res)
        
        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建解码器层
        self.decoder_layers = nn.ModuleList()
        
        # 从最深层开始构建解码器
        for i in range(num_layers):
            # 当前层的输入通道数（来自编码器或上一个解码器层）
            if i == 0:
                # 第一层：来自编码器最深层
                in_dim = encoder_channels[-1]
            else:
                # 后续层：来自上一个解码器层
                in_dim = decoder_channels[i-1] if i-1 < len(decoder_channels) else encoder_channels[-(i)]
            
            # 输出通道数
            out_dim = decoder_channels[i] if i < len(decoder_channels) else in_dim // 2
            
            # 跳跃连接的通道数
            if skip_connections and i > 0:  # 第一层（最深层）没有跳跃连接
                skip_idx = num_layers - 1 - i  # 对应的编码器层索引
                skip_dim = encoder_channels[skip_idx] if skip_idx >= 0 and skip_idx < len(encoder_channels) else 0
            else:
                skip_dim = 0
            
            # 如果有跳跃连接，需要融合层
            if skip_dim > 0:
                # 跳跃连接融合层
                fuse_layer = nn.Sequential(
                    nn.Linear(in_dim + skip_dim, out_dim, bias=False),
                    norm_layer(out_dim)
                )
            else:
                fuse_layer = None
            
            # Swin Transformer块
            swin_layer = BasicLayer(
                dim=out_dim,
                input_resolution=self.decoder_resolutions[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=None,  # 解码器不需要下采样
                use_checkpoint=use_checkpoint
            )
            
            # PatchExpanding层（除了最后一层）
            if i < num_layers - 1:
                patch_expand = PatchExpanding(
                    input_resolution=self.decoder_resolutions[i],
                    dim=out_dim,
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                patch_expand = None
            
            # 创建解码器层字典，只包含nn.Module对象
            layer_dict = nn.ModuleDict({
                'swin': swin_layer,
            })
            
            if fuse_layer is not None:
                layer_dict['fuse'] = fuse_layer
            if patch_expand is not None:
                layer_dict['expand'] = patch_expand
            
            # 将非Module属性存储为普通属性
            layer_dict.skip_dim = skip_dim
            layer_dict.out_dim = out_dim
            
            self.decoder_layers.append(layer_dict)
    
    def forward(self, x: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """解码器前向传播
        
        Args:
            x: 编码器输出 [B, N, C]
            skip_features: 跳跃连接特征列表（从浅到深）
            
        Returns:
            解码后的特征图 [B, C, H, W]
        """
        B, N, C = x.shape
        
        # 逐层解码
        for i, layer_dict in enumerate(self.decoder_layers):
            fuse_layer = layer_dict['fuse'] if 'fuse' in layer_dict else None
            swin_layer = layer_dict['swin']
            expand_layer = layer_dict['expand'] if 'expand' in layer_dict else None
            skip_dim = layer_dict.skip_dim
            
            # 处理跳跃连接
            if (skip_features is not None and 
                fuse_layer is not None and
                skip_dim > 0 and
                i > 0):  # 第一层（最深层）没有跳跃连接
                # 跳跃连接索引：对应的编码器层
                skip_idx = len(skip_features) - i
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip_feat = skip_features[skip_idx]
                    
                    # 确保跳跃连接特征的分辨率匹配
                    B_skip, N_skip, C_skip = skip_feat.shape
                    B_x, N_x, C_x = x.shape
                    
                    if N_skip != N_x:
                        # 需要调整跳跃连接特征的分辨率
                        H_skip = W_skip = int(math.sqrt(N_skip))
                        H_x = W_x = int(math.sqrt(N_x))
                        
                        # 转换为图像格式进行插值
                        skip_feat = skip_feat.transpose(1, 2).view(B_skip, C_skip, H_skip, W_skip)
                        skip_feat = F.interpolate(skip_feat, size=(H_x, W_x), mode='bilinear', align_corners=False)
                        skip_feat = skip_feat.view(B_skip, C_skip, -1).transpose(1, 2)  # [B, N_x, C_skip]
                    
                    # 融合跳跃连接
                    x = torch.cat([x, skip_feat], dim=-1)  # [B, N, C + skip_C]
                    x = fuse_layer(x)  # [B, N, out_dim]
            
            # Swin Transformer块
            x = swin_layer(x)
            
            # PatchExpanding（除了最后一层）
            if expand_layer is not None:
                x = expand_layer(x)
        
        # 转换为图像格式
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        # 确保输出尺寸与目标匹配
        target_size = self.patches_resolution
        if x.shape[-2:] != target_size:
            # 使用双线性插值调整到目标尺寸
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class SwinUNet(BaseModel):
    """完全对称的Swin-UNet模型
    
    基于Swin Transformer的UNet架构，编码器和解码器都使用Swin Transformer块。
    严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        # 解码器参数
        decoder_depths: Optional[List[int]] = None,
        decoder_num_heads: Optional[List[int]] = None,
        skip_connections: bool = True,
        # FNO瓶颈参数（可选）
        use_fno_bottleneck: bool = False,
        fno_modes: int = 16,
        # 最终激活函数
        final_activation: Optional[str] = None,  # None, 'tanh', 'sigmoid'
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        self.skip_connections = skip_connections
        self.use_fno_bottleneck = use_fno_bottleneck
        
        # 解码器参数（默认与编码器对称）
        if decoder_depths is None:
            decoder_depths = depths[::-1]  # 倒序
        if decoder_num_heads is None:
            decoder_num_heads = num_heads[::-1]  # 倒序
        
        self.decoder_depths = decoder_depths
        self.decoder_num_heads = decoder_num_heads
        
        # 分割图像为patch并嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建编码器层
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(depths) - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.encoder_layers.append(layer)

        self.norm = norm_layer(int(embed_dim * 2 ** (len(depths) - 1)))

        # FNO瓶颈层（可选）
        if use_fno_bottleneck:
            bottleneck_dim = int(embed_dim * 2 ** (len(depths) - 1))
            self.fno_bottleneck = FNOBottleneck(bottleneck_dim, fno_modes)
        else:
            self.fno_bottleneck = None

        # 对称的Swin-UNet解码器
        encoder_channels = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        # 解码器通道数：从最深层开始逐步减少
        decoder_channels = []
        for i in range(len(depths)):
            if i == 0:
                # 第一层：保持最深层的通道数
                decoder_channels.append(encoder_channels[-1])
            else:
                # 后续层：逐步减少通道数
                decoder_channels.append(encoder_channels[-(i+1)])
        
        # 最后一层输出通道数应该与输入图像通道数匹配
        decoder_channels[-1] = embed_dim
        
        self.decoder = SwinUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            depths=self.decoder_depths,
            num_heads=self.decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            skip_connections=skip_connections,
            patches_resolution=patches_resolution,
            use_checkpoint=use_checkpoint
        )

        # 最终输出层
        self.final_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        
        # 最终激活函数
        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        # 初始化权重
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """初始化权重 - 使用更保守的初始化策略"""
        if isinstance(m, nn.Linear):
            # 使用更小的标准差进行初始化，提高数值稳定性
            trunc_normal_(m.weight, std=.005)  # 进一步减少到0.005
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 使用更保守的Xavier初始化
            nn.init.xavier_uniform_(m.weight, gain=0.3)  # 进一步减少gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # 批归一化层的保守初始化
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（coords, mask等）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, N, embed_dim]
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 编码器前向传播，保存跳跃连接
        skip_connections = []
        for layer in self.encoder_layers:
            if self.skip_connections:
                # 保存当前特征用于跳跃连接
                skip_connections.append(x)
            x = layer(x)

        x = self.norm(x)  # [B, N, C]

        # FNO瓶颈层（可选）
        if self.fno_bottleneck is not None:
            x = self.fno_bottleneck(x)

        # 对称的Swin解码器
        x = self.decoder(x, skip_connections if self.skip_connections else None)

        # 最终输出层
        x = self.final_conv(x)
        
        # 确保输出尺寸与输入一致
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        x = self.final_activation(x)

        return x


class UNetDecoder(nn.Module):
    """UNet解码器"""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        skip_connections: bool = True,
        upsampling_mode: str = 'bilinear',
        patches_resolution: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.skip_connections = skip_connections
        self.upsampling_mode = upsampling_mode
        self.patches_resolution = patches_resolution
        
        # 构建解码器层
        self.decoder_blocks = nn.ModuleList()
        
        # 从最深层开始
        in_channels = encoder_channels[-1]
        
        for i, out_channels in enumerate(decoder_channels):
            # 跳跃连接的通道数 - 修正索引逻辑
            if skip_connections and i < len(encoder_channels) - 1:
                # 跳跃连接来自对应的编码器层（倒序）
                skip_idx = len(encoder_channels) - 2 - i
                skip_channels = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            else:
                skip_channels = 0
            
            # 调试信息
            print(f"Decoder layer {i}: in_channels={in_channels}, skip_channels={skip_channels}, out_channels={out_channels}")
            
            block = DecoderBlock(
                in_channels=in_channels,  # 上采样后的通道数
                out_channels=out_channels,
                skip_channels=skip_channels,  # 跳跃连接通道数
                upsampling_mode=upsampling_mode
            )
            self.decoder_blocks.append(block)
            in_channels = out_channels
    
    def forward(self, x: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """解码器前向传播
        
        Args:
            x: 编码器输出 [B, N, C]
            skip_features: 跳跃连接特征列表
            
        Returns:
            解码后的特征图 [B, C, H, W]
        """
        # 将序列格式转换为图像格式
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        print(f"Decoder input: {x.shape}")
        if skip_features is not None:
            print(f"Skip features shapes: {[sf.shape for sf in skip_features]}")
        
        # 逐层解码
        for i, decoder_block in enumerate(self.decoder_blocks):
            print(f"\nDecoder layer {i}:")
            print(f"  Input shape: {x.shape}")
            
            # 先上采样
            x = decoder_block.upsample(x)
            print(f"  After upsample: {x.shape}")
            
            # 添加跳跃连接
            skip_feat = None
            if (skip_features is not None and 
                self.skip_connections and 
                i < len(skip_features)):
                # 跳跃连接索引：从最新的开始（倒序）
                skip_idx = len(skip_features) - 1 - i
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip_feat = skip_features[skip_idx]
                    print(f"  Skip feature {skip_idx} shape: {skip_feat.shape}")
                    
                    # 将跳跃连接特征转换为图像格式
                    B_skip, N_skip, C_skip = skip_feat.shape
                    H_skip = W_skip = int(math.sqrt(N_skip))
                    skip_feat = skip_feat.transpose(1, 2).view(B_skip, C_skip, H_skip, W_skip)
                    print(f"  Skip feature reshaped: {skip_feat.shape}")
                    
                    # 调整尺寸匹配
                    if skip_feat.shape[-2:] != x.shape[-2:]:
                        skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                        print(f"  Skip feature after resize: {skip_feat.shape}")
                    
                    # 检查通道数是否匹配预期的skip_channels
                    expected_skip_channels = decoder_block.skip_channels
                    print(f"  Expected skip channels: {expected_skip_channels}, Actual: {C_skip}")
                    
                    # 如果通道数不匹配，使用1x1卷积调整
                    if C_skip != expected_skip_channels and expected_skip_channels > 0:
                        if not hasattr(decoder_block, 'skip_proj'):
                            decoder_block.skip_proj = nn.Conv2d(C_skip, expected_skip_channels, 1).to(skip_feat.device)
                            # 初始化权重
                            nn.init.xavier_uniform_(decoder_block.skip_proj.weight)
                            nn.init.zeros_(decoder_block.skip_proj.bias)
                        skip_feat = decoder_block.skip_proj(skip_feat)
                        print(f"  Skip feature after projection: {skip_feat.shape}")
                    
                    # 如果不需要跳跃连接，直接设为None
                    if expected_skip_channels == 0:
                        skip_feat = None
                        print(f"  Skip connection disabled for this layer")
            
            # 应用解码器块的卷积层
            if skip_feat is not None:
                # 拼接跳跃连接
                x = torch.cat([x, skip_feat], dim=1)
                print(f"  After concatenation: {x.shape}")
            
            # 直接调用卷积层，而不是整个decoder_block
            x = decoder_block.conv(x)
            print(f"  After conv: {x.shape}")
        
        # 确保输出尺寸与原始输入匹配
        target_size = self.patches_resolution
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class DecoderBlock(nn.Module):
    """解码器块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        upsampling_mode: str = 'bilinear'
    ):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.skip_channels = skip_channels
        
        # 上采样层
        if upsampling_mode == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        elif upsampling_mode == 'pixel_shuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
                nn.PixelShuffle(2)
            )
        else:  # bilinear
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 卷积层（考虑跳跃连接）
        conv_in_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：这个forward方法不应该被直接调用
        # 上采样和卷积应该在UNetDecoder中分别调用
        x = self.upsample(x)
        x = self.conv(x)
        return x


class FNOBottleneck(nn.Module):
    """FNO瓶颈层（可选）"""
    
    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.channels = channels
        self.modes = modes
        
        # 傅里叶权重
        self.weights1 = nn.Parameter(torch.view_as_complex(torch.randn(channels, channels, modes, modes, 2)))
        self.weights2 = nn.Parameter(torch.view_as_complex(torch.randn(channels, channels, modes, modes, 2)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FNO前向传播
        
        Args:
            x: 输入特征 [B, N, C]
            
        Returns:
            输出特征 [B, N, C]
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # 转换为图像格式
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # 频域卷积
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes, :self.modes], self.weights1
        )
        
        # IFFT
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        # 转换回序列格式
        x = x.view(B, C, -1).transpose(1, 2)
        
        return x