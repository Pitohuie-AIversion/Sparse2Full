"""
SwinFluidSR: 基于 Swin Transformer 与亚像素卷积的科学流体超分辨率模型

针对流体力学场（如 Rayleigh-Bénard 对流、湍流通道等）的超分辨率逆问题设计：
1. 直接低维输入（Direct Low-Resolution Input）：
   - 在原始低分辨率空间（如 32x32）进行特征提取与移位窗口自注意力，Token 计算量降为原来的 1/16；
2. 残差 Swin Transformer 模块 (RSTB)：
   - 结合多头窗口自注意力 (Window-MSA) 与跨窗口移位注意力 (Shifted Window-MSA)；
   - 具备局部残差与长跳跃连接 (Long Skip Connection)，保留深层多尺度物理拓扑；
3. 亚像素卷积上采样 (PixelShuffle)：
   - 数据驱动的特征空间重排展开，还原清晰锐利的细微涡旋与高频剪切边缘；
4. 全局物理残差学习 (Global Physical Residual)：
   - Pred = SubPixel(Features) + Bicubic(LR)，模型专注学习因退化丢失的高频微小物理细节。

References:
- SwinIR: Image Restoration Using Swin Transformer (ICCVW 2021)
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021)
- ESPCN: Real-Time Single Image Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (CVPR 2016)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model
from .swinir import (
    RSTB,
    PatchEmbed,
    PatchUnEmbed,
    _pad_to_window_size,
    _unpad,
    _to_int_list,
)


class UpsamplePixelShuffle(nn.Sequential):
    """亚像素卷积上采样模块 (PixelShuffle Upsampler)

    Args:
        scale (int): 上采样倍率 (例如 2, 4, 8)
        num_feat (int): 输入特征通道数
        out_feat (int): 输出通道数
    """

    def __init__(self, scale: int, num_feat: int, out_feat: int = 1):
        m = []
        if (scale & (scale - 1)) == 0:  # scale 是 2 的幂次 (2, 4, 8)
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1, bias=True))
                m.append(nn.PixelShuffle(2))
                m.append(nn.PReLU())
            m.append(nn.Conv2d(num_feat, out_feat, 3, 1, 1, bias=True))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1, bias=True))
            m.append(nn.PixelShuffle(3))
            m.append(nn.PReLU())
            m.append(nn.Conv2d(num_feat, out_feat, 3, 1, 1, bias=True))
        else:
            raise ValueError(f"Unsupported upscale factor: {scale}, must be 2, 3, 4, or 8.")
        super().__init__(*m)


@register_model(name="swin_fluid_sr", aliases=["SwinFluidSR", "swin_fluid_super_resolution"])
class SwinFluidSR(BaseModel):
    """
    SwinFluidSR: 面向 PDE 流场的高性能超分辨率模型

    接口规范：
        输入 x: [B, C_in, H_lr, W_lr] (例如 [B, 1, 32, 32])
        输出 y: [B, C_out, H_hr, W_hr] (例如 [B, 1, 128, 128])
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 32,
        upscale_factor: int = 4,
        embed_dim: int = 96,
        depths: Union[List[int], Tuple[int, ...]] = (6, 6, 6, 6),
        num_heads: Union[List[int], Tuple[int, ...], int] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        global_residual: bool = True,
        use_lowres_input: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, img_size=img_size, **kwargs)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if isinstance(img_size, (tuple, list)):
            self.img_size = int(img_size[0])
        else:
            self.img_size = int(img_size)
        self.upscale_factor = int(upscale_factor)
        self.embed_dim = int(embed_dim)
        self.window_size = int(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.global_residual = bool(global_residual)
        self.use_lowres_input = bool(use_lowres_input)

        self.depths = _to_int_list(depths, default=[6, 6, 6, 6])
        self.num_layers = len(self.depths)
        self.num_heads = _to_int_list(num_heads, default=[6, 6, 6, 6])
        if len(self.num_heads) == 1 and self.num_layers > 1:
            self.num_heads = self.num_heads * self.num_layers

        # 1. 浅层特征提取卷积 (Shallow Feature Extraction)
        self.conv_first = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        # 2. Token <-> Feature Map 转换器
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

        # 3. 随机深度调度
        total_blocks = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # 4. 深层特征提取：多组 RSTB 模块
        self.layers = nn.ModuleList()
        dp_index = 0
        for i_layer in range(self.num_layers):
            depth_i = self.depths[i_layer]
            heads_i = self.num_heads[i_layer]
            self.layers.append(
                RSTB(
                    dim=self.embed_dim,
                    input_resolution=(self.img_size, self.img_size),
                    depth=depth_i,
                    num_heads=heads_i,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[dp_index : dp_index + depth_i],
                )
            )
            dp_index += depth_i

        self.norm = nn.LayerNorm(self.embed_dim)

        # 5. 主干后卷积 (Conv after deep feature extraction)
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        # 6. 亚像素上采样重构头 (Sub-pixel Convolution Upsampler)
        self.upsample = UpsamplePixelShuffle(
            scale=self.upscale_factor,
            num_feat=self.embed_dim,
            out_feat=self.out_channels,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播：
            x: [B, C_in, H_lr, W_lr] 低分辨率物理场输入
            返回: [B, C_out, H_hr, W_hr] 高分辨率重建全场 (H_hr = H_lr * upscale_factor)
        """
        H_in, W_in = x.shape[-2], x.shape[-1]
        
        # 1. 计算全局物理残差基底 (Global Physical Baseline)
        if self.global_residual:
            target_h = H_in * self.upscale_factor
            target_w = W_in * self.upscale_factor
            in_c = x.shape[1]
            out_c = self.out_channels
            c_match = min(in_c, out_c)

            x_interp_src = x[:, :c_match]
            orig_dtype = x.dtype
            # 兼容老版本/特定加速卡下 FP16 bicubic 不支持或精度溢出
            if x_interp_src.dtype in (torch.float16, torch.bfloat16):
                base_c = F.interpolate(
                    x_interp_src.float(),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                ).to(orig_dtype)
            else:
                base_c = F.interpolate(
                    x_interp_src,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )

            if c_match < out_c:
                pad_c = torch.zeros(
                    x.shape[0], out_c - c_match, target_h, target_w,
                    device=x.device, dtype=orig_dtype
                )
                base_interp = torch.cat([base_c, pad_c], dim=1)
            else:
                base_interp = base_c
        else:
            base_interp = None

        # 2. 窗口 Pad 检查 (确保能被 window_size 整除)
        x_pad, pad = _pad_to_window_size(x, self.window_size)
        _, _, Hp, Wp = x_pad.shape
        x_size = (Hp, Wp)

        # 3. 浅层特征提取
        x_feat = self.conv_first(x_pad)
        x_res = x_feat  # 浅层残差

        # 4. Token 化并送入深层 RSTB 模块
        tokens = self.patch_embed(x_feat)
        for layer in self.layers:
            tokens = layer(tokens, x_size)
        tokens = self.norm(tokens)

        # 5. 反 Token 化与长跳跃残差融合
        deep_feat = self.patch_unembed(tokens, x_size)
        deep_feat = self.conv_after_body(deep_feat) + x_res

        # 6. 反向 Unpad (如果在低维空间有 pad)
        deep_feat = _unpad(deep_feat, pad)

        # 7. 亚像素卷积解码上采样 (PixelShuffle -> 目标高分辨率空间)
        high_freq_field = self.upsample(deep_feat)

        # 8. 全局残差加和：高频物理细节 + 宏观低频基底
        if base_interp is not None:
            out = high_freq_field + base_interp
        else:
            out = high_freq_field
        return out

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "arch": "SwinFluidSR",
                "upscale_factor": self.upscale_factor,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "global_residual": self.global_residual,
                "use_lowres_input": self.use_lowres_input,
                "num_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            }
        )
        return info


def create_swin_fluid_sr(**kwargs) -> SwinFluidSR:
    return SwinFluidSR(**kwargs)
