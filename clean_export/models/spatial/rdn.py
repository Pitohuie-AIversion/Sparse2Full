"""
RDN (Residual Dense Network) 模型实现（用于稀疏观测重建/图像复原基线）

RDN 最初用于图像超分辨率，但其“密集连接 + 局部残差 + 全局特征融合”的 CNN 结构
对 PDEBench 这类网格场重建同样非常稳健，尤其适合作为 RCAN/EDSR 之外的强 CNN 基线。

统一接口：
    forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
支持可选：add_input_residual（当 in/out 通道一致时，可做 out += x）

Reference:
    - Zhang et al., "Residual Dense Network for Image Super-Resolution", CVPR 2018.
      https://arxiv.org/abs/1802.08797
    - Official (architecture reference): https://github.com/yulunzhang/RDN
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Core blocks
# -------------------------
class DenseLayer(nn.Module):
    """
    Dense layer used in RDB:
        x -> Conv(3x3) -> ReLU -> concat([x, out])
    Reference: RDN, CVPR 2018 (Residual Dense Block design).
    """

    def __init__(self, in_channels: int, growth_rate: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv(x))
        return torch.cat([x, out], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB):
        - L dense layers
        - Local Feature Fusion (1x1 conv) to compress channels back to base channels
        - Local residual connection with optional residual scaling

    Reference:
        Zhang et al., "Residual Dense Network for Image Super-Resolution", CVPR 2018.
    """

    def __init__(
        self,
        base_channels: int,
        num_layers: int = 6,
        growth_rate: int = 32,
        bias: bool = True,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.residual_scale = residual_scale

        layers = []
        ch = base_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate, bias=bias))
            ch += growth_rate

        self.dense_layers = nn.Sequential(*layers)

        # Local Feature Fusion: 1x1 to compress channels back to base_channels
        self.lff = nn.Conv2d(ch, base_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense_layers(x)
        out = self.lff(out)
        return x + out * self.residual_scale


class Upsampler(nn.Module):
    """
    PixelShuffle upsampler (optional, for SR use-cases).
    For PDEBench sparse->full reconstruction, set scale=1 (default), and this block is bypassed.

    Reference:
        - PixelShuffle: Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient
          Sub-Pixel Convolutional Neural Network", CVPR 2016.
    """

    def __init__(self, channels: int, scale: int, bias: bool = True):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError("scale must be one of {2,3,4} for PixelShuffle upsampling.")

        modules = []
        if scale in (2, 4):
            # scale = 2^n
            n = 1 if scale == 2 else 2
            for _ in range(n):
                modules += [
                    nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(channels, channels * 9, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True),
            ]

        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# -------------------------
# RDN main model
# -------------------------
@register_model(name="RDN", aliases=["rdn"])
class RDN(BaseModel):
    """
    RDN for image-to-image regression / reconstruction.

    Typical SR setting in the paper:
        - base_channels(G0)=64, growth_rate(G)=32, num_blocks(D)=16, num_layers(C)=8

    For PDEBench 128x128 sparse->full reconstruction (recommended baseline):
        - base_channels=64, growth_rate=32
        - num_blocks=8~12, num_layers=6~8  (trade-off with speed)
        - scale=1 (same-resolution reconstruction)

    Unified interface:
        forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        base_channels: int = 64,
        growth_rate: int = 32,
        num_blocks: int = 8,
        num_layers: int = 6,
        residual_scale: float = 0.2,
        scale: int = 1,
        bias: bool = True,
        add_input_residual: Optional[bool] = None,
        **kwargs,
    ):
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", kwargs.pop("in_chans", 1))
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", kwargs.pop("num_classes", 1))
        if img_size is None:
            img_size = kwargs.get("img_size", 128)
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.base_channels = int(base_channels)
        self.growth_rate = int(growth_rate)
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        self.residual_scale = float(residual_scale)
        self.scale = int(scale)
        self.bias = bool(bias)

        if add_input_residual is None:
            self.add_input_residual = (self.in_channels == self.out_channels and self.scale == 1)
        else:
            self.add_input_residual = bool(add_input_residual)

        # Shallow feature extraction
        # Reference: RDN paper shallow feature extraction (two convs).
        self.sfe1 = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.sfe2 = nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # Residual Dense Blocks (RDBs)
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(
                base_channels=self.base_channels,
                num_layers=self.num_layers,
                growth_rate=self.growth_rate,
                bias=bias,
                residual_scale=self.residual_scale,
            )
            for _ in range(self.num_blocks)
        ])

        # Global Feature Fusion (GFF)
        # concat all RDB outputs -> 1x1 -> 3x3
        self.gff_1x1 = nn.Conv2d(self.base_channels * self.num_blocks, self.base_channels, kernel_size=1, bias=bias)
        self.gff_3x3 = nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # Optional upsampling (for SR); for PDEBench keep scale=1
        if self.scale == 1:
            self.upsampler = nn.Identity()
        else:
            self.upsampler = Upsampler(self.base_channels, scale=self.scale, bias=bias)

        # Reconstruction head
        self.recon = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self._init_weights()

    def _init_weights(self):
        # Conservative init (matches your other models' preference)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H*scale, W*scale]  (scale=1 => same resolution)
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x

        f1 = self.sfe1(x)
        f2 = self.sfe2(f1)

        rdb_outs = []
        out = f2
        for rdb in self.rdbs:
            out = rdb(out)
            rdb_outs.append(out)

        # Global Feature Fusion + global residual
        out = torch.cat(rdb_outs, dim=1)
        out = self.gff_1x1(out)
        out = self.gff_3x3(out)
        out = out + f1  # global residual (paper’s global feature fusion + shallow feature)

        out = self.upsampler(out)
        out = self.recon(out)

        if self.add_input_residual and inp.shape[1] == out.shape[1] and self.scale == 1:
            out = out + inp

        return out

    def get_model_info(self) -> dict:
        return {
            "name": "RDN",
            "type": "CNN",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "base_channels": self.base_channels,
            "growth_rate": self.growth_rate,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "scale": self.scale,
            "add_input_residual": self.add_input_residual,
        }
