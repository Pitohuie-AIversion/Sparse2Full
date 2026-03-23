"""
U-NO / UNO (U-shaped Neural Operator) 模型实现（按统一口径）

面向 PDEBench 稀疏观测重建的 Operator Learning baseline：
- Lift -> 多尺度 Encoder-Decoder（U 形）
- 每个尺度用 SpectralConv2d (FNO-style) + 1x1 pointwise 作为算子核
- 下采样：stride=2 conv；上采样：bilinear + conv
- 可选：输出加 input residual（in/out 通道相同默认开启）

Reference:
    Rahman, Ross, Azizzadenesheli, "U-NO: U-shaped Neural Operators", arXiv:2204.11127 (2022, v1; 2023 v3).
    https://arxiv.org/abs/2204.11127
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Padding helpers (match Restormer style)
# -------------------------
def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    mode = "reflect" if (h > 1 and w > 1) else "constant"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, pad_h, pad_w


def _unpad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    h = x.shape[-2] - pad_h
    w = x.shape[-1] - pad_w
    return x[..., :h, :w]


# -------------------------
# Spectral Conv (FNO-style)
# -------------------------
class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution (rFFT2) used in FNO-family.
    Keeps only low-frequency modes (modes1, modes2).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (in_channels * out_channels)
        # two sets of weights for positive/negative height frequencies (standard FNO trick)
        self.weight1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (B, in, Hm, Wm) x (in, out, Hm, Wm) -> (B, out, Hm, Wm)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        """
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C_in, H, W//2+1]

        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        # top-left corner
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weight1[:, :, :m1, :m2])
        # bottom-left corner (negative frequencies in height)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weight2[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x


class FourierBlock2d(nn.Module):
    """SpectralConv2d + pointwise conv (1x1) + norm + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.spectral = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x) + self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------
# Down / Up
# -------------------------
class Down(nn.Module):
    """Stride-2 conv downsample + activation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Up(nn.Module):
    """Bilinear upsample + conv + activation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.act(self.conv(x))


# -------------------------
# UNO main model
# -------------------------
@register_model(name="UNO", aliases=["uno", "u-no", "UNeuralOperator"])
class UNO(BaseModel):
    """
    UNO / U-NO-style U-shaped neural operator.

    forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]

    Default (PDEBench 128x128 reasonable):
        width=64, modes=(16,16), blocks=(2,2,2,2)
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        blocks: List[int] = (2, 2, 2, 2),
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

        if isinstance(blocks, tuple):
            blocks = list(blocks)
        assert len(blocks) == 4, "UNO expects 4 levels: blocks length must be 4"

        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.blocks = blocks

        if add_input_residual is None:
            self.add_input_residual = (self.in_channels == self.out_channels)
        else:
            self.add_input_residual = bool(add_input_residual)

        # lift
        self.lift = nn.Conv2d(self.in_channels, self.width, kernel_size=3, padding=1, bias=True)

        # encoder level 1/2/3 + latent (level4)
        self.enc1 = nn.Sequential(*[
            FourierBlock2d(self.width, self.width, self.modes1, self.modes2) for _ in range(blocks[0])
        ])
        self.down1 = Down(self.width, self.width * 2)

        self.enc2 = nn.Sequential(*[
            FourierBlock2d(self.width * 2, self.width * 2, self.modes1, self.modes2) for _ in range(blocks[1])
        ])
        self.down2 = Down(self.width * 2, self.width * 4)

        self.enc3 = nn.Sequential(*[
            FourierBlock2d(self.width * 4, self.width * 4, self.modes1, self.modes2) for _ in range(blocks[2])
        ])
        self.down3 = Down(self.width * 4, self.width * 8)

        self.latent = nn.Sequential(*[
            FourierBlock2d(self.width * 8, self.width * 8, self.modes1, self.modes2) for _ in range(blocks[3])
        ])

        # decoder: up + concat skip + reduce + blocks
        self.up3 = Up(self.width * 8, self.width * 4)
        self.red3 = nn.Conv2d(self.width * 8, self.width * 4, kernel_size=1, bias=True)
        self.dec3 = nn.Sequential(*[
            FourierBlock2d(self.width * 4, self.width * 4, self.modes1, self.modes2) for _ in range(blocks[2])
        ])

        self.up2 = Up(self.width * 4, self.width * 2)
        self.red2 = nn.Conv2d(self.width * 4, self.width * 2, kernel_size=1, bias=True)
        self.dec2 = nn.Sequential(*[
            FourierBlock2d(self.width * 2, self.width * 2, self.modes1, self.modes2) for _ in range(blocks[1])
        ])

        self.up1 = Up(self.width * 2, self.width)
        self.red1 = nn.Conv2d(self.width * 2, self.width, kernel_size=1, bias=True)
        self.dec1 = nn.Sequential(*[
            FourierBlock2d(self.width, self.width, self.modes1, self.modes2) for _ in range(blocks[0])
        ])

        # head
        self.proj = nn.Conv2d(self.width, self.out_channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # minimal numeric safety
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x
        # 3 times downsample => require multiple of 8
        x, pad_h, pad_w = _pad_to_multiple(x, multiple=8)

        x = self.lift(x)

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.latent(x)

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.red3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.red2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.red1(x)
        x = self.dec1(x)

        out = self.proj(x)
        out = _unpad(out, pad_h, pad_w)

        if self.add_input_residual and (out.shape[1] == inp.shape[1]):
            out = out + inp

        return out

    def get_model_info(self) -> dict:
        return {
            "name": "UNO",
            "type": "NeuralOperator",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "width": self.width,
            "modes1": self.modes1,
            "modes2": self.modes2,
            "blocks": self.blocks,
            "add_input_residual": self.add_input_residual,
        }


# alias
UNONet = UNO
