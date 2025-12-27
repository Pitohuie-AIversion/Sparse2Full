"""
PartialConv-UNet 模型实现（mask-aware baseline）

Partial Convolution 用于处理缺测/空洞区域：
- 卷积只对 mask=1 的位置参与计算
- 根据卷积核内有效像素数做归一化
- mask 也随层更新（全无有效像素则输出置零）

该模型非常适合 PDEBench 的稀疏观测重建口径：
- x 输入可为“稀疏观测填零后的张量”
- mask 显式指示哪些网格点是观测值（1）/缺测（0）
- 若训练管线暂时不给 mask，本实现会默认 mask=全1（之后再改口径即可）

Reference:
    Liu et al., "Image Inpainting for Irregular Holes Using Partial Convolutions", ECCV 2018.
    https://arxiv.org/abs/1804.07723
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Partial Convolution
# -------------------------
class PartialConv2d(nn.Module):
    """
    PartialConv2d:
    out = Conv(x * m) normalized by valid_count in each receptive field.
    Supports mask shape [B,1,H,W] or [B,C,H,W].
    Returns (out, updated_mask).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]

        # 用于计算 valid_count 的 ones kernel（注册为 buffer，不参与训练）
        # mask 统一压到 1 通道后计算
        self.register_buffer("weight_mask", torch.ones(1, 1, self.kernel_size, self.kernel_size))

    @staticmethod
    def _to_1ch_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() != 4:
            raise ValueError("mask must be a 4D tensor [B,1,H,W] or [B,C,H,W].")
        if mask.shape[1] == 1:
            return (mask > 0).float()
        # 多通道 mask：只要任一通道有效就算有效
        return (mask.sum(dim=1, keepdim=True) > 0).float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask1 = self._to_1ch_mask(mask).to(dtype=x.dtype, device=x.device)

        # masked input
        x_masked = x * mask1

        # standard conv on masked input
        out = self.conv(x_masked)

        # valid count per location: conv(mask, ones)
        with torch.no_grad():
            valid = F.conv2d(
                mask1,
                self.weight_mask.to(device=x.device, dtype=x.dtype),
                bias=None,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )
            new_mask = (valid > 0).to(dtype=x.dtype)

        # normalize
        if self.conv.bias is not None:
            bias_view = self.conv.bias.view(1, -1, 1, 1)
            out = out - bias_view

        # 避免除零
        denom = torch.where(valid > 0, valid, torch.ones_like(valid))
        out = out / denom

        if self.conv.bias is not None:
            out = out + bias_view

        # 对无有效像素区域置零
        out = out * new_mask

        return out, new_mask


# -------------------------
# UNet blocks (partialconv)
# -------------------------
class PConvDoubleConv(nn.Module):
    """(PartialConv -> ReLU) x2"""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.pconv1 = PartialConv2d(in_ch, out_ch, 3, 1, 1, bias=bias)
        self.pconv2 = PartialConv2d(out_ch, out_ch, 3, 1, 1, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, m = self.pconv1(x, m)
        x = self.act(x)
        x, m = self.pconv2(x, m)
        x = self.act(x)
        return x, m


class PConvDown(nn.Module):
    """Down: MaxPool + DoubleConv (mask 同步 maxpool)"""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = PConvDoubleConv(in_ch, out_ch, bias=bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(x)
        m = self.pool(m)
        return self.conv(x, m)


class PConvUp(nn.Module):
    """Up: Upsample + concat + DoubleConv（mask 最近邻上采样，concat 后取 OR）"""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, bias: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.upm = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            # 为简单起见：特征用反卷积，mask 仍用 nearest
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.upm = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv = PConvDoubleConv(in_ch, out_ch, bias=bias)

    @staticmethod
    def _pad_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

    def forward(
        self,
        x1: torch.Tensor, m1: torch.Tensor,
        x2: torch.Tensor, m2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.up(x1)
        m1 = self.upm(m1)

        x1 = self._pad_like(x1, x2)
        m1 = self._pad_like(m1, m2)

        x = torch.cat([x2, x1], dim=1)
        m = torch.maximum(m2, m1)  # OR

        return self.conv(x, m)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------
# Model
# -------------------------
@register_model(name="PartialConvUNet", aliases=["pconv_unet", "partialconv_unet"])
class PartialConvUNet(BaseModel):
    """
    PartialConv-UNet baseline.

    Unified interface:
        forward(x[B,C_in,H,W], mask=... optional) -> y[B,C_out,H,W]
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        features: Optional[List[int]] = None,
        bilinear: bool = True,
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

        if features is None:
            features = [64, 128, 256, 512]

        self.features = features
        self.bilinear = bilinear
        self.bias = bias

        if add_input_residual is None:
            self.add_input_residual = (in_channels == out_channels)
        else:
            self.add_input_residual = bool(add_input_residual)

        self.inc = PConvDoubleConv(in_channels, features[0], bias=bias)
        self.down1 = PConvDown(features[0], features[1], bias=bias)
        self.down2 = PConvDown(features[1], features[2], bias=bias)
        self.down3 = PConvDown(features[2], features[3], bias=bias)

        factor = 2 if bilinear else 1
        self.down4 = PConvDown(features[3], features[3] * 2 // factor, bias=bias)
        bott = features[3] * 2 // factor

        self.up1 = PConvUp(bott + features[3], features[3] // factor, bilinear=bilinear, bias=bias)
        self.up2 = PConvUp((features[3] // factor) + features[2], features[2] // factor, bilinear=bilinear, bias=bias)
        self.up3 = PConvUp((features[2] // factor) + features[1], features[1] // factor, bilinear=bilinear, bias=bias)
        self.up4 = PConvUp((features[1] // factor) + features[0], features[0], bilinear=bilinear, bias=bias)

        self.outc = OutConv(features[0], out_channels, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = kwargs.get("mask", None)
        if mask is None:
            # 默认全有效；之后你把“观测 mask”真正接进来即可
            mask = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(device=x.device, dtype=x.dtype)

        inp = x

        x1, m1 = self.inc(x, mask)
        x2, m2 = self.down1(x1, m1)
        x3, m3 = self.down2(x2, m2)
        x4, m4 = self.down3(x3, m3)
        x5, m5 = self.down4(x4, m4)

        x, m = self.up1(x5, m5, x4, m4)
        x, m = self.up2(x,  m,  x3, m3)
        x, m = self.up3(x,  m,  x2, m2)
        x, m = self.up4(x,  m,  x1, m1)

        out = self.outc(x)

        if self.add_input_residual and inp.shape[1] == out.shape[1]:
            out = out + inp

        return out


# alias
PConvUNet = PartialConvUNet
