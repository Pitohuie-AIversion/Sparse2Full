import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model

"""
CNNAttnLite (cnn_attn_lite)
==========================

What this model IS
- A lightweight CNN using:
  (1) Depthwise Separable Convolution (DW 3x3 + PW 1x1) for efficiency
  (2) Squeeze-and-Excitation style Channel Attention (global pooling -> bottleneck -> gate)
  (3) Residual connections + a simple pointwise "FFN-like" conv stack

What this model is NOT
- It is NOT Restormer (Transformer-based restoration model with MDTA + GDFN).
  Therefore, do NOT use "Restormer" in aliases or class name.

References / Origins
- Depthwise Separable Convolution (efficiency pattern), popularized by MobileNet:
  Andrew G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
  arXiv:1704.04861

- Channel Attention (Squeeze-and-Excitation, SE):
  Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, "Squeeze-and-Excitation Networks"
  arXiv:1709.01507

- Restormer (for disambiguation only; NOT implemented here):
  Syed Waqas Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration"
  arXiv:2111.09881, official: https://github.com/swz30/Restormer

Naming/Registry policy
- Canonical registry name: "cnn_attn_lite"
- Aliases should ONLY include names that truly refer to this architecture.
  Avoid adding unrelated aliases (e.g., "RestormerLite") to prevent name–architecture mismatch.
"""


class ChannelAttention(nn.Module):
    """
    SE-style Channel Attention.
    Source: Squeeze-and-Excitation Networks (Hu et al., arXiv:1709.01507)
    Mechanism: GAP -> bottleneck -> gate (sigmoid) -> channel-wise reweighting.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        r = max(1, dim // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Conv2d(dim, r, 1),  # squeeze to r channels
            nn.GELU(),
            nn.Conv2d(r, dim, 1),  # excite back to dim channels
            nn.Sigmoid(),  # channel gate
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class DepthwisePointwise(nn.Module):
    """
    Depthwise Separable Convolution.
    Source: MobileNet (Howard et al., arXiv:1704.04861)
    Structure: DW conv (groups=dim) + PW 1x1 conv.
    """

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # depthwise 3x3
        self.pw = nn.Conv2d(dim, dim, 1)  # pointwise 1x1
        self.act = nn.GELU()

    def forward(self, x):
        y = self.dw(x)
        y = self.act(y)
        y = self.pw(y)
        return y


class CNNAttnBlock(nn.Module):
    """
    Residual CNN block with:
    - BatchNorm2d normalization (CNN typical)
    - Depthwise Separable Conv operator (MobileNet-style)
    - SE-style Channel Attention
    - Pointwise "FFN-like" conv stack (1x1 -> GELU -> 1x1), inspired by MLP/FFN pattern
      (Note: not a Transformer FFN; here it is purely 1x1 conv in spatial feature maps.)
    """

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.op = DepthwisePointwise(dim)
        self.ca = ChannelAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        # Residual branch 1: DW+PW conv + channel attention
        y = self.op(self.norm1(x))
        y = self.ca(y)
        x = x + y

        # Residual branch 2: pointwise conv "FFN-like"
        z = self.ff(self.norm2(x))
        x = x + z
        return x


@register_model(
    name="cnn_attn_lite",
    # 注意：aliases 里不需要重复写 "cnn_attn_lite"（name 已经覆盖了）
    # 只保留对人更友好的别名即可
    aliases=["CNNAttnLite"],
)
class CNNAttnLite(BaseModel):
    """
    Lightweight CNN + Channel Attention model.

    Design intention:
    - "Lite": small parameter count and fast inference
    - CNN backbone with SE-style channel attention and depthwise separable conv

    Naming note:
    - This is NOT Restormer (no self-attention / MDTA / GDFN).
    - If you previously used "RestormerLite" to refer to this model, migrate that name
      via a deprecation shim elsewhere (recommended), rather than keeping a misleading alias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        embed_dim: int = 48,
        depth: int = 6,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **kwargs,
        )
        self.stem = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[CNNAttnBlock(embed_dim) for _ in range(depth)])
        self.head = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.blocks(y)
        y = self.head(y)
        return y
