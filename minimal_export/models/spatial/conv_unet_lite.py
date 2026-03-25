import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model


class EncoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)


@register_model(name="conv_unet_lite", aliases=["ConvUNetLite", "UformerLite"])
class ConvUNetLite(BaseModel):
    """
    Lightweight Convolutional UNet.
    Previously named UformerLite, but renamed to reflect actual architecture (CNN-based).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        embed_dim: int = 64,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **kwargs,
        )
        self.enc1 = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.block1 = EncoderBlock(embed_dim)
        self.pool = nn.MaxPool2d(2)
        self.block2 = EncoderBlock(embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DecoderBlock(embed_dim)
        self.head = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc1(x)
        y = self.block1(y)
        y = self.pool(y)
        y = self.block2(y)
        y = self.up(y)
        y = self.dec1(y)
        y = self.head(y)
        return y
