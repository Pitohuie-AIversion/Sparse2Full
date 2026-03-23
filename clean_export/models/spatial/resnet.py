import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


@register_model(name="resnet_lite", aliases=["ResNetLite", "SwinIRLite"])
class ResNetLite(BaseModel):
    """
    Lightweight ResNet for restoration.
    Previously named SwinIRLite, but renamed to reflect actual architecture (CNN-based).
    """
    def __init__(self, in_channels: int, out_channels: int, img_size: int, embed_dim: int = 64, depth: int = 6, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, img_size=img_size, **kwargs)
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim) for _ in range(depth)])
        self.recon = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shallow(x)
        y = self.blocks(y)
        y = self.recon(y)
        return y
