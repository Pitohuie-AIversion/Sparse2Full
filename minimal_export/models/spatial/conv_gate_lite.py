import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model


class ConvGateBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()  # Added activation
        self.pw = nn.Conv2d(dim, dim, 1)
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.dw(self.norm(x))
        y = self.act(y)  # Added activation
        y = self.pw(y)
        return x + self.beta * y + self.gamma * y


@register_model(name="conv_gate_lite", aliases=["ConvGateLite", "NAFNetLite"])
class ConvGateLite(BaseModel):
    """
    Lightweight Convolutional Gated Network.
    Previously named NAFNetLite, but renamed and fixed to include non-linearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        embed_dim: int = 64,
        depth: int = 8,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **kwargs,
        )
        self.stem = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvGateBlock(embed_dim) for _ in range(depth)])
        self.head = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.blocks(y)
        y = self.head(y)
        return y
