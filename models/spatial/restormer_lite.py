import torch
import torch.nn as nn

from ..base import BaseModel


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        r = max(1, dim // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, r, 1),
            nn.GELU(),
            nn.Conv2d(r, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class DepthwisePointwise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.dw(x)
        y = self.act(y)
        y = self.pw(y)
        return y


class RestormerBlock(nn.Module):
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
        y = self.op(self.norm1(x))
        y = self.ca(y)
        x = x + y
        z = self.ff(self.norm2(x))
        x = x + z
        return x


class RestormerLite(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, img_size: int, embed_dim: int = 48, depth: int = 6):
        super().__init__(in_channels=in_channels, out_channels=out_channels, img_size=img_size)
        self.stem = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[RestormerBlock(embed_dim) for _ in range(depth)])
        self.head = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.blocks(y)
        y = self.head(y)
        return y
