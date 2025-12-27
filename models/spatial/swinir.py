import torch
import torch.nn as nn

from ..base import BaseModel


class ResidualSwinIRBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class SwinIRLite(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, img_size: int, embed_dim: int = 64, depth: int = 6):
        super().__init__(in_channels=in_channels, out_channels=out_channels, img_size=img_size)
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualSwinIRBlock(embed_dim) for _ in range(depth)])
        self.recon = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shallow(x)
        y = self.blocks(y)
        y = self.recon(y)
        return y

