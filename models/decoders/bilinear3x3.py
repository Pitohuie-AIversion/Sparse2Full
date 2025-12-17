import torch
import torch.nn as nn
import torch.nn.functional as F

class Bilinear3x3Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x

