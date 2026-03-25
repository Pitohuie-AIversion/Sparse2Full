"""
U-FNO瓶颈模型（U-Net + FNO bottleneck）

结合U-Net和FNO的混合架构，在U-Net的瓶颈层使用FNO进行全局建模。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]

Reference:
    U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow
    https://doi.org/10.1016/j.advwatres.2022.104180
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model

# =========================================================
# SpectralConv2d import with fallback
# =========================================================
try:
    from .fno2d import SpectralConv2d  # type: ignore
except Exception:

    class SpectralConv2d(nn.Module):
        """
        FNO-style 2D spectral convolution (fallback implementation).

        x: [B, C_in, H, W]
        FFT -> multiply low-frequency modes -> IFFT
        """

        def __init__(
            self, in_channels: int, out_channels: int, modes1: int, modes2: int
        ):
            super().__init__()
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.modes1 = int(modes1)
            self.modes2 = int(modes2)

            # complex weights for top-left and bottom-left frequency blocks
            scale = 1 / (self.in_channels * self.out_channels)
            self.weights1 = nn.Parameter(
                scale
                * torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    dtype=torch.cfloat,
                )
            )
            self.weights2 = nn.Parameter(
                scale
                * torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    dtype=torch.cfloat,
                )
            )

        @staticmethod
        def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a: [B, inC, m1, m2], b: [inC, outC, m1, m2] -> [B, outC, m1, m2]
            return torch.einsum("bixy,ioxy->boxy", a, b)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]

            out_ft = torch.zeros(
                B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
            )

            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)

            # top-left
            out_ft[:, :, :m1, :m2] = self.compl_mul2d(
                x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
            )
            # bottom-left
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
                x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
            )

            x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
            return x


# =========================================================
# U-Net blocks
# =========================================================
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ):
        super().__init__()
        mid = out_channels if mid_channels is None else int(mid_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downsample: MaxPool -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """
    Upsample + skip concat + DoubleConv

    We make channel math explicit:
      x1: from deeper layer (in_ch)
      x2: skip feature      (skip_ch)
      concat -> in_ch + skip_ch -> out_ch
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bool(bilinear)
        self.in_ch = int(in_ch)
        self.skip_ch = int(skip_ch)
        self.out_ch = int(out_ch)

        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # bilinear does not change channels
            self.reduce = nn.Identity()
        else:
            # transposed conv can also keep channels; typical U-Net halves channels here,
            # but we keep it simple and stable.
            self.up = nn.ConvTranspose2d(
                self.in_ch, self.in_ch, kernel_size=2, stride=2
            )
            self.reduce = nn.Identity()

        self.conv = DoubleConv(self.in_ch + self.skip_ch, self.out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # upsample x1
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        # pad to match skip size (robust to odd sizes)
        diff_y = x2.size(-2) - x1.size(-2)
        diff_x = x2.size(-1) - x1.size(-1)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([x2, x1], dim=1)  # [B, skip_ch+in_ch, H, W]
        return self.conv(x)


# =========================================================
# FNO Bottleneck
# =========================================================
class FNOBottleneck(nn.Module):
    """
    FNO bottleneck:
      x -> SpectralConv2d + 1x1 Conv -> residual -> norm -> act
    """

    def __init__(
        self,
        channels: int,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 2,
        activation: str = "gelu",
        norm: str = "bn",
    ):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.n_layers = int(n_layers)

        act = activation.lower()
        if act == "gelu":
            self.act_fn = F.gelu
        elif act == "relu":
            self.act_fn = F.relu
        elif act == "tanh":
            self.act_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.spectral = nn.ModuleList(
            [
                SpectralConv2d(self.channels, self.channels, self.modes1, self.modes2)
                for _ in range(self.n_layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [
                nn.Conv2d(self.channels, self.channels, kernel_size=1)
                for _ in range(self.n_layers)
            ]
        )

        norm = norm.lower()
        if norm == "bn":
            self.norms = nn.ModuleList(
                [nn.BatchNorm2d(self.channels) for _ in range(self.n_layers)]
            )
        elif norm == "gn":
            self.norms = nn.ModuleList(
                [nn.GroupNorm(8, self.channels) for _ in range(self.n_layers)]
            )
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            x_f = self.spectral[i](x)
            x_p = self.pointwise[i](x)
            x = x + x_f + x_p
            x = self.norms[i](x)
            if i < self.n_layers - 1:
                x = self.act_fn(x)
        return x


# =========================================================
# U-FNO U-Net Model
# =========================================================
@register_model(name="ufno_unet", aliases=["UFNOUNet", "UFNOModel"])
class UFNOUNet(BaseModel):
    """
    U-FNO bottleneck U-Net.

    Encoder:
      inc -> down1 -> down2 -> down3 -> down4
    Bottleneck:
      FNO
    Decoder:
      up1 -> up2 -> up3 -> up4 -> outc
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        features: list[int] | None = None,
        fno_modes1: int = 16,
        fno_modes2: int = 16,
        fno_layers: int = 2,
        bilinear: bool = True,
        dropout: float = 0.0,
        fno_activation: str = "gelu",
        fno_norm: str = "bn",
        final_activation: str | None = None,  # None | "tanh" | "sigmoid"
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if features is None:
            features = [64, 128, 256, 512]
        if len(features) != 4:
            raise ValueError("features must have length 4, e.g. [64,128,256,512].")

        self.features = [int(v) for v in features]
        self.bilinear = bool(bilinear)
        self.dropout = float(dropout)

        f0, f1, f2, f3 = self.features

        # Encoder
        self.inc = DoubleConv(in_channels, f0)
        self.down1 = Down(f0, f1)
        self.down2 = Down(f1, f2)
        self.down3 = Down(f2, f3)

        # Bottleneck: typical U-Net uses 2*f3
        bottleneck_ch = f3 * 2
        self.down4 = Down(f3, bottleneck_ch)

        self.fno_bottleneck = FNOBottleneck(
            channels=bottleneck_ch,
            modes1=fno_modes1,
            modes2=fno_modes2,
            n_layers=fno_layers,
            activation=fno_activation,
            norm=fno_norm,
        )

        self.dropout_layer = (
            nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity()
        )

        # Decoder channel plan:
        # up1: in=bottleneck_ch,  skip=f3 -> out=f3
        # up2: in=f3,            skip=f2 -> out=f2
        # up3: in=f2,            skip=f1 -> out=f1
        # up4: in=f1,            skip=f0 -> out=f0
        self.up1 = Up(
            in_ch=bottleneck_ch, skip_ch=f3, out_ch=f3, bilinear=self.bilinear
        )
        self.up2 = Up(in_ch=f3, skip_ch=f2, out_ch=f2, bilinear=self.bilinear)
        self.up3 = Up(in_ch=f2, skip_ch=f1, out_ch=f1, bilinear=self.bilinear)
        self.up4 = Up(in_ch=f1, skip_ch=f0, out_ch=f0, bilinear=self.bilinear)

        self.outc = nn.Conv2d(f0, out_channels, kernel_size=1)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        # Encoder
        x1 = self.inc(x)  # f0
        x2 = self.down1(x1)  # f1
        x3 = self.down2(x2)  # f2
        x4 = self.down3(x3)  # f3
        x5 = self.down4(x4)  # 2*f3

        # FNO bottleneck
        x5 = self.fno_bottleneck(x5)
        x5 = self.dropout_layer(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        y = self.outc(x)
        y = self.final_activation(y)
        return y

    # Optional utilities (keep minimal, avoid fragile FLOPs estimators)
    def freeze_encoder(self) -> None:
        for m in [self.inc, self.down1, self.down2, self.down3, self.down4]:
            for p in m.parameters():
                p.requires_grad = False

    def freeze_fno(self) -> None:
        for p in self.fno_bottleneck.parameters():
            p.requires_grad = False


# Backward compatible alias
UFNOModel = UFNOUNet
