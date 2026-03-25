"""
经典U-Net基线模型（稳定可训练版本）

实现标准的U-Net架构，用作基线对比模型。
遵循统一接口：forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# =========================================================
# Utils
# =========================================================
def _align_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Align src spatial size to ref by symmetric pad (if smaller) or center-crop (if larger).
    This makes the model robust to odd H/W and multi-stage pooling/upsampling size drift.

    src: [B,C,Hs,Ws], ref: [B,C,Hr,Wr]
    """
    hs, ws = src.shape[-2], src.shape[-1]
    hr, wr = ref.shape[-2], ref.shape[-1]

    # pad if needed
    pad_y = hr - hs
    pad_x = wr - ws
    if pad_y > 0 or pad_x > 0:
        src = F.pad(
            src,
            [
                max(pad_x // 2, 0),
                max(pad_x - pad_x // 2, 0),
                max(pad_y // 2, 0),
                max(pad_y - pad_y // 2, 0),
            ],
        )

    # crop if needed
    hs, ws = src.shape[-2], src.shape[-1]
    if hs > hr:
        y0 = (hs - hr) // 2
        src = src[:, :, y0 : y0 + hr, :]
    if ws > wr:
        x0 = (ws - wr) // 2
        src = src[:, :, :, x0 : x0 + wr]

    return src


# =========================================================
# Blocks
# =========================================================
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2，含可选 Dropout"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    """下采样：MaxPool2d -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    上采样模块：
    - bilinear=True : Upsample -> 1x1 reduce -> concat(skip) -> DoubleConv
    - bilinear=False: ConvTranspose2d -> concat(skip) -> DoubleConv
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)

        # 对齐空间尺寸，避免奇数尺寸导致的 1 像素误差
        x = _align_like(x, skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出 1x1 卷积"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =========================================================
# Model
# =========================================================
@register_model(name="unet", aliases=["UNet", "unet_baseline"])
class UNet(BaseModel):
    """
    经典 U-Net Baseline（可扩展深度）

    features: 每个 encoder stage 的通道数，例如 [64,128,256,512]
    结构：
        inc -> down1 -> down2 -> ... -> down(L-1) -> bottleneck(down) -> up(L) -> out

    注意：
    - 默认会额外加一个 bottleneck Down（再池化一次），这与经典 U-Net 的 /16 bottleneck 更一致
    - 如果你希望 bottleneck 不再额外池化，可把 self.bottleneck 改为 DoubleConv 即可
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        features: list[int] | None = None,
        bilinear: bool = True,
        dropout: float = 0.0,
        final_activation: str | None = None,  # None | "tanh" | "sigmoid"
        **kwargs,
    ):
        # 兼容 Hydra/旧字段
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", kwargs.pop("in_chans", 1))
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", kwargs.pop("num_classes", 1))
        if img_size is None:
            img_size = kwargs.get("img_size", 128)

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if features is None:
            features = [64, 128, 256, 512]
        if len(features) < 2:
            raise ValueError("features length must be >= 2, e.g. [64,128,256,512].")

        self.features = [int(v) for v in features]
        self.bilinear = bool(bilinear)
        self.dropout = float(dropout)

        # Encoder
        self.inc = DoubleConv(self.in_channels, self.features[0], dropout=self.dropout)

        self.down_blocks = nn.ModuleList()
        for i in range(1, len(self.features)):
            self.down_blocks.append(
                Down(self.features[i - 1], self.features[i], dropout=self.dropout)
            )

        # Bottleneck: 再下采样一次（经典 U-Net 的 bottleneck 更常见）
        # bilinear 版本为了节省显存，常用 factor=2 缩减 bottleneck 通道
        factor = 2 if self.bilinear else 1
        self.bottleneck_channels = (self.features[-1] * 2) // factor
        self.bottleneck = Down(
            self.features[-1], self.bottleneck_channels, dropout=self.dropout
        )

        # Decoder: 对应 features 反向逐级上采样（总共 len(features) 次 up）
        self.up_blocks = nn.ModuleList()
        in_ch = self.bottleneck_channels

        # skip 从 encoder 最深层开始：features[-1], features[-2], ..., features[0]
        for skip_ch in reversed(self.features):
            out_ch = skip_ch
            self.up_blocks.append(
                Up(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    bilinear=self.bilinear,
                    dropout=self.dropout,
                )
            )
            in_ch = out_ch

        self.outc = OutConv(self.features[0], self.out_channels)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        # 稳健初始化：conv 用 kaiming，BN 用 1/0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        # Encoder feature maps for skip connections
        skips: list[torch.Tensor] = []

        x0 = self.inc(x)  # level 0
        skips.append(x0)

        xi = x0
        for down in self.down_blocks:
            xi = down(xi)
            skips.append(xi)  # levels 1..L-1

        # Bottleneck (extra down)
        xb = self.bottleneck(xi)

        # Decoder: use skips from deepest to shallowest
        xd = xb
        # skips currently: [x0, x1, ..., x_{L-1}] ; we consume reversed(skips)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            xd = up(xd, skip)

        y = self.outc(xd)
        y = self.final_activation(y)
        return y

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回 encoder 侧各层特征（用于可视化/调试）"""
        self.eval()
        with torch.no_grad():
            feats: list[torch.Tensor] = []
            x0 = self.inc(x)
            feats.append(x0)
            xi = x0
            for down in self.down_blocks:
                xi = down(xi)
                feats.append(xi)
            return feats

    def freeze_encoder(self) -> None:
        """冻结 encoder + bottleneck（如果你只想冻结 encoder，不冻结 bottleneck，可自行拆开）"""
        for p in self.inc.parameters():
            p.requires_grad = False
        for blk in self.down_blocks:
            for p in blk.parameters():
                p.requires_grad = False
        for p in self.bottleneck.parameters():
            p.requires_grad = False

    def compute_flops(self, input_shape: tuple[int, ...] = None) -> int:
        """
        简化 FLOPs 估算（用于相对对比，不用于精确论文统计）
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)

        B, Cin, H, W = input_shape
        flops = 0

        # inc: two 3x3 conv
        c0 = self.features[0]
        flops += (Cin * c0 * 9 + c0 * c0 * 9) * H * W * 2

        # down blocks
        h, w = H, W
        prev = c0
        for ch in self.features[1:]:
            h //= 2
            w //= 2
            flops += (prev * ch * 9 + ch * ch * 9) * h * w * 2
            prev = ch

        # bottleneck down
        h //= 2
        w //= 2
        bott = self.bottleneck_channels
        flops += (prev * bott * 9 + bott * bott * 9) * h * w * 2

        # decoder (rough): symmetric conv cost (upsample cost ignored)
        # 每个 up 后做一次 DoubleConv(out+skip -> out)
        # 这里用近似：两次 3x3 conv
        for skip_ch in reversed(self.features):
            h *= 2
            w *= 2
            out_ch = skip_ch
            in_cat = out_ch + skip_ch
            flops += (in_cat * out_ch * 9 + out_ch * out_ch * 9) * h * w * 2

        # out 1x1
        flops += self.features[0] * self.out_channels * H * W

        self._flops = int(flops * B)
        return self._flops


def create_unet(**kwargs) -> UNet:
    """工厂函数（可选）"""
    return UNet(**kwargs)
