"""
U-Net++模型实现（Nested U-Net）

实现嵌套U-Net架构（UNet++），通过密集跳跃连接与（可选）深度监督提升性能。
严格遵循统一接口：forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]

Reference:
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    https://arxiv.org/abs/1807.10165
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
    This makes the network robust to odd H/W or multiple pool/upsample stages.

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
class ConvBlock(nn.Module):
    """卷积块：(Conv3x3 -> BN -> ReLU) * 2，含可选Dropout"""

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


class UpSample(nn.Module):
    """上采样模块：ConvTranspose2d，输出通道可控，并做尺寸对齐"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _align_like(x, ref)
        return x


# =========================================================
# UNet++
# =========================================================
@register_model(name="unetpp", aliases=["UNetPlusPlus", "unet_plus_plus", "unet++"])
class UNetPlusPlus(BaseModel):
    """
    UNet++ (Nested U-Net)

    features: 每个stage的通道数，例如 [32, 64, 128, 256]
    deep_supervision:
      - False: 输出 x_{0,depth}
      - True : 输出多个 head 的融合（均值），仍保持单输出张量以兼容统一接口
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        features: list[int] | None = None,
        deep_supervision: bool = False,
        dropout: float = 0.0,
        final_activation: str | None = None,  # None | "tanh" | "sigmoid"
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if features is None:
            features = [32, 64, 128, 256]
        if len(features) < 2:
            raise ValueError("features length must be >= 2, e.g. [32,64,128,256].")

        self.features = [int(v) for v in features]
        self.num_layers = len(self.features)
        self.deep_supervision = bool(deep_supervision)
        self.dropout = float(dropout)

        # Encoder conv blocks for x_{i,0}
        self.encoders = nn.ModuleList()
        self.encoders.append(
            ConvBlock(in_channels, self.features[0], dropout=self.dropout)
        )
        for i in range(1, self.num_layers):
            self.encoders.append(
                ConvBlock(self.features[i - 1], self.features[i], dropout=self.dropout)
            )

        # Pools between encoder levels
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(self.num_layers - 1)]
        )

        # Up-samplers: from level (i+1) to level i, channel map features[i+1] -> features[i]
        self.ups = nn.ModuleList(
            [
                UpSample(self.features[i + 1], self.features[i])
                for i in range(self.num_layers - 1)
            ]
        )

        # Nested conv blocks for x_{i,j}, j>=1
        # x_{i,j} input channels = features[i] * (j + 1)   (concat of: x_{i,0..j-1} plus up(x_{i+1,j-1}))
        self.nested_convs = nn.ModuleDict()
        for j in range(1, self.num_layers):  # depth of nesting
            for i in range(0, self.num_layers - j):  # spatial level
                in_ch = self.features[i] * (j + 1)
                out_ch = self.features[i]
                self.nested_convs[self._key(i, j)] = ConvBlock(
                    in_ch, out_ch, dropout=self.dropout
                )

        # Output heads: deep supervision uses x_{0,1..depth}
        if self.deep_supervision:
            self.heads = nn.ModuleList(
                [
                    nn.Conv2d(self.features[0], out_channels, kernel_size=1)
                    for _ in range(1, self.num_layers)  # x_{0,1} ... x_{0,num_layers-1}
                ]
            )
        else:
            self.head = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        self._init_weights()

    @staticmethod
    def _key(i: int, j: int) -> str:
        return f"{i}_{j}"

    def _init_weights(self) -> None:
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
        feat: dict[str, torch.Tensor] = {}

        # -------------------------
        # Encoder: x_{i,0}
        # -------------------------
        feat[self._key(0, 0)] = self.encoders[0](x)
        for i in range(1, self.num_layers):
            pooled = self.pools[i - 1](feat[self._key(i - 1, 0)])
            feat[self._key(i, 0)] = self.encoders[i](pooled)

        # -------------------------
        # Nested decoder: x_{i,j}, j>=1
        # UNet++ rule:
        # x_{i,j} = Conv( concat( x_{i,0}, x_{i,1}, ..., x_{i,j-1}, up(x_{i+1,j-1}) ) )
        # -------------------------
        for j in range(1, self.num_layers):
            for i in range(0, self.num_layers - j):
                # upsample x_{i+1, j-1} to level i
                up = self.ups[i](
                    feat[self._key(i + 1, j - 1)], ref=feat[self._key(i, 0)]
                )

                # concat all previous nodes at same level i: x_{i,0..j-1}, plus up
                cat_list = [feat[self._key(i, k)] for k in range(0, j)]
                cat_list.append(up)
                concat = torch.cat(cat_list, dim=1)

                feat[self._key(i, j)] = self.nested_convs[self._key(i, j)](concat)

        # -------------------------
        # Output
        # -------------------------
        if self.deep_supervision:
            # Use x_{0,1..depth} heads and fuse (mean) to keep single output tensor
            outs = []
            max_j = self.num_layers - 1
            for j in range(1, max_j + 1):
                node = feat[self._key(0, j)]
                outs.append(self.heads[j - 1](node))
            y = torch.stack(outs, dim=0).mean(dim=0)
        else:
            y = self.head(feat[self._key(0, self.num_layers - 1)])

        y = self.final_activation(y)
        return y

    # Optional: feature extraction for visualization/debug
    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            feat: dict[str, torch.Tensor] = {}
            feat[self._key(0, 0)] = self.encoders[0](x)
            for i in range(1, self.num_layers):
                pooled = self.pools[i - 1](feat[self._key(i - 1, 0)])
                feat[self._key(i, 0)] = self.encoders[i](pooled)

            for j in range(1, self.num_layers):
                for i in range(0, self.num_layers - j):
                    up = self.ups[i](
                        feat[self._key(i + 1, j - 1)], ref=feat[self._key(i, 0)]
                    )
                    cat_list = [feat[self._key(i, k)] for k in range(0, j)]
                    cat_list.append(up)
                    concat = torch.cat(cat_list, dim=1)
                    feat[self._key(i, j)] = self.nested_convs[self._key(i, j)](concat)
            return feat

    def freeze_encoder(self) -> None:
        for enc in self.encoders:
            for p in enc.parameters():
                p.requires_grad = False

    def compute_flops(self, input_shape: tuple[int, ...] = None) -> int:
        """
        简化 FLOPs 估算（用于对比，不用于论文精确统计）
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)

        B, Cin, H, W = input_shape
        flops = 0
        h, w = H, W

        # encoder convs
        for i, ch in enumerate(self.features):
            in_ch = Cin if i == 0 else self.features[i - 1]
            if i > 0:
                h //= 2
                w //= 2
            # two 3x3 convs
            flops += 2 * in_ch * ch * 9 * h * w
            flops += 2 * ch * ch * 9 * h * w

        # nested paths (roughly multiple of encoder)
        flops *= 2

        self._flops = flops * B
        return self._flops
