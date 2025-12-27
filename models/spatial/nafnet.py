"""
NAFNet 模型实现（Nonlinear Activation Free Network）

面向图像复原/重建任务的强 CNN 基线；在多个复原任务中表现稳定。
适配 PDEBench 稀疏观测重建任务：严格遵循统一接口
    forward(x[B, C_in, H, W]) -> y[B, C_out, H, W]

Reference / 出处：
- NAFNet: Nonlinear Activation Free Network for Image Restoration (CVPR 2022)
  Chen Liangyu et al. arXiv:2204.04676
- 官方实现（同名项目）：https://github.com/megvii-research/NAFNet
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# utils
# -------------------------
def _as_int_list(x: Union[int, List[int], Tuple[int, ...]], name: str) -> List[int]:
    """兼容 Hydra/OmegaConf：允许 int / list[int] / tuple[int]"""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError(f"{name} should not be empty.")
        return [int(v) for v in x]
    return [int(x)]


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    将 H,W pad 到 multiple 的整数倍（右/下补零），返回 pad 后张量和 pad 信息。
    pad_info: (pad_left, pad_right, pad_top, pad_bottom)
    """
    if multiple <= 1:
        return x, (0, 0, 0, 0)

    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_left = 0
    pad_top = 0
    pad_right = pad_w
    pad_bottom = pad_h

    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return x, (pad_left, pad_right, pad_top, pad_bottom)


def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return x
    _, _, h, w = x.shape
    return x[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]


# -------------------------
# layers (NAFNet)
# -------------------------
class LayerNorm2d(nn.Module):
    """对 [B,C,H,W] 做 LayerNorm（沿通道维 C 归一化），NAFNet 常用实现。"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    """NAFNet 的核心门控：将通道一分为二后逐元素相乘。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """
    Simplified Channel Attention（NAFNet）
    使用全局平均池化 + 1x1 卷积得到通道权重。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv(self.avg_pool(x))
        return x * w


class NAFBlock(nn.Module):
    """
    NAFBlock：不显式使用 ReLU/GELU 等非线性激活，依靠 SimpleGate + 残差缩放实现表达。
    出处：NAFNet (CVPR 2022), arXiv:2204.04676
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        # ---- block 1 (spatial mixing) ----
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, kernel_size=3, stride=1, padding=1, groups=dw_channels, bias=True
        )
        self.sg = SimpleGate()
        self.sca = SCA(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ---- block 2 (FFN) ----
        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 残差缩放参数（论文/官方实现常用）
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- first sub-block -----
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)           # 通道减半
        y = self.sca(y)
        y = self.conv3(y)
        y = self.drop1(y)
        x = x + y * self.beta

        # ----- second sub-block -----
        z = self.norm2(x)
        z = self.conv4(z)
        z = self.sg(z)           # 通道减半
        z = self.conv5(z)
        z = self.drop2(z)
        x = x + z * self.gamma
        return x


class Downsample(nn.Module):
    """NAFNet 下采样：2x2 stride=2 卷积，通道翻倍。"""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * 2, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """NAFNet 上采样：1x1 卷积扩通道 + PixelShuffle(2)，通道减半。"""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


# -------------------------
# model
# -------------------------
@register_model(name="nafnet", aliases=["NAFNet", "naf"])
class NAFNet(BaseModel):
    """
    NAFNet（CVPR 2022）用于 PDEBench 稀疏观测重建的实现版本。

    训练口径建议（与你现有口径一致）：
      x = concat([obs, mask], dim=1) -> in_channels = C_obs + 1

    参数说明（保持可控的 baseline）：
    - width: 主干基准通道数
    - enc_blk_nums / dec_blk_nums: 每个尺度的 NAFBlock 数量（可为 list 或单 int）
    - middle_blk_num: bottleneck 的 block 数量
    - residual: 是否做输出残差 y = out + x（仅当 in_channels == out_channels 才建议开启）
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        width: int = 32,
        enc_blk_nums: Union[int, List[int], Tuple[int, ...]] = (2, 2, 4, 8),
        dec_blk_nums: Union[int, List[int], Tuple[int, ...]] = (2, 2, 4, 8),
        middle_blk_num: int = 12,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
        **kwargs,
    ):
        # 兼容常见别名
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", kwargs.pop("in_chans", 1))
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", kwargs.pop("num_classes", 1))
        if img_size is None:
            img_size = kwargs.get("img_size", 128)

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.width = int(width)
        self.residual = bool(residual)
        self.dw_expand = int(dw_expand)
        self.ffn_expand = int(ffn_expand)
        self.dropout = float(dropout)

        enc_nums = _as_int_list(enc_blk_nums, "enc_blk_nums")
        dec_nums = _as_int_list(dec_blk_nums, "dec_blk_nums")
        if len(enc_nums) != len(dec_nums):
            raise ValueError(f"enc_blk_nums and dec_blk_nums must have same length. Got {len(enc_nums)} vs {len(dec_nums)}")

        self.num_stages = len(enc_nums)
        # pad 需要满足 2**num_stages 的倍数（下采样 num_stages 次）
        self.pad_multiple = 2 ** self.num_stages

        # 输入/输出
        self.intro = nn.Conv2d(self.in_channels, self.width, kernel_size=3, stride=1, padding=1, bias=True)
        self.ending = nn.Conv2d(self.width, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # 编码器
        channels = self.width
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for n_blk in enc_nums:
            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(
                        channels,
                        dw_expand=self.dw_expand,
                        ffn_expand=self.ffn_expand,
                        dropout=self.dropout
                    )
                    for _ in range(n_blk)
                ])
            )
            self.downs.append(Downsample(channels))
            channels *= 2

        # bottleneck
        self.middle = nn.Sequential(*[
            NAFBlock(
                channels,
                dw_expand=self.dw_expand,
                ffn_expand=self.ffn_expand,
                dropout=self.dropout
            )
            for _ in range(int(middle_blk_num))
        ])

        # 解码器
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blk in dec_nums[::-1]:
            self.ups.append(Upsample(channels))
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(
                        channels,
                        dw_expand=self.dw_expand,
                        ffn_expand=self.ffn_expand,
                        dropout=self.dropout
                    )
                    for _ in range(n_blk)
                ])
            )

        self._init_weights()

    def _init_weights(self):
        # NAFNet 官方常用截断正态/kaiming 均可；这里用较稳健的 kaiming/xavier 组合
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
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
        inp = x
        x, pad = _pad_to_multiple(x, self.pad_multiple)

        x = self.intro(x)

        # encoder
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # middle
        x = self.middle(x)

        # decoder（与 NAFNet 官方一致：skip 采用逐元素相加，不拼接）
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # 尺寸对齐保护（理论上 pad 后应严格对齐；这里留保险）
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = dec(x)

        x = self.ending(x)
        x = _unpad(x, pad)

        # 可选 residual（仅当通道一致时更合理）
        if self.residual and (self.in_channels == self.out_channels):
            x = x + inp

        return x

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "arch": "NAFNet",
            "width": self.width,
            "num_stages": self.num_stages,
            "pad_multiple": self.pad_multiple,
            "dw_expand": self.dw_expand,
            "ffn_expand": self.ffn_expand,
            "dropout": self.dropout,
            "residual": self.residual,
        })
        return info


def create_nafnet(**kwargs) -> NAFNet:
    """工厂函数（与项目内其他模型一致）"""
    return NAFNet(**kwargs)
