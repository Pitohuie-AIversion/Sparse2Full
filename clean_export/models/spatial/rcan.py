"""
RCAN 模型实现（Residual Channel Attention Network）

RCAN 是经典图像复原/超分辨率（SR）强基线：
- Residual-in-Residual (RIR) 结构：多个 Residual Group + Long Skip
- Channel Attention (CA) ：通过全局平均池化 + 两层 1x1 Conv 的通道注意力
- 不使用 BN（提升 SR/复原性能且更稳定）

在 PDEBench 稀疏观测重建任务中，通常使用：
- upscale=1：同分辨率输入->输出（输入已对齐到目标网格）
也可用于 SR：
- upscale=2/4 等：PixelShuffle 上采样

Reference:
    Zhang et al., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks (RCAN)",
    ECCV 2018.
    https://arxiv.org/abs/1807.02758
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Utilities
# -------------------------
class Upsampler(nn.Sequential):
    """
    PixelShuffle upsampler:
    - supports scale in {1, 2, 3, 4, 8}
    """

    def __init__(self, scale: int, n_feats: int, bias: bool = True):
        m = []
        if scale == 1:
            pass
        elif scale in (2, 4, 8):
            n = int(torch.log2(torch.tensor(scale)).item())
            for _ in range(n):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, 1, bias=bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upscale={scale}. Use 1/2/3/4/8.")
        super().__init__(*m)


# -------------------------
# Channel Attention
# -------------------------
class CALayer(nn.Module):
    """
    Channel Attention (CA) layer from RCAN:
    GAP -> Conv(1x1) -> ReLU -> Conv(1x1) -> Sigmoid -> scale
    """

    def __init__(self, n_feats: int, reduction: int = 16, bias: bool = True):
        super().__init__()
        reduced = max(n_feats // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feats, reduced, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, n_feats, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# -------------------------
# RCAB / Residual Group / RIR
# -------------------------
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB):
    Conv -> ReLU -> Conv -> CA -> residual (with scaling)
    """

    def __init__(
        self,
        n_feats: int,
        reduction: int = 16,
        res_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.res_scale = float(res_scale)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=bias),
            CALayer(n_feats, reduction=reduction, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


class ResidualGroup(nn.Module):
    """
    Residual Group:
    (RCAB x n_blocks) + Conv -> group residual
    """

    def __init__(
        self,
        n_feats: int,
        n_blocks: int,
        reduction: int = 16,
        res_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        modules = [RCAB(n_feats, reduction=reduction, res_scale=res_scale, bias=bias) for _ in range(n_blocks)]
        modules.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=bias))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        return x + res


# -------------------------
# RCAN main model
# -------------------------
@register_model(name="RCAN", aliases=["rcan"])
class RCAN(BaseModel):
    """
    RCAN baseline.

    Unified interface:
        forward(x[B,C_in,H,W]) -> y[B,C_out,H*s,W*s]

    Recommended (PDEBench same-res reconstruction):
        n_feats=64, n_groups=5~10, n_blocks=10~20, upscale=1
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        n_feats: int = 64,
        n_groups: int = 10,
        n_blocks: int = 20,
        reduction: int = 16,
        res_scale: float = 1.0,
        upscale: int = 1,
        bias: bool = True,
        add_input_residual: Optional[bool] = None,
        residual_interp_mode: str = "bicubic",
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

        self.n_feats = int(n_feats)
        self.n_groups = int(n_groups)
        self.n_blocks = int(n_blocks)
        self.reduction = int(reduction)
        self.res_scale = float(res_scale)
        self.upscale = int(upscale)
        self.bias = bool(bias)

        if add_input_residual is None:
            self.add_input_residual = (in_channels == out_channels)
        else:
            self.add_input_residual = bool(add_input_residual)

        self.residual_interp_mode = str(residual_interp_mode)

        # Head
        self.head = nn.Conv2d(in_channels, self.n_feats, 3, 1, 1, bias=self.bias)

        # Body: Residual-in-Residual (RIR)
        groups = [
            ResidualGroup(
                n_feats=self.n_feats,
                n_blocks=self.n_blocks,
                reduction=self.reduction,
                res_scale=self.res_scale,
                bias=self.bias,
            )
            for _ in range(self.n_groups)
        ]
        groups.append(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=self.bias))
        self.body = nn.Sequential(*groups)

        # Upsample (optional)
        self.upsampler = Upsampler(self.upscale, self.n_feats, bias=self.bias) if self.upscale != 1 else nn.Identity()

        # Tail
        self.tail = nn.Conv2d(self.n_feats, out_channels, 3, 1, 1, bias=self.bias)

        self._init_weights()

    def _init_weights(self):
        # 轻量初始化（与你当前仓库风格一致）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x

        x = self.head(x)

        # long skip in feature space
        res = self.body(x)
        x = x + res

        # upsample if needed
        x = self.upsampler(x)

        out = self.tail(x)

        # input residual (optional)
        if self.add_input_residual and (inp.shape[1] == out.shape[1]):
            if self.upscale == 1:
                out = out + inp
            else:
                inp_up = F.interpolate(
                    inp,
                    scale_factor=self.upscale,
                    mode=self.residual_interp_mode,
                    align_corners=False if self.residual_interp_mode in ("bilinear", "bicubic") else None,
                )
                out = out + inp_up

        return out

    def get_model_info(self) -> dict:
        return {
            "name": "RCAN",
            "type": "CNN_Attention_SR_Restoration",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "n_feats": self.n_feats,
            "n_groups": self.n_groups,
            "n_blocks": self.n_blocks,
            "reduction": self.reduction,
            "res_scale": self.res_scale,
            "upscale": self.upscale,
            "add_input_residual": self.add_input_residual,
        }


# 别名
RCANNet = RCAN
