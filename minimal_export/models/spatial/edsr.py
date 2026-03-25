"""
EDSR 模型实现（强基线 / 可用于 SR 或同分辨率重建）

EDSR（Enhanced Deep Super-Resolution Network）是经典的“无 BN 残差堆叠”SR/复原基线。
在你的 PDEBench 稀疏观测重建任务中，通常使用：
- upscale=1：同分辨率输入->输出（mask/稀疏观测已经对齐到目标网格）
如果你做 SRx2/x4（低分辨率->高分辨率），可使用：
- upscale=2/4：内部 PixelShuffle 上采样

Reference:
    Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)",
    CVPR Workshops 2017.
    https://arxiv.org/abs/1707.02921
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Building blocks
# -------------------------
class ResBlock(nn.Module):
    """EDSR Residual Block: Conv-ReLU-Conv with residual scaling (no BN)."""

    def __init__(self, n_feats: int, res_scale: float = 0.1, bias: bool = True):
        super().__init__()
        self.res_scale = float(res_scale)
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act(self.conv1(x)))
        return x + res * self.res_scale


class Upsampler(nn.Sequential):
    """
    PixelShuffle upsampler used in classic SR models.
    Supports scale in {2, 3, 4, 8} (4/8 realized by repeated x2).
    """

    def __init__(self, scale: int, n_feats: int, bias: bool = True):
        m = []
        is_power_of_two = (scale > 0) and ((scale & (scale - 1)) == 0)

        if scale == 1:
            # no upsample
            pass
        elif is_power_of_two:
            n = int(torch.log2(torch.tensor(scale)).item())
            for _ in range(n):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, 1, bias=bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upscale={scale}. Use power of 2 or 3.")
        super().__init__(*m)


# -------------------------
# EDSR main model
# -------------------------
@register_model(name="EDSR", aliases=["edsr", "edsrnet"])
class EDSR(BaseModel):
    """
    EDSR baseline.

    Unified interface:
        forward(x[B,C_in,H,W]) -> y[B,C_out,H*(s),W*(s)]

    Typical PDEBench (same resolution):
        upscale=1, add_input_residual=True (when in_channels == out_channels)

    Typical SR:
        upscale=4, residual uses upsampled input by bicubic (optional).
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        n_feats: int = 64,
        n_resblocks: int = 16,
        res_scale: float = 0.1,
        upscale: int = 1,
        bias: bool = True,
        add_input_residual: bool | None = None,
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
        self.n_resblocks = int(n_resblocks)
        self.res_scale = float(res_scale)
        self.upscale = int(upscale)
        self.bias = bool(bias)
        self.grad_checkpointing = False

        if add_input_residual is None:
            # restoration 常用：同通道时直接 residual learning
            self.add_input_residual = in_channels == out_channels
        else:
            self.add_input_residual = bool(add_input_residual)

        self.residual_interp_mode = str(residual_interp_mode)

        # Head
        self.head = nn.Conv2d(in_channels, self.n_feats, 3, 1, 1, bias=self.bias)

        # Body
        body = [
            ResBlock(self.n_feats, res_scale=self.res_scale, bias=self.bias)
            for _ in range(self.n_resblocks)
        ]
        body.append(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=self.bias))
        self.body = nn.Sequential(*body)

        # Upsample (optional)
        self.upsampler = (
            Upsampler(self.upscale, self.n_feats, bias=self.bias)
            if self.upscale != 1
            else nn.Identity()
        )

        # Tail
        self.tail = nn.Conv2d(self.n_feats, out_channels, 3, 1, 1, bias=self.bias)

        self._init_weights()

    def set_gradient_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def _init_weights(self):
        # 与你现有风格一致：轻量初始化（避免过大初值）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x

        # feature extraction
        x = self.head(x)

        # Body with gradient checkpointing
        if self.grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            # Split body into chunks
            # 32 blocks -> 4 chunks of 8 blocks
            num_chunks = 4
            chunks = []
            chunk_size = len(self.body) // num_chunks
            if chunk_size < 1:
                chunk_size = 1

            # Convert Sequential to list for slicing
            body_layers = list(self.body)

            current_x = x
            for i in range(0, len(body_layers), chunk_size):
                segment = nn.Sequential(*body_layers[i : i + chunk_size])

                def run_segment(input_feats, s=segment):
                    return s(input_feats)

                current_x = checkpoint(run_segment, current_x, use_reentrant=False)

            res = current_x
        else:
            res = self.body(x)

        x = x + res  # global residual in feature space

        # upsample if needed
        x = self.upsampler(x)

        # reconstruction
        out = self.tail(x)

        # input residual (optional)
        if self.add_input_residual and (inp.shape[1] == out.shape[1]):
            if self.upscale == 1:
                out = out + inp
            else:
                # SR 情况：把输入先插值到输出分辨率再加 residual（常见做法）
                inp_up = F.interpolate(
                    inp,
                    scale_factor=self.upscale,
                    mode=self.residual_interp_mode,
                    align_corners=(
                        False
                        if self.residual_interp_mode in ("bilinear", "bicubic")
                        else None
                    ),
                )
                out = out + inp_up

        return out

    def get_model_info(self) -> dict:
        return {
            "name": "EDSR",
            "type": "CNN_SR_Restoration",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "n_feats": self.n_feats,
            "n_resblocks": self.n_resblocks,
            "res_scale": self.res_scale,
            "upscale": self.upscale,
            "add_input_residual": self.add_input_residual,
        }


# 别名
EDSRNet = EDSR
