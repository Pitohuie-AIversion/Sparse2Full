"""
UNetFormer（稳定可训练版本，显存友好）

UNet-like CNN encoder-decoder + Transformer blocks (with Spatial-Reduction Attention).
Designed for 2D reconstruction tasks (e.g., PDEBench), I/O:
    forward(x[B, C_in, H, W]) -> y[B, C_out, H, W]

Reference (conceptual):
    UNetFormer: A UNet-like transformer for efficient semantic segmentation
    https://arxiv.org/abs/2109.08417

Note:
- To avoid O(N^2) attention at high resolutions (e.g., 128x128 => N=16384),
  this implementation uses Spatial-Reduction Attention (SR) for keys/values.
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
def _as_int(x, default: int) -> int:
    """Hydra 可能传 list/tuple，这里取第一个。"""
    if isinstance(x, (list, tuple)):
        return int(x[0]) if len(x) > 0 else int(default)
    return int(x)


def _align_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Align src spatial size to ref by symmetric pad (if smaller) or center-crop (if larger).
    src: [B,C,Hs,Ws], ref: [B,C,Hr,Wr]
    """
    hs, ws = src.shape[-2], src.shape[-1]
    hr, wr = ref.shape[-2], ref.shape[-1]

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

    hs, ws = src.shape[-2], src.shape[-1]
    if hs > hr:
        y0 = (hs - hr) // 2
        src = src[:, :, y0 : y0 + hr, :]
    if ws > wr:
        x0 = (ws - wr) // 2
        src = src[:, :, :, x0 : x0 + wr]
    return src


# =========================================================
# CNN blocks
# =========================================================
class ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2 + optional dropout"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# =========================================================
# Transformer components (SR Attention)
# =========================================================
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class SRAttention(nn.Module):
    """
    Spatial-Reduction Multi-Head Self-Attention.
    - Q from full tokens (H*W)
    - K,V from downsampled tokens via Conv2d stride=sr_ratio
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        num_heads = _as_int(num_heads, 8)
        if dim % num_heads != 0:
            # 向下调整 heads，保证可整除
            while num_heads > 1 and dim % num_heads != 0:
                num_heads -= 1

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = int(sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=False
            )
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.sr_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: [B, N, C], N=H*W
        """
        B, N, C = x.shape

        q = (
            self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # [B,h,N,hd]

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]
            x_ = self.sr(x_)  # [B,C,H',W']
            Hk, Wk = x_.shape[-2], x_.shape[-1]
            x_ = x_.reshape(B, C, Hk * Wk).transpose(1, 2)  # [B,Nk,C]
            x_ = self.sr_norm(x_)
            kv = self.kv(x_)
            Nk = x_.shape[1]
        else:
            kv = self.kv(x)
            Nk = N

        kv = kv.reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B,h,Nk,hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,N,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B,N,C]
        out = self.proj_drop(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerConvBlock(nn.Module):
    """
    CNN + Transformer (SR attention) hybrid block.
    Input/Output: [B,C,H,W]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(channels, channels, dropout=dropout)
        self.trans = TransformerBlock(
            dim=channels,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN branch
        conv_out = self.conv(x)

        # Transformer branch
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = self.trans(tokens, H, W)
        trans_out = tokens.transpose(1, 2).reshape(B, C, H, W)

        return conv_out + trans_out


# =========================================================
# UNetFormer
# =========================================================
@register_model(name="UNetFormer", aliases=["unetformer", "UNetformer"])
class UNetFormer(BaseModel):
    """
    UNetFormer for reconstruction

    Encoder stages:
        stage1: H,W
        stage2: H/2,W/2
        stage3: H/4,W/4
        stage4: H/8,W/8 (optional by num_stages)

    Bottleneck:
        H/16,W/16 (if num_stages=4)
        H/8,W/8   (if num_stages=3)
        H/4,W/4   (if num_stages=2)

    Decoder mirrors encoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        base_channels: int = 64,
        num_stages: int = 4,  # 2/3/4
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout: float = 0.0,  # CNN dropout
        # SR ratios per stage (shallow -> deep), tuned for memory safety
        sr_ratios: list[int] | None = None,
        # how many hybrid blocks each stage uses
        depths: list[int] | None = None,
        **kwargs,
    ):
        # Common aliases
        in_channels = kwargs.get("in_ch", kwargs.get("in_chans", in_channels))
        out_channels = kwargs.get("out_ch", kwargs.get("num_classes", out_channels))

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.num_heads = _as_int(num_heads, 8)

        if self.num_stages not in (2, 3, 4):
            raise ValueError("num_stages must be 2, 3, or 4")

        # default depths
        if depths is None:
            depths = (
                [1, 1]
                if self.num_stages == 2
                else ([1, 1, 2] if self.num_stages == 3 else [2, 2, 4, 2])
            )
        if len(depths) != self.num_stages:
            raise ValueError(
                f"depths length must equal num_stages ({self.num_stages})."
            )
        self.depths = [int(d) for d in depths]

        # default SR ratios (important for memory)
        # stage1 has the largest H*W -> use larger sr_ratio
        if sr_ratios is None:
            sr_ratios = (
                [8, 4]
                if self.num_stages == 2
                else ([8, 4, 2] if self.num_stages == 3 else [8, 4, 2, 1])
            )
        if len(sr_ratios) != self.num_stages:
            raise ValueError(
                f"sr_ratios length must equal num_stages ({self.num_stages})."
            )
        self.sr_ratios = [int(r) for r in sr_ratios]

        # channel plan per stage
        chs = [self.base_channels * (2**i) for i in range(self.num_stages)]
        self.chs = chs

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            ConvBlock(self.in_channels, chs[0], dropout=dropout),
            *[
                TransformerConvBlock(
                    chs[0],
                    num_heads=self.num_heads,
                    sr_ratio=self.sr_ratios[0],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    dropout=dropout,
                )
                for _ in range(self.depths[0])
            ],
        )

        if self.num_stages >= 2:
            self.enc2 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(chs[0], chs[1], dropout=dropout),
                *[
                    TransformerConvBlock(
                        chs[1],
                        num_heads=self.num_heads,
                        sr_ratio=self.sr_ratios[1],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        dropout=dropout,
                    )
                    for _ in range(self.depths[1])
                ],
            )

        if self.num_stages >= 3:
            self.enc3 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(chs[1], chs[2], dropout=dropout),
                *[
                    TransformerConvBlock(
                        chs[2],
                        num_heads=self.num_heads,
                        sr_ratio=self.sr_ratios[2],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        dropout=dropout,
                    )
                    for _ in range(self.depths[2])
                ],
            )

        if self.num_stages >= 4:
            self.enc4 = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock(chs[2], chs[3], dropout=dropout),
                *[
                    TransformerConvBlock(
                        chs[3],
                        num_heads=self.num_heads,
                        sr_ratio=self.sr_ratios[3],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        dropout=dropout,
                    )
                    for _ in range(self.depths[3])
                ],
            )

        # -------- Bottleneck --------
        # bottleneck channels = 2x deepest stage
        bott_in = chs[-1]
        bott_out = chs[-1] * 2

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(bott_in, bott_out, dropout=dropout),
            TransformerConvBlock(
                bott_out,
                num_heads=self.num_heads,
                sr_ratio=1,  # bottleneck 已经很小，允许全局（或 sr=1）
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                dropout=dropout,
            ),
        )

        # -------- Decoder --------
        # Mirror: bott_out -> chs[-1] -> ... -> chs[0]
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = bott_out
        for stage in reversed(range(self.num_stages)):
            out_ch = chs[stage]
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # concat with skip => out_ch + out_ch
            self.dec_blocks.append(
                nn.Sequential(
                    ConvBlock(out_ch * 2, out_ch, dropout=dropout),
                    # 可选：decoder 也加一个轻量 hybrid block（用更大 sr_ratio 进一步省显存）
                    TransformerConvBlock(
                        out_ch,
                        num_heads=self.num_heads,
                        sr_ratio=(
                            max(self.sr_ratios[stage], 2)
                            if stage == 0
                            else max(self.sr_ratios[stage], 1)
                        ),
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        dropout=dropout,
                    ),
                )
            )
            in_ch = out_ch

        self.final_conv = nn.Conv2d(chs[0], self.out_channels, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            # 小标准差更稳
            (
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(nn.init, "trunc_normal_")
                else nn.init.normal_(m.weight, std=0.02)
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B,C_in,H,W]
        y: [B,C_out,H,W]
        """
        # Encoder
        e1 = self.enc1(x)
        skips = [e1]

        if self.num_stages >= 2:
            e2 = self.enc2(e1)
            skips.append(e2)
        if self.num_stages >= 3:
            e3 = self.enc3(e2)
            skips.append(e3)
        if self.num_stages >= 4:
            e4 = self.enc4(e3)
            skips.append(e4)

        # Bottleneck input depends on stages
        deepest = skips[-1]
        b = self.bottleneck(deepest)

        # Decoder (match skips from deepest->shallow)
        d = b
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            d = up(d)
            d = _align_like(d, skip)
            d = torch.cat([d, skip], dim=1)
            d = dec(d)

        out = self.final_conv(d)
        return out

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "name": "UNetFormer",
                "type": "Hybrid(CNN+Transformer-SR)",
                "base_channels": self.base_channels,
                "num_stages": self.num_stages,
                "channels_per_stage": self.chs,
                "depths": self.depths,
                "sr_ratios": self.sr_ratios,
                "num_heads": self.num_heads,
            }
        )
        return info


def create_unetformer(**kwargs) -> UNetFormer:
    return UNetFormer(**kwargs)
