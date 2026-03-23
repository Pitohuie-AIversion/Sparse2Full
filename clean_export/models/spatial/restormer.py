"""
Restormer 模型实现（正确结构版本）

Restormer 是面向高分辨率图像复原（Image Restoration）的高效 Transformer，
核心设计包含：
- Overlapped Patch Embedding（3x3 Conv, stride=1）
- MDTA（Multi-DConv Head Transposed Attention）：在“通道维度”做注意力以降低复杂度
- GDFN（Gated-DConv Feed-Forward Network）：门控 + 深度可分离卷积增强局部建模
- U 形多尺度编码器-解码器：PixelUnshuffle 下采样 / PixelShuffle 上采样 + skip 连接
- Refinement stage：在最高分辨率再堆叠若干块细化输出

Reference:
    Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022.
    https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf
    Official repo (architecture reference): https://github.com/swz30/Restormer
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Utilities
# -------------------------
def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    """Pad H,W to a multiple of `multiple` (bottom/right padding)."""
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # reflect padding is common for restoration; fallback to constant if too small
    mode = "reflect" if (h > 1 and w > 1) else "constant"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, pad_h, pad_w


def _unpad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    h = x.shape[-2] - pad_h
    w = x.shape[-1] - pad_w
    return x[..., :h, :w]


# -------------------------
# LayerNorm variants used in Restormer
# (BiasFree / WithBias) applied on channel-last tokens
# -------------------------
class BiasFreeLayerNorm(nn.Module):
    """Bias-Free LayerNorm (Restormer paper)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + self.eps)
        return x * self.weight


class WithBiasLayerNorm(nn.Module):
    """With-Bias LayerNorm (Restormer paper)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """Apply (BiasFree / WithBias) LayerNorm on 4D feature maps by tokenizing H*W."""

    def __init__(self, dim: int, ln_type: str = "WithBias"):
        super().__init__()
        if ln_type.lower() == "biasfree":
            self.norm = BiasFreeLayerNorm(dim)
        elif ln_type.lower() == "withbias":
            self.norm = WithBiasLayerNorm(dim)
        else:
            raise ValueError(f"Unsupported layer_norm_type: {ln_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H*W, C] -> LN -> back
        b, c, h, w = x.shape
        x_ = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_ = self.norm(x_)
        x_ = x_.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x_


# -------------------------
# Core Restormer blocks
# -------------------------
class FeedForwardGDFN(nn.Module):
    """
    GDFN: Gated-DConv Feed-Forward Network (Restormer paper).
    1x1 Conv -> depthwise 3x3 -> gate (GELU) -> 1x1 Conv
    """

    def __init__(self, dim: int, expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class AttentionMDTA(nn.Module):
    """
    MDTA: Multi-DConv Head Transposed Attention (Restormer paper).
    注意力在“通道维度”做，复杂度更低：attn ~ [B, head, C_head, C_head]
    """

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        super().__init__()

        # 保障 dim 可被 heads 整除（与你前面 UNetFormer 的处理一致）
        if hasattr(num_heads, "__iter__"):
            num_heads = int(list(num_heads)[0]) if len(list(num_heads)) > 0 else 8
        num_heads = int(num_heads)
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        ch = c // self.num_heads
        # [B, head, ch, HW]
        q = q.reshape(b, self.num_heads, ch, h * w)
        k = k.reshape(b, self.num_heads, ch, h * w)
        v = v.reshape(b, self.num_heads, ch, h * w)

        # 论文/官方实现常用：L2 normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 注意力在通道维度：[B, head, ch, ch]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B, head, ch, HW]
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Restormer Transformer Block: LN -> MDTA -> residual -> LN -> GDFN -> residual"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, ln_type=layer_norm_type)
        self.attn = AttentionMDTA(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim, ln_type=layer_norm_type)
        self.ffn = FeedForwardGDFN(dim, expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlap Patch Embedding (Restormer paper): 3x3 Conv, stride=1"""

    def __init__(self, in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """
    Downsample (Restormer official structure):
    Conv2d(n, n//2) + PixelUnshuffle(2)  -> channels: (n//2)*4 = 2n, spatial: /2
    """

    def __init__(self, n_feat: int, bias: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """
    Upsample (Restormer official structure):
    Conv2d(n, 2n) + PixelShuffle(2) -> channels: (2n)/4 = n/2, spatial: *2
    """

    def __init__(self, n_feat: int, bias: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# -------------------------
# Restormer main model
# -------------------------
@register_model(name="Restormer", aliases=["restormer"])
class Restormer(BaseModel):
    """
    Restormer (CVPR 2022) — U-shaped multi-scale Transformer for restoration/reconstruction.

    统一接口：
        forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]

    参数建议（PDEBench 128x128 常用）：
        dim=48,
        num_blocks=[4,6,6,8],
        heads=[1,2,4,8],
        num_refinement_blocks=4
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        dim: int = 48,
        num_blocks: List[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: List[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        add_input_residual: Optional[bool] = None,
        **kwargs,
    ):
        # 兼容常见别名
        in_channels = kwargs.get("in_ch", kwargs.get("in_chans", in_channels))
        out_channels = kwargs.get("out_ch", kwargs.get("num_classes", out_channels))
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if isinstance(num_blocks, tuple):
            num_blocks = list(num_blocks)
        if isinstance(heads, tuple):
            heads = list(heads)

        assert len(num_blocks) == 4, "Restormer expects 4 levels: num_blocks length must be 4"
        assert len(heads) == 4, "Restormer expects 4 levels: heads length must be 4"

        self.dim = dim
        self.num_blocks = num_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.layer_norm_type = layer_norm_type

        # 默认：当 in/out 通道相同，使用残差（out += input），符合 restoration 习惯
        if add_input_residual is None:
            self.add_input_residual = (in_channels == out_channels)
        else:
            self.add_input_residual = bool(add_input_residual)

        # 1) Overlap patch embed
        self.patch_embed = OverlapPatchEmbed(in_channels, dim, bias=bias)

        # 2) Encoder
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim, bias=bias)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2, bias=bias)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4, bias=bias)

        # 3) Latent
        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[3])]
        )

        # 4) Decoder
        self.up4_3 = Upsample(dim * 8, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2, bias=bias)
        # 注意：decoder_level1 的通道是 dim*2（上采样 dim 与 encoder1 dim 拼接）
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_blocks[0])]
        )

        # 5) Refinement + Output
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layer_norm_type) for _ in range(num_refinement_blocks)]
        )
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self._init_weights()

    def _init_weights(self):
        # 轻量初始化策略：conv 使用 xavier，LN 参数默认 1/0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        # 基础数值保护（不做强 clamp，避免限制回归幅度）
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x

        # Restormer 有 3 次 /2 下采样，要求 H,W 可被 8 整除；这里自动 pad
        x, pad_h, pad_w = _pad_to_multiple(x, multiple=8)

        x = self.patch_embed(x)

        # Encoder
        enc1 = self.encoder_level1(x)          # [B, dim,   H,   W]
        x = self.down1_2(enc1)                 # [B, dim*2, H/2, W/2]

        enc2 = self.encoder_level2(x)          # [B, dim*2, H/2, W/2]
        x = self.down2_3(enc2)                 # [B, dim*4, H/4, W/4]

        enc3 = self.encoder_level3(x)          # [B, dim*4, H/4, W/4]
        x = self.down3_4(enc3)                 # [B, dim*8, H/8, W/8]

        # Latent
        x = self.latent(x)                     # [B, dim*8, H/8, W/8]

        # Decoder level3
        x = self.up4_3(x)                      # [B, dim*4, H/4, W/4]
        x = torch.cat([x, enc3], dim=1)        # [B, dim*8, H/4, W/4]
        x = self.reduce_chan_level3(x)         # [B, dim*4, H/4, W/4]
        x = self.decoder_level3(x)             # [B, dim*4, H/4, W/4]

        # Decoder level2
        x = self.up3_2(x)                      # [B, dim*2, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)        # [B, dim*4, H/2, W/2]
        x = self.reduce_chan_level2(x)         # [B, dim*2, H/2, W/2]
        x = self.decoder_level2(x)             # [B, dim*2, H/2, W/2]

        # Decoder level1
        x = self.up2_1(x)                      # [B, dim, H, W]
        x = torch.cat([x, enc1], dim=1)        # [B, dim*2, H, W]
        x = self.decoder_level1(x)             # [B, dim*2, H, W]

        # Refinement + Output
        x = self.refinement(x)                 # [B, dim*2, H, W]
        out = self.output(x)                   # [B, C_out, H, W]

        out = _unpad(out, pad_h, pad_w)

        # 可选：加 input residual（恢复任务常用；PDE 任务可按需关）
        if self.add_input_residual and inp.shape[1] == out.shape[1]:
            out = out + inp

        return out

    def get_model_info(self) -> dict:
        return {
            "name": "Restormer",
            "type": "HybridTransformer",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "dim": self.dim,
            "num_blocks": self.num_blocks,
            "heads": self.heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "layer_norm_type": self.layer_norm_type,
            "add_input_residual": self.add_input_residual,
        }


# 别名，保持一致性
RestormerNet = Restormer
