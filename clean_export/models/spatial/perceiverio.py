"""
PerceiverIO（2D 稀疏观测 → 全场重建 适配版）

Perceiver/PerceiverIO 的关键优势是：
- 输入 token 数很大（H*W），但用固定数量的 latent（M）进行 cross-attn 聚合，复杂度 ~ O(MN)
- 对稀疏观测任务非常合适：输入可以是“值+mask”的图，模型通过 cross-attn 汇聚有效信息
- 输出同样可用坐标 query 生成稠密场（PerceiverIO 的核心范式）

本实现工程化适配你的统一接口：
- 输入：x[B, C_in, H, W]
- 输出：y[B, C_out, H, W]
- 可选：若 kwargs 提供 mask[B,1,H,W]，会将 mask 作为额外特征拼进 token（use_mask_as_feature=True）

Reference:
    Perceiver:
        Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021.
        https://arxiv.org/abs/2103.03206
    Perceiver IO:
        Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs", ICML 2021.
        https://arxiv.org/abs/2107.14795
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from omegaconf import ListConfig

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Utils: coord cache + Fourier features
# -------------------------
class FourierFeatures(nn.Module):
    """
    Fourier features for 2D coords in [-1,1].
    (Common trick for coordinate-conditioned decoders; not unique to Perceiver.)
    """
    def __init__(self, num_frequencies: int = 8, scale: float = 1.0):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)
        freqs = torch.tensor([2**i for i in range(self.num_frequencies)], dtype=torch.float32) * self.scale
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N,2]
        x = coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) * (2.0 * math.pi)  # [N,2,F]
        sin = torch.sin(x)
        cos = torch.cos(x)
        # [N, 2 + 4F]
        return torch.cat([coords, sin.flatten(1), cos.flatten(1)], dim=1)


@torch.no_grad()
def build_coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # [N,2]


# -------------------------
# Attention blocks (PreNorm)
# -------------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, act: str = "gelu"):
        super().__init__()
        if act == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act == "gelu":
            act_layer = nn.GELU()
        elif act == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act: {act}")
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Standard MHA supporting cross-attention:
    - queries: x
    - keys/values: context (optional; if None, self-attn)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, qkv_bias: bool = True):
        super().__init__()
        # Handle ListConfig or other types for num_heads
        if hasattr(num_heads, '__len__') and not isinstance(num_heads, str):
            num_heads = num_heads[0]
        num_heads = int(num_heads)
        
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, M, D], context: [B, N, D]
        if context is None:
            context = x
        b, m, d = x.shape
        _, n, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # [B, heads, M/N, head_dim]
        q = q.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, M, N]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ v  # [B, heads, M, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, m, d)
        out = self.proj(out)
        out = self.drop(out)
        return out


class PerceiverEncoderLayer(nn.Module):
    """
    One encoder layer:
    - Cross-attn: latents <- inputs
    - Latent self-attn blocks
    Ref: Perceiver / Perceiver IO (Jaegle et al.)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        self_attn_depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.cross_attn = PreNorm(dim, MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout))
        self.cross_mlp = PreNorm(dim, MLP(dim, int(dim * mlp_ratio), dropout=dropout, act=act))

        self.self_blocks = nn.ModuleList([])
        for _ in range(int(self_attn_depth)):
            self.self_blocks.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)),
                PreNorm(dim, MLP(dim, int(dim * mlp_ratio), dropout=dropout, act=act))
            ]))

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # latents: [B,M,D], inputs: [B,N,D]
        latents = latents + self.cross_attn(latents, context=inputs)
        latents = latents + self.cross_mlp(latents)
        for attn, mlp in self.self_blocks:
            latents = latents + attn(latents)
            latents = latents + mlp(latents)
        return latents


class PerceiverDecoder(nn.Module):
    """
    Decoder:
    - output queries (e.g., coords) attend to latents via cross-attn
    - optional MLP
    - projection to out_channels
    Ref: Perceiver IO (structured outputs via queries)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        out_channels: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.cross_attn = PreNorm(dim, MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout))
        self.mlp = PreNorm(dim, MLP(dim, int(dim * mlp_ratio), dropout=dropout, act=act))
        self.to_out = nn.Linear(dim, out_channels)

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # queries: [B,N,D], latents: [B,M,D]
        x = queries + self.cross_attn(queries, context=latents)
        x = x + self.mlp(x)
        return self.to_out(x)  # [B,N,out_ch]


# -------------------------
# PerceiverIO 2D model
# -------------------------
@register_model(name="PerceiverIO", aliases=["perceiverio", "perceiver_io"])
class PerceiverIO2D(BaseModel):
    """
    PerceiverIO (2D dense output) for PDEBench-style sparse reconstruction.

    forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        dim: int = 256,                 # token/latent dim
        num_latents: int = 256,         # M
        num_heads: int = 8,
        depth: int = 2,                 # number of encoder layers (each has cross + latent self)
        self_attn_depth: int = 2,       # latent self-attn blocks per layer
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "gelu",
        use_fourier_features: bool = True,
        fourier_frequencies: int = 8,
        fourier_scale: float = 1.0,
        use_mask_as_feature: bool = True,
        final_activation: Optional[str] = None,
        add_input_residual: Optional[bool] = None,
        **kwargs,
    ):
        # 兼容别名
        in_channels = kwargs.get("in_ch", kwargs.get("in_chans", in_channels))
        out_channels = kwargs.get("out_ch", kwargs.get("num_classes", out_channels))
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.dim = int(dim)
        self.num_latents = int(num_latents)
        self.use_mask_as_feature = bool(use_mask_as_feature)

        # coordinate features
        self.use_fourier_features = bool(use_fourier_features)
        if self.use_fourier_features:
            self.ff = FourierFeatures(num_frequencies=fourier_frequencies, scale=fourier_scale)
            coord_dim = 2 + 4 * int(fourier_frequencies)
        else:
            self.ff = None
            coord_dim = 2

        # input token projection (x token + coord [+mask])
        extra = 1 if self.use_mask_as_feature else 0
        token_in_dim = in_channels + coord_dim + extra
        self.input_proj = nn.Linear(token_in_dim, self.dim)

        # learnable latents
        self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.dim) * 0.02)

        # encoder stack
        self.encoder = nn.ModuleList([
            PerceiverEncoderLayer(
                dim=self.dim,
                num_heads=num_heads,
                self_attn_depth=self_attn_depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                act=act,
            )
            for _ in range(int(depth))
        ])

        # output query projection: coord -> dim
        self.query_proj = nn.Linear(coord_dim, self.dim)

        # decoder
        self.decoder = PerceiverDecoder(
            dim=self.dim,
            num_heads=num_heads,
            out_channels=out_channels,
            mlp_ratio=2.0,
            dropout=dropout,
            act=act,
        )

        # optional output activation
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        # optional input residual
        self.add_input_residual = (in_channels == out_channels) if add_input_residual is None else bool(add_input_residual)

        # coord cache
        self._coord_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _get_coord_feat(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key in self._coord_cache:
            return self._coord_cache[key]
        coords = build_coords(h, w, device, dtype)  # [N,2]
        if self.ff is not None:
            coords = self.ff(coords)  # [N, coord_dim]
        self._coord_cache[key] = coords
        return coords

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B,C_in,H,W]
            kwargs:
                mask (optional): [B,1,H,W]  (若提供，将作为 token 的额外特征拼接；推荐用于稀疏观测)
        Returns:
            y: [B,C_out,H,W]
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x
        b, c, h, w = x.shape
        n = h * w

        # build coord features
        coord_feat = self._get_coord_feat(h, w, x.device, x.dtype)  # [N, coord_dim]
        coord_feat_b = coord_feat.unsqueeze(0).expand(b, -1, -1)    # [B,N,coord_dim]

        # tokenize input
        x_tok = x.flatten(2).transpose(1, 2)                        # [B,N,C_in]

        # optional mask feature
        if self.use_mask_as_feature:
            mask = kwargs.get("mask", None)
            if mask is not None:
                mask_tok = mask.flatten(2).transpose(1, 2)          # [B,N,1]
            else:
                # 若没提供 mask，就用“是否为零”作为弱提示（不强制；有些 PDE 变量允许为 0）
                mask_tok = torch.zeros(b, n, 1, device=x.device, dtype=x.dtype)
            x_in = torch.cat([x_tok, coord_feat_b, mask_tok], dim=-1)
        else:
            x_in = torch.cat([x_tok, coord_feat_b], dim=-1)

        inputs = self.input_proj(x_in)                              # [B,N,dim]

        # latents
        latents = self.latents.expand(b, -1, -1).contiguous()        # [B,M,dim]

        # encoder
        for layer in self.encoder:
            latents = layer(latents, inputs)

        # output queries: coords -> dim
        queries = self.query_proj(coord_feat).unsqueeze(0).expand(b, -1, -1)  # [B,N,dim]

        # decode to per-pixel tokens
        out_tok = self.decoder(queries, latents)                     # [B,N,out_ch]

        # reshape back
        y = out_tok.transpose(1, 2).contiguous().view(b, self.out_channels, h, w)
        y = self.final_activation(y)

        if self.add_input_residual and (inp.shape[1] == y.shape[1]):
            y = y + inp

        return y

    def get_model_info(self) -> dict:
        return {
            "name": "PerceiverIO2D",
            "type": "LatentCrossAttention",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "dim": self.dim,
            "num_latents": self.num_latents,
            "use_fourier_features": self.use_fourier_features,
            "use_mask_as_feature": self.use_mask_as_feature,
            "add_input_residual": self.add_input_residual,
        }


# alias
PerceiverIO = PerceiverIO2D
