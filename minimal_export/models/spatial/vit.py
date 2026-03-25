"""
Vision Transformer (ViT) for PDEBench reconstruction (stable + memory-friendly)

Unified I/O:
    forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]

Core references (architecture + training tricks):
1) ViT backbone:
   - Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
     (arXiv:2010.11929)

2) Autoencoder-style ViT reconstruction (encoder -> decoder -> patch pixels -> unpatchify):
   - He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
     (arXiv:2111.06377)
   Note: This implementation follows the same *patchify/unpatchify + decoder prediction* pattern,
   but is adapted for dense field reconstruction rather than masked pretraining.

3) Stochastic Depth / DropPath:
   - Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.

4) Spatial-Reduction Attention (SR-Attention) for memory-friendly attention at large token counts:
   - Wang et al., "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions", ICCV 2021.
   Note: Here SR-Attention is used as an engineering option to reduce O(N^2) attention cost.

Implementation notes:
- Positional embedding interpolation is commonly used in ViT-style models when adapting to new resolutions.
- For reconstruction tasks, cls token is typically unnecessary; kept as an optional switch for compatibility.

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# =========================================================
# Helpers
# =========================================================
def _as_int(x, default: int) -> int:
    """Hydra compatibility: num_heads may arrive as list/tuple; use the first element."""
    if isinstance(x, (list, tuple)):
        return int(x[0]) if len(x) > 0 else int(default)
    return int(x)


def _make_divisible_heads(dim: int, heads: int) -> int:
    """Ensure dim % heads == 0 by decreasing heads if needed (robust config handling)."""
    heads = max(int(heads), 1)
    while heads > 1 and (dim % heads != 0):
        heads -= 1
    return heads


# =========================================================
# DropPath (Stochastic Depth)
# Reference:
#   Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
# =========================================================
class DropPath(nn.Module):
    """Stochastic Depth / DropPath (Huang et al., ECCV 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =========================================================
# Patch Embedding
# Reference:
#   Dosovitskiy et al., ViT, ICLR 2021.
# =========================================================
class PatchEmbedding(nn.Module):
    """Image -> Patch tokens via Conv2d (ViT patch embedding; Dosovitskiy et al., ICLR 2021)."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        # Patch projection as a strided conv (equivalent to linear projection of flattened patches)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Args:
            x: [B,C,H,W]
        Returns:
            tokens: [B, N, D]
            grid: (Hp, Wp) where N = Hp*Wp
        """
        B, C, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(
                f"H,W must be divisible by patch_size={self.patch_size}, got {(H, W)}"
            )

        x = self.proj(x)  # [B,D,Hp,Wp]
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, D]
        x = self.norm(x)
        return x, (Hp, Wp)


# =========================================================
# SR-Attention (optional, memory-friendly)
# Reference:
#   Wang et al., "Pyramid Vision Transformer (PVT)", ICCV 2021.
# =========================================================
class SRAttention(nn.Module):
    """
    Spatial-Reduction Multi-Head Self-Attention (PVT-style SR attention).

    If sr_ratio > 1:
      - Q from full tokens (N)
      - K,V from downsampled tokens (Nk ~ N / sr_ratio^2)
    This reduces attention complexity from O(N^2) to ~O(N * Nk).

    Reference:
      Wang et al., PVT, ICCV 2021.
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
        num_heads = _make_divisible_heads(dim, _as_int(num_heads, 8))
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = int(sr_ratio)

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
                bias=False,
            )
            self.sr_norm = nn.LayerNorm(self.dim)
        else:
            self.sr = None
            self.sr_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, Hp: int, Wp: int, has_cls: bool = False
    ) -> torch.Tensor:
        """
        x: [B, N(+1), C]
        If has_cls=True: first token is cls, remaining are patch tokens.
        """
        if has_cls:
            cls_tok = x[:, :1, :]
            tok = x[:, 1:, :]
        else:
            cls_tok = None
            tok = x

        B, N, C = tok.shape
        q = (
            self.q(tok).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # [B,h,N,hd]

        if self.sr_ratio > 1:
            t2d = tok.transpose(1, 2).reshape(B, C, Hp, Wp)
            t2d = self.sr(t2d)  # [B,C,Hp',Wp']
            Hp_k, Wp_k = t2d.shape[-2], t2d.shape[-1]
            tok_kv = t2d.reshape(B, C, Hp_k * Wp_k).transpose(1, 2)  # [B,Nk,C]
            tok_kv = self.sr_norm(tok_kv)
        else:
            tok_kv = tok

        Nk = tok_kv.shape[1]
        kv = (
            self.kv(tok_kv)
            .reshape(B, Nk, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # [B,h,Nk,hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,N,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        if has_cls:
            # For reconstruction, cls is not used for decoding; keep it stable.
            return torch.cat([cls_tok, out], dim=1)
        return out


class MLP(nn.Module):
    """Transformer FFN (ViT; Dosovitskiy et al., ICLR 2021)."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block (ViT-style), optionally with SR-Attention (PVT-style).

    References:
      - Dosovitskiy et al., ViT, ICLR 2021.
      - Wang et al., PVT (SR attention), ICCV 2021.
      - Huang et al., Stochastic Depth, ECCV 2016 (DropPath).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SRAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(
        self, x: torch.Tensor, Hp: int, Wp: int, has_cls: bool = False
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), Hp, Wp, has_cls=has_cls))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@register_model(name="vit", aliases=["ViT", "VisionTransformer"])
class VisionTransformer(BaseModel):
    """
    ViT autoencoder-style reconstruction for PDEBench.

    High-level pattern (AE-style):
      patch_embed -> encoder -> decoder -> per-patch pixel prediction -> unpatchify

    References:
      - Dosovitskiy et al., ViT, ICLR 2021.
      - He et al., MAE (encoder-decoder + patchify/unpatchify design pattern), CVPR 2022.
      - Wang et al., PVT (optional SR attention), ICCV 2021.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        # SR ratios
        sr_ratio_enc: int = 1,
        sr_ratio_dec: int = 1,
        # decoder
        decoder_embed_dim: int | None = None,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # cls token
        use_cls_token: bool = False,
        # final activation
        final_activation: str | None = None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = _make_divisible_heads(self.embed_dim, _as_int(num_heads, 12))

        self.use_cls_token = bool(use_cls_token)

        self.decoder_embed_dim = (
            int(decoder_embed_dim) if decoder_embed_dim is not None else self.embed_dim
        )
        self.decoder_depth = int(decoder_depth)
        self.decoder_num_heads = _make_divisible_heads(
            self.decoder_embed_dim, _as_int(decoder_num_heads, 16)
        )

        self.sr_ratio_enc = int(sr_ratio_enc)
        self.sr_ratio_dec = int(sr_ratio_dec)

        # Patch embedding (ViT)
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer,
        )

        # Default num patches at model init resolution (pos_embed will be interpolated if input differs)
        num_patches_default = (self.img_size // self.patch_size) * (
            self.img_size // self.patch_size
        )

        # Positional embedding (learnable). Interpolation is a standard practice for resolution transfer in ViT.
        self.pos_drop = nn.Dropout(p=float(drop_rate))

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches_default + 1, self.embed_dim)
            )
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches_default, self.embed_dim)
            )

        # Encoder blocks (ViT-style, optional SR attention)
        dpr = torch.linspace(0, float(drop_path_rate), self.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    sr_ratio=self.sr_ratio_enc,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        # Encoder -> Decoder projection (MAE-style encoder/decoder split pattern)
        if self.decoder_embed_dim != self.embed_dim:
            self.decoder_embed = nn.Linear(
                self.embed_dim, self.decoder_embed_dim, bias=True
            )
        else:
            self.decoder_embed = nn.Identity()

        # Decoder positional embedding (MAE-style)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_default, self.decoder_embed_dim)
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    sr_ratio=self.sr_ratio_dec,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(self.decoder_embed_dim)

        # Per-patch pixel predictor (MAE-style: predict p^2 * C_out per patch)
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim,
            (self.patch_size**2) * self.out_channels,
            bias=True,
        )

        # Final activation for output field (optional)
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        self._init_weights()

    def _interpolate_pos_embed(
        self, pos_embed: torch.Tensor, Hp: int, Wp: int, has_cls: bool
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings to match (Hp, Wp).
        This is a common ViT practice when transferring to different resolutions.
        """
        if has_cls:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
        else:
            cls_pos = None
            patch_pos = pos_embed

        B, N, C = patch_pos.shape
        H0 = W0 = int(math.sqrt(N))
        if H0 * W0 != N:
            return pos_embed

        patch_pos_2d = patch_pos.reshape(1, H0, W0, C).permute(0, 3, 1, 2)
        patch_pos_2d = F.interpolate(
            patch_pos_2d, size=(Hp, Wp), mode="bicubic", align_corners=False
        )
        patch_pos_new = patch_pos_2d.permute(0, 2, 3, 1).reshape(1, Hp * Wp, C)

        if has_cls:
            return torch.cat([cls_pos, patch_pos_new], dim=1)
        return patch_pos_new

    def _init_weights(self):
        # Positional embeddings (ViT-style)
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.decoder_pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=0.02)

        def _init(m: nn.Module):
            if isinstance(m, nn.Linear):
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens, (Hp, Wp) = self.patch_embed(x)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_tok, tokens], dim=1)
            pos = self._interpolate_pos_embed(self.pos_embed, Hp, Wp, has_cls=True)
        else:
            pos = self._interpolate_pos_embed(self.pos_embed, Hp, Wp, has_cls=False)

        tokens = self.pos_drop(tokens + pos)

        for blk in self.blocks:
            tokens = blk(tokens, Hp, Wp, has_cls=self.use_cls_token)

        tokens = self.norm(tokens)

        # Reconstruction: drop cls token and keep patch tokens
        if self.use_cls_token:
            tokens = tokens[:, 1:, :]

        return tokens, (Hp, Wp)

    def forward_decoder(
        self, tokens: torch.Tensor, grid: tuple[int, int]
    ) -> torch.Tensor:
        Hp, Wp = grid
        tokens = self.decoder_embed(tokens)

        dec_pos = self._interpolate_pos_embed(
            self.decoder_pos_embed, Hp, Wp, has_cls=False
        )
        tokens = tokens + dec_pos

        for blk in self.decoder_blocks:
            tokens = blk(tokens, Hp, Wp, has_cls=False)

        tokens = self.decoder_norm(tokens)
        pred = self.decoder_pred(tokens)
        return pred

    def unpatchify(
        self, patch_pixels: torch.Tensor, grid: tuple[int, int]
    ) -> torch.Tensor:
        """
        Patch tokens -> image (unpatchify).
        This follows the MAE-style reconstruction pattern (He et al., CVPR 2022).
        """
        p = self.patch_size
        Hp, Wp = grid
        B, N, D = patch_pixels.shape
        if N != Hp * Wp:
            raise ValueError(f"Token count mismatch: N={N}, grid={grid}")

        patch_pixels = patch_pixels.reshape(B, Hp, Wp, p, p, self.out_channels)
        patch_pixels = patch_pixels.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patch_pixels.reshape(B, self.out_channels, Hp * p, Wp * p)
        return img

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        latent, grid = self.forward_encoder(x)
        pred = self.forward_decoder(latent, grid)
        y = self.unpatchify(pred, grid)
        y = self.final_activation(y)
        return y

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "name": "VisionTransformer",
                "type": "ViT(AE) + SR-Attention",
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "decoder_embed_dim": self.decoder_embed_dim,
                "decoder_depth": self.decoder_depth,
                "decoder_num_heads": self.decoder_num_heads,
                "sr_ratio_enc": self.sr_ratio_enc,
                "sr_ratio_dec": self.sr_ratio_dec,
                "use_cls_token": self.use_cls_token,
            }
        )
        return info

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, input_shape: tuple[int, int, int, int]) -> int:
        """
        Rough FLOPs estimation.
        For SR attention: cost scales ~ O(N * N/sr^2), inspired by PVT SR-attention complexity.
        """
        B, C, H, W = input_shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Hp * Wp

        patch_flops = C * (self.patch_size**2) * self.embed_dim * N

        sr_e = max(self.sr_ratio_enc, 1)
        Nk_e = max(N // (sr_e * sr_e), 1)
        enc_attn = self.depth * (
            self.num_heads * N * Nk_e * (self.embed_dim // self.num_heads)
        )
        enc_proj = self.depth * (
            3 * self.embed_dim * self.embed_dim * N
            + self.embed_dim * self.embed_dim * N
        )
        enc_mlp = self.depth * (2 * self.embed_dim * int(self.embed_dim * 4.0) * N)

        sr_d = max(self.sr_ratio_dec, 1)
        Nk_d = max(N // (sr_d * sr_d), 1)
        dec_attn = self.decoder_depth * (
            self.decoder_num_heads
            * N
            * Nk_d
            * (self.decoder_embed_dim // self.decoder_num_heads)
        )
        dec_proj = self.decoder_depth * (
            3 * self.decoder_embed_dim * self.decoder_embed_dim * N
            + self.decoder_embed_dim * self.decoder_embed_dim * N
        )
        dec_mlp = self.decoder_depth * (
            2 * self.decoder_embed_dim * int(self.decoder_embed_dim * 4.0) * N
        )

        out_flops = (
            N * self.decoder_embed_dim * (self.patch_size**2) * self.out_channels
        )

        total = (
            patch_flops
            + enc_attn
            + enc_proj
            + enc_mlp
            + dec_attn
            + dec_proj
            + dec_mlp
            + out_flops
        )
        return int(total * B)


# Alias
ViT = VisionTransformer


def create_vit(**kwargs) -> VisionTransformer:
    return VisionTransformer(**kwargs)
