"""
Swin-UNet模型实现

基于Swin Transformer的UNet架构，支持可选的FNO瓶颈层。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

# -------------------------
# Optional deps: einops
# -------------------------
try:  # pragma: no cover
    from einops import rearrange
except ModuleNotFoundError as err:  # pragma: no cover
    if err.name != "einops":
        raise

    def rearrange(x: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
        """Fallback rearrange: only supports Swin-UNet needed pattern."""
        expected_pattern = "b h w (p1 p2 c)-> b (h p1) (w p2) c"
        if pattern != expected_pattern:
            raise NotImplementedError(
                "Fallback rearrange only supports pattern "
                "'b h w (p1 p2 c)-> b (h p1) (w p2) c'. Install einops for full functionality."
            )
        p1 = axes_lengths.get("p1")
        p2 = axes_lengths.get("p2")
        c = axes_lengths.get("c")
        if None in (p1, p2, c):
            missing = [n for n, v in (("p1", p1), ("p2", p2), ("c", c)) if v is None]
            raise ValueError(f"Missing axes lengths for fallback rearrange: {missing}")
        b, h, w, _ = x.shape
        x = x.reshape(b, h, w, p1, p2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * p1, w * p2, c)
        return x


# -------------------------
# Optional deps: timm
# -------------------------
try:  # pragma: no cover
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ModuleNotFoundError as err:  # pragma: no cover
    if not str(getattr(err, "name", "")).startswith("timm"):
        raise

    class DropPath(nn.Module):
        """Lightweight DropPath compatible with timm."""

        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)
            self.scale_by_keep = bool(scale_by_keep)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device
            )
            random_tensor.floor_()
            if self.scale_by_keep and keep_prob > 0.0:
                x = x / keep_prob
            return x * random_tensor

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value, value)

    def trunc_normal_(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
    ) -> torch.Tensor:
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# -------------------------
# BaseModel import (project-local)
# -------------------------
try:
    from ..base import BaseModel
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from base import BaseModel


# =========================================================
# Utils: window partition/reverse (FIXED, symmetric)
# =========================================================
def window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Split feature map into non-overlapping windows with padding.
    Args:
        x: [B, H, W, C]
    Returns:
        windows: [B*nW, window_size, window_size, C]
        pad_info: (H, W, Hp, Wp) original/padded sizes for reverse
    """
    B, H, W, C = x.shape
    pad_b = (window_size - H % window_size) % window_size
    pad_r = (window_size - W % window_size) % window_size
    if pad_b > 0 or pad_r > 0:
        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]
        x = F.pad(x, (0, pad_r, 0, pad_b))
        x = x.permute(0, 2, 3, 1)  # [B,Hp,Wp,C]
    Hp, Wp = H + pad_b, W + pad_r

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (H, W, Hp, Wp)


def window_reverse(
    windows: torch.Tensor, window_size: int, pad_info: tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Reverse windows to feature map and crop padding.
    Args:
        windows: [B*nW, window_size, window_size, C]
        pad_info: (H, W, Hp, Wp)
    Returns:
        x: [B, H, W, C]
    """
    H, W, Hp, Wp = pad_info
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp != H or Wp != W:
        x = x[:, :H, :W, :]
    return x


# =========================================================
# Core blocks
# =========================================================
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, N]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        """[num_heads, N, N]"""
        N = self.window_size[0] * self.window_size[1]
        rel = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel = rel.view(N, N, -1).permute(2, 0, 1).contiguous()
        return rel

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B*nW, N, C]
            attn_mask: [nW, N, N] additive mask (0 or -100)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, heads, N, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        rel_bias = self._get_rel_pos_bias()  # [heads, N, N]

        # SDPA path (optional)
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            head_dim = C // self.num_heads
            scale_adjust = self.scale * (head_dim**0.5)
            if scale_adjust != 1.0:
                q = q * scale_adjust

            if attn_mask is not None:
                nW = attn_mask.shape[0]
                # [nW, N, N] -> [nW, 1, N, N] then add rel_bias [1, heads, N, N]
                combined = attn_mask.unsqueeze(1) + rel_bias.unsqueeze(0)
                combined = combined.unsqueeze(0).expand(B_ // nW, -1, -1, -1, -1)
                combined = combined.reshape(
                    -1, self.num_heads, N, N
                )  # [B_, heads, N, N]
            else:
                combined = rel_bias.unsqueeze(0).expand(B_, -1, -1, -1)

            def _sdpa(q_, k_, v_, mask_):
                return F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=mask_,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                )

            use_ctx = hasattr(torch.backends, "cuda") and hasattr(
                torch.backends.cuda, "sdp_kernel"
            )
            if use_ctx and self.sdpa_kernel in ("flash", "flash_attention", "fa"):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ):
                    out = _sdpa(q, k, v, combined)
            elif use_ctx and self.sdpa_kernel in (
                "mem_efficient",
                "memory_efficient",
                "me",
            ):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=False, enable_mem_efficient=True
                ):
                    out = _sdpa(q, k, v, combined)
            elif use_ctx and self.sdpa_kernel in ("math", "naive"):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    out = _sdpa(q, k, v, combined)
            else:
                out = _sdpa(q, k, v, combined)

            x = out.transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Classic path
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]
        attn = attn + rel_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (W-MSA/SW-MSA)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        H, W = input_resolution
        if min(H, W) <= self.window_size:
            self.window_size = min(H, W)
            self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
            sdpa_kernel=sdpa_kernel,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        # precompute attn_mask for shifted windows (depends on resolution, set later via set_resolution)
        self.register_buffer("attn_mask", None, persistent=False)

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """Update input_resolution & attn_mask for shifted window attention."""
        self.input_resolution = resolution
        H, W = resolution

        if min(H, W) <= self.window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        if self.shift_size == 0:
            self.attn_mask = None
            return

        # Create mask on padded resolution to ensure divisibility
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        Hp, Wp = H + pad_b, W + pad_r

        # Ensure mask is on the same device as the model parameters
        if self.attn_mask is not None:
            device = self.attn_mask.device
        else:
            device = self.norm1.weight.device if hasattr(self, "norm1") else None

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows, _ = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C], where L = H*W
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            # fall back if resolution is inconsistent (should not happen if tracked correctly)
            H = int(math.sqrt(L))
            W = L // H
            self.set_resolution((H, W))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows (with padding)
        x_windows, pad_info = window_partition(x, self.window_size)  # [B*nW, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, pad_info)  # [B,H,W,C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging: (H,W,C) -> (H/2,W/2,2C) in token form."""

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        self.input_resolution = resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            H = int(math.sqrt(L))
            W = L // H
            self.input_resolution = (H, W)

        if (H % 2 != 0) or (W % 2 != 0):
            # pad to even
            x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
            x_img = F.pad(x_img, (0, W % 2, 0, H % 2))
            x_img = x_img.permute(0, 2, 3, 1)
            H, W = x_img.shape[1], x_img.shape[2]
            x = x_img.reshape(B, H * W, C)
            self.input_resolution = (H, W)

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding: inverse of PatchMerging (token upsample by 2)."""

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        if dim_scale != 2:
            raise ValueError("PatchExpanding currently supports dim_scale=2 only.")
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        self.input_resolution = resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            H = int(math.sqrt(L))
            W = L // H
            self.input_resolution = (H, W)

        x = self.expand(x)  # [B, L, 2C]
        B, L, C2 = x.shape
        x = x.view(B, H, W, C2)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C2 // 4)
        x = x.view(B, -1, C2 // 4)  # [B, 4L, C/2]
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """Conv patch embedding -> tokens."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        # base resolution for absolute_pos_embed
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.current_resolution: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, Ph, Pw]
        self.current_resolution = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        if self.norm is not None:
            x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """A stage (multiple SwinTransformerBlocks) with optional downsample."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = bool(use_checkpoint)

        if isinstance(drop_path, list):
            dpr = drop_path
        else:
            dpr = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_sdpa=use_sdpa,
                sdpa_kernel=sdpa_kernel,
            )
            self.blocks.append(blk)

        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        self.input_resolution = resolution
        for blk in self.blocks:
            blk.set_resolution(resolution)
        if self.downsample is not None:
            self.downsample.set_resolution(resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# =========================================================
# Decoder (token-based, symmetric)
# =========================================================
class SwinUNetDecoder(nn.Module):
    """
    Symmetric decoder:
    - For i=0: deepest stage, no skip, Swin blocks, then expand (if not last)
    - For i>0: fuse with corresponding encoder skip, Swin blocks, then expand
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        depths: list[int],
        num_heads: list[int],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_connections: bool = True,
        patches_resolution: tuple[int, int] = (64, 64),
        use_checkpoint: bool = False,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto",
    ):
        super().__init__()
        self.skip_connections = bool(skip_connections)
        self.patches_resolution = patches_resolution

        num_layers = len(depths)
        assert (
            len(num_heads) == num_layers
        ), "decoder num_heads length must match depths length"
        assert (
            len(decoder_channels) == num_layers
        ), "decoder_channels length must match depths length"

        # decoder resolutions from deepest to shallowest
        # deepest res = patches_resolution // 2^(num_layers-1), then upsample by 2 each stage
        self.stage_resolutions: list[tuple[int, int]] = []
        for i in range(num_layers):
            scale = 2 ** (num_layers - 1 - i)
            self.stage_resolutions.append(
                (patches_resolution[0] // scale, patches_resolution[1] // scale)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.fuse_layers = nn.ModuleList()
        self.swin_layers = nn.ModuleList()
        self.expand_layers = nn.ModuleList()

        dp_cursor = 0
        for i in range(num_layers):
            out_dim = decoder_channels[i]
            res = self.stage_resolutions[i]
            heads = num_heads[i]
            if out_dim % heads != 0:
                raise ValueError(
                    f"Decoder stage {i}: out_dim={out_dim} must be divisible by heads={heads}"
                )

            # fuse (skip only for i>0)
            if self.skip_connections and i > 0:
                skip_dim = encoder_channels[
                    num_layers - 1 - i
                ]  # match: i=1 -> encoder stage num_layers-2
                # FIX: Previous stage output was expanded (dim/2), so input to fuse is halved.
                in_dim = decoder_channels[i - 1] // 2
                fuse = nn.Sequential(
                    nn.Linear(in_dim + skip_dim, out_dim, bias=False),
                    norm_layer(out_dim),
                )
            else:
                fuse = None
            self.fuse_layers.append(fuse if fuse is not None else nn.Identity())

            # swin layer at this resolution
            stage_depth = depths[i]
            stage_dp = dpr[dp_cursor : dp_cursor + stage_depth]
            dp_cursor += stage_depth

            swin = BasicLayer(
                dim=out_dim,
                input_resolution=res,
                depth=stage_depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dp,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_sdpa=use_sdpa,
                sdpa_kernel=sdpa_kernel,
            )
            self.swin_layers.append(swin)

            # expand except last stage
            if i < num_layers - 1:
                expand = PatchExpanding(
                    input_resolution=res,
                    dim=out_dim,
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                expand = None
            self.expand_layers.append(expand if expand is not None else nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        skip_features: list[torch.Tensor] | None,
        start_resolution: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] (deepest tokens)
            skip_features: list of encoder tokens saved at each stage BEFORE downsample
                          order: [stage0 (high res), stage1, ..., stage_{L-1}]
            start_resolution: deepest token resolution (after encoder), e.g. patches_res//2^(L-1)
        Returns:
            feat_map: [B, C, Hpatch, Wpatch] at patches_resolution
        """
        cur_h, cur_w = start_resolution
        num_layers = len(self.swin_layers)

        for i in range(num_layers):
            # set proper resolution to swin blocks
            self.swin_layers[i].set_resolution((cur_h, cur_w))

            # fuse skip for i>0
            if self.skip_connections and i > 0 and skip_features is not None:
                skip_idx = num_layers - 1 - i  # i=1 -> stage2 (for 4 stages)
                skip = skip_features[skip_idx]

                # DEBUG INFO
                # print(f"DEBUG Fuse i={i}: x={x.shape}, skip={skip.shape}")

                if skip.shape[1] != x.shape[1]:
                    # resize skip tokens to current resolution
                    B, N_skip, C_skip = skip.shape
                    Hs = int(math.sqrt(N_skip))
                    Ws = N_skip // Hs
                    skip_img = skip.transpose(1, 2).reshape(B, C_skip, Hs, Ws)
                    skip_img = F.interpolate(
                        skip_img,
                        size=(cur_h, cur_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    skip = skip_img.reshape(B, C_skip, cur_h * cur_w).transpose(1, 2)

                x = torch.cat([x, skip], dim=-1)
                x = self.fuse_layers[i](x)

            else:
                # Identity fuse for i=0 or no skip
                x = self.fuse_layers[i](x)

            # swin refine
            x = self.swin_layers[i](x)

            # expand (if any)
            if not isinstance(self.expand_layers[i], nn.Identity):
                self.expand_layers[i].set_resolution((cur_h, cur_w))
                x = self.expand_layers[i](x)
                cur_h, cur_w = cur_h * 2, cur_w * 2

        # tokens -> image
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = N // H
        feat = x.transpose(1, 2).reshape(B, C, H, W)

        # ensure patch resolution match
        if feat.shape[-2:] != self.patches_resolution:
            feat = F.interpolate(
                feat, size=self.patches_resolution, mode="bilinear", align_corners=False
            )
        return feat


# =========================================================
# Optional FNO Bottleneck (VECTORIZED)
# =========================================================
class FNOBottleneck(nn.Module):
    """Vectorized Fourier Neural Operator bottleneck on deepest tokens."""

    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)

        # complex weights: [Cin, Cout, modes, modes]
        w1 = torch.randn(self.channels, self.channels, self.modes, self.modes, 2) * 0.02
        w2 = torch.randn(self.channels, self.channels, self.modes, self.modes, 2) * 0.02
        self.weights1 = nn.Parameter(torch.view_as_complex(w1))
        self.weights2 = nn.Parameter(torch.view_as_complex(w2))

    def forward(
        self, x: torch.Tensor, resolution: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            resolution: (H, W) for tokens. If None, uses sqrt heuristic.
        """
        B, N, C = x.shape
        if resolution is None:
            H = int(math.sqrt(N))
            W = N // H
        else:
            H, W = resolution
            if H * W != N:
                # fallback
                H = int(math.sqrt(N))
                W = N // H

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]

        out_ft = torch.zeros_like(x_ft)
        mh = min(self.modes, x_ft.shape[2])
        mw = min(self.modes, x_ft.shape[3])

        out_ft[:, :, :mh, :mw] = torch.einsum(
            "b c h w, c o h w -> b o h w",
            x_ft[:, :, :mh, :mw],
            self.weights1[:, :, :mh, :mw],
        )
        out_ft[:, :, -mh:, :mw] = torch.einsum(
            "b c h w, c o h w -> b o h w",
            x_ft[:, :, -mh:, :mw],
            self.weights2[:, :, :mh, :mw],
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        x = x.reshape(B, C, H * W).transpose(1, 2)  # [B, N, C]
        return x


# =========================================================
# Swin-UNet
# =========================================================
@register_model(name="swin_unet", aliases=["SwinUNet"])
class SwinUNet(BaseModel):
    """
    Symmetric Swin-UNet:
    - Encoder: PatchEmbed + Swin stages + PatchMerging
    - Optional bottleneck: FNO on deepest tokens
    - Decoder: symmetric Swin stages + PatchExpanding + token skip fusions
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        # decoder configs (default symmetric)
        decoder_depths: list[int] | None = None,
        decoder_num_heads: list[int] | None = None,
        skip_connections: bool = True,
        # optional FNO bottleneck
        use_fno_bottleneck: bool = False,
        fno_modes: int = 16,
        # output activation
        final_activation: str | None = None,  # None, 'tanh', 'sigmoid'
        # sdpa configs
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = int(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.use_checkpoint = bool(use_checkpoint)
        self.skip_connections = bool(skip_connections)

        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()

        if decoder_depths is None:
            decoder_depths = list(depths[::-1])
        if decoder_num_heads is None:
            decoder_num_heads = list(num_heads[::-1])
        self.decoder_depths = list(decoder_depths)
        self.decoder_num_heads = list(decoder_num_heads)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        base_ph, base_pw = self.patch_embed.patches_resolution
        self.patches_resolution = (base_ph, base_pw)

        # Absolute pos embed (base grid), interpolated to current grid at runtime
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule for encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]

        # Encoder stages
        self.encoder_layers = nn.ModuleList()
        dp_cursor = 0
        encoder_channels: list[int] = []
        cur_dim = embed_dim

        for i_layer in range(len(self.depths)):
            heads = self.num_heads[i_layer]
            if cur_dim % heads != 0:
                raise ValueError(
                    f"Encoder stage {i_layer}: dim={cur_dim} must be divisible by heads={heads}"
                )
            encoder_channels.append(cur_dim)

            stage_depth = self.depths[i_layer]
            stage_dpr = dpr[dp_cursor : dp_cursor + stage_depth]
            dp_cursor += stage_depth

            layer = BasicLayer(
                dim=cur_dim,
                input_resolution=(
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                depth=stage_depth,
                num_heads=heads,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dpr,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(self.depths) - 1) else None,
                use_checkpoint=self.use_checkpoint,
                use_sdpa=self.use_sdpa,
                sdpa_kernel=self.sdpa_kernel,
            )
            self.encoder_layers.append(layer)
            if i_layer < len(self.depths) - 1:
                cur_dim *= 2

        # Norm at deepest
        self.final_encoder_dim = (
            encoder_channels[-1]
            * (2 ** (len(self.depths) - 1))
            // (2 ** (len(self.depths) - 1))
        )
        # Actually deepest dim is embed_dim * 2^(L-1)
        self.final_encoder_dim = embed_dim * (2 ** (len(self.depths) - 1))
        # Because we enforced divisibility stage-by-stage, this is safe.
        self.norm = norm_layer(self.final_encoder_dim)

        # Optional FNO bottleneck
        self.fno_bottleneck = (
            FNOBottleneck(self.final_encoder_dim, fno_modes)
            if use_fno_bottleneck
            else None
        )

        # Build decoder channels (symmetric)
        # encoder_channels list: [embed, 2embed, 4embed, 8embed] (for 4 stages)
        enc_ch = [embed_dim * (2**i) for i in range(len(self.depths))]
        dec_ch = [enc_ch[-1]] + [enc_ch[-(i + 1)] for i in range(1, len(self.depths))]
        dec_ch[-1] = embed_dim  # final decoder stage outputs embed_dim tokens

        self.decoder = SwinUNetDecoder(
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            depths=self.decoder_depths,
            num_heads=self.decoder_num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            skip_connections=self.skip_connections,
            patches_resolution=self.patches_resolution,
            use_checkpoint=self.use_checkpoint,
            use_sdpa=self.use_sdpa,
            sdpa_kernel=self.sdpa_kernel,
        )

        # Final projection
        self.final_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        # Output activation
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        # Conservative initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.005)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        B, C, H_in, W_in = x.shape

        # patch embed
        tokens = self.patch_embed(x)  # [B, N, embed_dim]
        cur_ph, cur_pw = self.patch_embed.current_resolution  # patch grid resolution

        # interpolate abs pos embed to current patch grid
        base_ph, base_pw = self.patches_resolution
        pos = self.absolute_pos_embed.view(1, base_ph, base_pw, self.embed_dim).permute(
            0, 3, 1, 2
        )  # [1,C,ph,pw]
        pos = F.interpolate(
            pos, size=(cur_ph, cur_pw), mode="bilinear", align_corners=False
        )
        pos = pos.permute(0, 2, 3, 1).reshape(1, cur_ph * cur_pw, self.embed_dim)
        tokens = self.pos_drop(tokens + pos)

        # encoder forward + collect skips (before downsample at each stage)
        skips: list[torch.Tensor] = []
        cur_h, cur_w = cur_ph, cur_pw
        for i, layer in enumerate(self.encoder_layers):
            layer.set_resolution((cur_h, cur_w))
            if self.skip_connections:
                skips.append(tokens)
            tokens = layer(tokens)
            if layer.downsample is not None:
                cur_h, cur_w = (cur_h + (cur_h % 2)) // 2, (
                    cur_w + (cur_w % 2)
                ) // 2  # robust when odd

        tokens = self.norm(tokens)

        # optional FNO bottleneck
        if self.fno_bottleneck is not None:
            tokens = self.fno_bottleneck(tokens, resolution=(cur_h, cur_w))

        # decoder: start from deepest resolution (cur_h, cur_w)
        feat = self.decoder(
            tokens,
            skips if self.skip_connections else None,
            start_resolution=(cur_h, cur_w),
        )  # [B, embed_dim, ph, pw]

        # final conv
        y = self.final_conv(feat)

        # ensure output matches input spatial size (strict interface)
        if y.shape[-2:] != (H_in, W_in):
            y = F.interpolate(
                y, size=(H_in, W_in), mode="bilinear", align_corners=False
            )

        y = self.final_activation(y)
        return y


# Alias
SwinUNetModel = SwinUNet
