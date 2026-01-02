"""
SwinIR 模型实现（Swin Transformer for Image Restoration）

面向图像复原/重建任务的强 Transformer 基线（窗口注意力 + Shifted Window）。
适配 PDEBench 稀疏观测重建任务：严格遵循统一接口
    forward(x[B, C_in, H, W]) -> y[B, C_out, H, W]

默认实现：同分辨率重建（denoising / inpainting / reconstruction），不做超分辨率上采样。
若你后续需要 SR（x2/x4），再在尾部加 upsampler 分支即可。

Reference / 出处：
- SwinIR: Image Restoration Using Swin Transformer
  Jingyun Liang et al., arXiv:2108.10257
- 官方实现（同名项目）：https://github.com/JingyunLiang/SwinIR
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# small utils
# -------------------------
def _to_int(x, default: int):
    """兼容 Hydra/OmegaConf：允许 int / list[int] / tuple[int]"""
    if isinstance(x, (list, tuple)):
        return int(x[0]) if len(x) > 0 else int(default)
    return int(x)


def _to_int_list(x, default: List[int]) -> List[int]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    # 单个 int -> 复制到和 default 等长
    return [int(x) for _ in range(len(default))]


def _pad_to_window_size(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    pad H,W 到 window_size 的整数倍（右/下补零），返回 pad 后张量和 pad 信息。
    pad_info: (pad_left, pad_right, pad_top, pad_bottom)
    """
    if window_size <= 1:
        return x, (0, 0, 0, 0)
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom) for F.pad is (w_left,w_right,h_top,h_bottom)
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, pad, mode="constant", value=0.0)
    return x, (0, pad_w, 0, pad_h)


def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    pl, pr, pt, pb = pad
    if pl == pr == pt == pb == 0:
        return x
    _, _, h, w = x.shape
    return x[:, :, pt : h - pb, pl : w - pr]


# -------------------------
# DropPath
# -------------------------
class DropPath(nn.Module):
    """Stochastic Depth"""

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


# -------------------------
# window ops
# -------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: [B, H, W, C]
    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: [num_windows*B, window_size, window_size, C]
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# -------------------------
# MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------
# Window Attention (with relative position bias)
# -------------------------
class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) with relative position bias.
    出处：Swin Transformer / SwinIR (arXiv:2108.10257)
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias table
        # (2*Ws-1)*(2*Ws-1), nH
        Ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Ws - 1) * (2 * Ws - 1), self.num_heads)
        )

        # relative_position_index: [Ws*Ws, Ws*Ws]
        coords_h = torch.arange(Ws)
        coords_w = torch.arange(Ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Ws, Ws]
        coords_flatten = coords.flatten(1)  # [2, Ws*Ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Ws*Ws, Ws*Ws]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Ws*Ws, Ws*Ws, 2]
        relative_coords[:, :, 0] += Ws - 1
        relative_coords[:, :, 1] += Ws - 1
        relative_coords[:, :, 0] *= 2 * Ws - 1
        relative_position_index = relative_coords.sum(-1)  # [Ws*Ws, Ws*Ws]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows*B, Ws*Ws, C]
            mask: [num_windows, Ws*Ws, Ws*Ws] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, nH, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, nH, N, N]

        # add relative position bias
        rpbt = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpbt = rpbt.view(N, N, -1).permute(2, 0, 1).contiguous()  # [nH, N, N]
        attn = attn + rpbt.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------
# Swin Transformer Block
# -------------------------
class SwinTransformerBlock(nn.Module):
    """
    Shifted window based MSA block.
    出处：SwinIR / Swin Transformer (arXiv:2108.10257)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = input_resolution
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        nh = _to_int(num_heads, default=6)
        # 保证整除
        while nh > 1 and (self.dim % nh != 0):
            nh -= 1
        self.num_heads = nh

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
        if self.shift_size >= self.window_size:
            self.shift_size = self.shift_size % self.window_size

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.dim)
        hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = MLP(self.dim, hidden_dim, drop=drop)

        self.register_buffer("attn_mask", None, persistent=False)

    def _build_attn_mask(self, H: int, W: int, device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=device)  # [1,H,W,1]
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            x_size: (H,W)
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Token length L={L} not match H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # attention mask (lazy build)
        if (self.attn_mask is None) or (self.attn_mask.shape[1] != self.window_size * self.window_size):
            self.attn_mask = self._build_attn_mask(H, W, x.device)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B,H,W,C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------------
# BasicLayer + RSTB
# -------------------------
class BasicLayer(nn.Module):
    """由多个 SwinTransformerBlock 组成的一层（同分辨率）。"""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: List[float],
    ):
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = input_resolution
        self.depth = int(depth)

        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=self.dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else float(drop_path),
                )
            )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """SwinIR 常用：feature map <-> tokens（patch_size=1 的版本）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,H*W,C]
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)


class PatchUnEmbed(nn.Module):
    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        # x: [B,H*W,C] -> [B,C,H,W]
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W
        return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block（SwinIR）
    BasicLayer + Conv（在 2D 上）+ residual
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: List[float],
    ):
        super().__init__()
        self.dim = int(dim)
        self.layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )
        self.patch_unembed = PatchUnEmbed()
        self.patch_embed = PatchEmbed()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        shortcut = x
        x = self.layer(x, x_size)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x + shortcut


# -------------------------
# SwinIR model (same-resolution reconstruction)
# -------------------------
@register_model(name="swinir", aliases=["SwinIR", "swin_ir"])
class SwinIR(BaseModel):
    """
    SwinIR（同分辨率重建版本）

    典型口径（与你现在的 PDEBench 稀疏观测一致）：
      x = concat([obs, mask], dim=1) -> in_channels = C_obs + 1
    输出：
      y: [B, C_out, H, W]

    Reference / 出处：
    - SwinIR: Image Restoration Using Swin Transformer (arXiv:2108.10257)
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        embed_dim: int = 96,
        depths: Union[List[int], Tuple[int, ...]] = (6, 6, 6, 6),
        num_heads: Union[List[int], Tuple[int, ...], int] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        residual: bool = True,
        **kwargs,
    ):
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", kwargs.pop("in_chans", 1))
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", kwargs.pop("num_classes", 1))
        if img_size is None:
            img_size = kwargs.get("img_size", 128)

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.embed_dim = int(embed_dim)
        self.window_size = int(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.residual = bool(residual)

        self.depths = _to_int_list(depths, default=[6, 6, 6, 6])
        self.num_layers = len(self.depths)
        self.num_heads = _to_int_list(num_heads, default=[6, 6, 6, 6])
        if len(self.num_heads) == 1 and self.num_layers > 1:
            self.num_heads = self.num_heads * self.num_layers
        if len(self.num_heads) != self.num_layers:
            raise ValueError(f"num_heads length {len(self.num_heads)} must match depths length {self.num_layers}.")

        # shallow feature extraction
        self.conv_first = nn.Conv2d(self.in_channels, self.embed_dim, 3, 1, 1, bias=True)

        # tokens <-> feature map
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

        # stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # build RSTB layers (all at same resolution for restoration)
        self.layers = nn.ModuleList()
        dp_index = 0
        for i_layer in range(self.num_layers):
            depth_i = self.depths[i_layer]
            heads_i = self.num_heads[i_layer]
            self.layers.append(
                RSTB(
                    dim=self.embed_dim,
                    input_resolution=(self.img_size, self.img_size),  # 训练时常固定；推理时用 x_size 覆盖
                    depth=depth_i,
                    num_heads=heads_i,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[dp_index : dp_index + depth_i],
                )
            )
            dp_index += depth_i

        self.norm = nn.LayerNorm(self.embed_dim)

        # after body conv
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1, bias=True)

        # reconstruction head
        self.conv_last = nn.Conv2d(self.embed_dim, self.out_channels, 3, 1, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        # 采用较稳健的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W]
        """
        inp = x

        # 仅对窗口大小做 pad（同分辨率重建）
        x, pad = _pad_to_window_size(x, self.window_size)
        B, _, Hp, Wp = x.shape
        x_size = (Hp, Wp)

        # shallow features
        x = self.conv_first(x)
        x0 = x  # residual in feature space

        # tokens
        x = self.patch_embed(x)  # [B, Hp*Wp, C]

        # body
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)

        # back to feature map
        x = self.patch_unembed(x, x_size)  # [B, C, Hp, Wp]
        x = self.conv_after_body(x) + x0

        # output
        y = self.conv_last(x)

        # unpad
        y = _unpad(y, pad)

        # 可选：如果你希望学习 residual（仅当输入输出同通道且语义一致时）
        if self.residual and (self.in_channels == self.out_channels):
            y = y + inp

        return y

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "arch": "SwinIR",
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "residual": self.residual,
            }
        )
        return info


def create_swinir(**kwargs) -> SwinIR:
    return SwinIR(**kwargs)
