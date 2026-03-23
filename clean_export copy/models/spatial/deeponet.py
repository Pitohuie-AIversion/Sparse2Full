"""
DeepONet（2D 适配版）模型实现

DeepONet 是经典的神经算子学习架构：Branch-Net 学习输入函数（或观测）的系数，
Trunk-Net 学习坐标相关的基函数，二者通过内积组合输出，适合“稀疏观测 -> 连续场/稠密场重建”。

本实现针对你的统一接口做了工程化适配：
- 输入：x[B, C_in, H, W]（可包含 sparse value、mask、或多物理量通道）
- 输出：y[B, C_out, H, W]
- Branch：轻量 CNN 编码 + GAP -> 系数向量 b[B, P]
- Trunk：坐标 (x,y) -> 基函数 t[HW, P*C_out]（可选 Fourier features）
- 组合：按输出通道分组做 einsum 内积 -> 重排回图像

Reference:
    Lu et al., "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators"
    (Nature Machine Intelligence, 2021) / arXiv:1910.03193 (architecture lineage)
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


# -------------------------
# Building blocks
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: str = "gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BranchEncoder(nn.Module):
    """
    CNN branch encoder:
    x[B,C,H,W] -> coeff[B,P]
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        channels: List[int] = (64, 128, 256),
        act: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        chs = list(channels)

        layers = []
        prev = in_channels
        for i, c in enumerate(chs):
            # 逐级下采样（stride=2），保证对稀疏观测更稳健
            layers.append(ConvBNAct(prev, c, k=3, s=2, p=1, act=act))
            layers.append(ConvBNAct(c, c, k=3, s=1, p=1, act=act))
            prev = c
        self.backbone = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev, latent_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping for coordinates.
    coords: [N, 2] in [-1,1]
    out: [N, 2 + 2*2*num_freq]  (include raw coords + sin/cos)
    """
    def __init__(self, num_frequencies: int = 8, scale: float = 1.0):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)

        # frequencies: 1,2,4,...  (log-spaced)
        freqs = torch.tensor([2**i for i in range(self.num_frequencies)], dtype=torch.float32) * self.scale
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N,2]
        # [N, 2, F]
        x = coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) * (2.0 * math.pi)
        # [N, 2F] per coord dim, then concat x/y
        sin = torch.sin(x)
        cos = torch.cos(x)
        feat = torch.cat([coords, sin.flatten(1), cos.flatten(1)], dim=1)
        return feat


class TrunkMLP(nn.Module):
    """
    Trunk network:
    coords[N,2] -> basis[N, P*out_ch]
    """
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        hidden: List[int] = (256, 256, 256),
        act: str = "gelu",
        dropout: float = 0.0,
        use_fourier_features: bool = True,
        fourier_frequencies: int = 8,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.use_fourier_features = bool(use_fourier_features)

        if use_fourier_features:
            self.ff = FourierFeatures(num_frequencies=fourier_frequencies, scale=fourier_scale)
            in_dim = 2 + 2 * 2 * fourier_frequencies
        else:
            self.ff = None
            in_dim = 2

        dims = [in_dim] + list(hidden) + [latent_dim * out_channels]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if act == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif act == "gelu":
                    layers.append(nn.GELU())
                elif act == "silu":
                    layers.append(nn.SiLU(inplace=True))
                else:
                    raise ValueError(f"Unsupported act: {act}")
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N,2]
        if self.ff is not None:
            coords = self.ff(coords)
        return self.mlp(coords)  # [N, P*out_ch]


# -------------------------
# DeepONet main model
# -------------------------
@register_model(name="DeepONet", aliases=["deeponet", "deeponet2d"])
class DeepONet(BaseModel):
    """
    DeepONet (2D) for sparse observation reconstruction.

    统一接口：
        forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 128,
        latent_dim: int = 256,                       # P
        branch_channels: List[int] = (64, 128, 256),
        trunk_hidden: List[int] = (256, 256, 256),
        act: str = "gelu",
        dropout: float = 0.0,
        use_fourier_features: bool = True,
        fourier_frequencies: int = 8,
        fourier_scale: float = 1.0,
        add_input_residual: Optional[bool] = None,
        final_activation: Optional[str] = None,
        **kwargs,
    ):
        # 兼容常见别名
        in_channels = kwargs.get("in_ch", kwargs.get("in_chans", in_channels))
        out_channels = kwargs.get("out_ch", kwargs.get("num_classes", out_channels))
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.latent_dim = int(latent_dim)
        self.add_input_residual = (in_channels == out_channels) if add_input_residual is None else bool(add_input_residual)

        self.branch = BranchEncoder(
            in_channels=in_channels,
            latent_dim=self.latent_dim,
            channels=list(branch_channels),
            act=act,
            dropout=dropout,
        )
        self.trunk = TrunkMLP(
            latent_dim=self.latent_dim,
            out_channels=out_channels,
            hidden=list(trunk_hidden),
            act=act,
            dropout=dropout,
            use_fourier_features=use_fourier_features,
            fourier_frequencies=fourier_frequencies,
            fourier_scale=fourier_scale,
        )

        # 可选输出激活
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        # 输出偏置（DeepONet 常用一个全局 bias）
        self.out_bias = nn.Parameter(torch.zeros(out_channels))

        # coord cache: {(H,W,device,dtype): coords[N,2]}
        self._coord_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _get_coords(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key in self._coord_cache:
            return self._coord_cache[key]
        # coords in [-1,1]
        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [N,2]
        self._coord_cache[key] = coords
        return coords

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # 基础数值保护
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        inp = x
        b, _, h, w = x.shape

        # Branch: coeffs [B,P]
        coeff = self.branch(x)  # [B, P]

        # Trunk: basis [N, P*out_ch]
        coords = self._get_coords(h, w, x.device, x.dtype)  # [N,2]
        basis = self.trunk(coords)  # [N, P*out]
        basis = basis.view(1, h * w, self.out_channels, self.latent_dim).expand(b, -1, -1, -1)  # [B,N,out,P]

        # Combine: y[B,N,out] = einsum(coeff[B,P], basis[B,N,out,P])
        y = torch.einsum("bp,bnop->bno", coeff, basis)  # [B,N,out]
        y = y + self.out_bias.view(1, 1, -1)
        y = y.permute(0, 2, 1).contiguous().view(b, self.out_channels, h, w)

        y = self.final_activation(y)

        if self.add_input_residual and (inp.shape[1] == y.shape[1]):
            y = y + inp

        return y

    def get_model_info(self) -> dict:
        return {
            "name": "DeepONet",
            "type": "NeuralOperator",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "img_size": self.img_size,
            "latent_dim": self.latent_dim,
            "add_input_residual": self.add_input_residual,
        }


# 别名
DeepONet2D = DeepONet
