"""
Standard MLP Model (Pointwise / 1x1 Conv)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model


@register_model(name="mlp", aliases=["MLP", "MLPModel"])
class MLP(BaseModel):
    """
    Standard MLP baseline.

    By default, implements a Pointwise MLP (1x1 Convolutions) which is translation equivariant
    and operates on each grid point independently. This is a strong baseline for dense prediction tasks.

    If `flatten=True`, it behaves as a global MLP (Flatten -> Dense -> Unflatten).

    Unified interface:
        forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        hidden_features: int | list[int] | None = None,
        num_layers: int = 4,
        act_layer: str = "gelu",
        drop: float = 0.0,
        flatten: bool = False,
        **kwargs,
    ):
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", kwargs.pop("in_chans", 1))
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", kwargs.pop("num_classes", 1))
        if img_size is None:
            img_size = kwargs.get("img_size", 128)

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        self.flatten = flatten

        # Resolve hidden features
        if hidden_features is None:
            hidden_features = [64] * (num_layers - 1)
        elif isinstance(hidden_features, int):
            hidden_features = [hidden_features] * (num_layers - 1)

        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.act_layer = act_layer
        self.drop = drop

        # Build layers
        layers = []
        in_dim = in_channels

        # Activation
        if act_layer == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif act_layer == "gelu":
            act_fn = nn.GELU()
        elif act_layer == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")

        # Construct MLP layers
        for hidden_dim in hidden_features:
            layers.append(self._make_layer(in_dim, hidden_dim))
            layers.append(act_fn)
            if drop > 0.0:
                layers.append(self._make_dropout(drop))
            in_dim = hidden_dim

        # Output layer
        layers.append(self._make_layer(in_dim, out_channels))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _make_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.flatten:
            return nn.Linear(in_dim, out_dim)
        else:
            return nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def _make_dropout(self, p: float) -> nn.Module:
        if self.flatten:
            return nn.Dropout(p)
        else:
            return nn.Dropout2d(p)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.flatten:
            # [B, C, H, W] -> [B, H*W, C] -> MLP -> [B, H*W, C_out] -> [B, C_out, H, W]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
            x = self.model(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            # [B, C, H, W] -> 1x1 Conv -> [B, C_out, H, W]
            x = self.model(x)

        return x


# Alias for compatibility
MLPModel = MLP
