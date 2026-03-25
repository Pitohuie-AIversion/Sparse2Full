"""
MLP-Mixer模型

基于MLP的视觉架构，使用两种类型的MLP层：
1) Token-mixing MLP：在 token/patch 维度混合（N = num_patches）
2) Channel-mixing MLP：在通道维度混合（C = embed_dim）

Reference:
    Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision", 2021.
    arXiv:2105.01601
Notes:
    - 原始 MLP-Mixer 用于分类；此实现通过 PatchRestore 适配 dense prediction / reconstruction。
"""

import torch
import torch.nn as nn

from ..base import BaseModel
from ..registry import register_model


class MLP(nn.Module):
    """标准MLP：Linear -> Act -> Drop -> Linear -> Drop"""

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
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """
    Mixer Block

    Token-mixing:
        对每个 channel 独立地在 token 维度(N)做 MLP
        输入 x: [B, N, C] -> transpose -> [B, C, N] -> MLP(N -> hidden -> N)
    Channel-mixing:
        对每个 token 独立地在 channel 维度(C)做 MLP
        输入 x: [B, N, C] -> MLP(C -> hidden -> C)

    Reference:
        MLP-Mixer (Tolstikhin et al., 2021) arXiv:2105.01601
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        mlp_ratio: tuple[float, float] = (0.5, 4.0),
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        token_ratio, channel_ratio = mlp_ratio

        # 关键修正：
        # - token-mixing 的隐藏维度应基于 token 数 seq_len (N)
        # - channel-mixing 的隐藏维度应基于通道数 dim (C)
        tokens_hidden = max(1, int(token_ratio * seq_len))
        channels_hidden = max(1, int(channel_ratio * dim))

        self.norm1 = nn.LayerNorm(dim)
        self.mlp_tokens = MLP(seq_len, tokens_hidden, seq_len, act_layer, drop)

        # 这里用 Dropout 近似 drop_path；如你项目里已有 StochasticDepth，可替换
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp_channels = MLP(dim, channels_hidden, dim, act_layer, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token-mixing
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        )
        # Channel-mixing
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """图像 -> patch 序列嵌入"""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 为保持 MixerBlock 的 seq_len 固定，本实现要求输入尺寸固定为 img_size
        if (H != self.img_size) or (W != self.img_size):
            raise ValueError(
                f"Input size {(H, W)} must match img_size {(self.img_size, self.img_size)}"
            )
        x = self.proj(x)  # [B, C', H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C']
        return x


class PatchRestore(nn.Module):
    """patch 序列 -> 图像"""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int, out_chans: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if N != self.num_patches:
            raise ValueError(
                f"Token length {N} must match num_patches {self.num_patches}"
            )

        x = self.proj(x)  # [B, N, out_chans * P^2]
        x = x.reshape(
            B, self.grid_size, self.grid_size, -1, self.patch_size, self.patch_size
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, -1, self.img_size, self.img_size)
        return x


@register_model(name="MLPMixer", aliases=["mlp_mixer", "MixerModel"])
class MLPMixer(BaseModel):
    """MLP-Mixer (dense-prediction adaptation)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        mlp_ratio: tuple[float, float] = (0.5, 4.0),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    embed_dim, self.num_patches, mlp_ratio, nn.GELU, drop_rate, dpr[i]
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.patch_restore = PatchRestore(img_size, patch_size, embed_dim, out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # [B, N, C]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.patch_restore(x)  # [B, out_channels, H, W]
        return x

    def compute_flops(self, input_shape: tuple[int, ...] = None) -> int:
        """
        简化 FLOPs 估算（不含激活/Norm/Dropout）：
        - PatchEmbed: Conv: Cin*C*P^2*N
        - 每个 block:
            Token MLP: per-channel, (N -> Nt -> N): 2*N*Nt*C
            Channel MLP: per-token, (C -> Ct -> C): 2*C*Ct*N
        - PatchRestore: Linear: C * Cout*P^2 * N
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)

        B, Cin, H, W = input_shape
        if (H != self.img_size) or (W != self.img_size):
            raise ValueError("compute_flops assumes fixed img_size input")

        N = self.num_patches
        P2 = self.patch_size * self.patch_size
        C = self.embed_dim
        Cout = self.out_channels

        # PatchEmbed conv
        patch_embed = Cin * C * P2 * N

        token_ratio, channel_ratio = self.mlp_ratio
        Nt = max(1, int(token_ratio * N))
        Ct = max(1, int(channel_ratio * C))

        per_block = (2 * N * Nt * C) + (2 * C * Ct * N)
        mixer = self.depth * per_block

        patch_restore = C * (Cout * P2) * N

        total = (patch_embed + mixer + patch_restore) * B
        self._flops = int(total)
        return self._flops


# 别名
MixerModel = MLPMixer
