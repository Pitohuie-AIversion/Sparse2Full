"""Swin Transformer Tiny (Swin-T) 模型实现

基于Swin Transformer Tiny架构，适配PDEBench稀疏观测重建任务。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]

References (出处):
    - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
      https://arxiv.org/abs/2103.14030
    - Swin Transformer V2 (相对位置偏置/窗口注意力等实现细节的后续版本参考；若你借鉴了V2实现细节可并列引用)
      https://arxiv.org/abs/2111.09883
    - 官方/开源实现参考（若你的实现改写自这些代码，请在仓库中保留对应LICENSE并在此注明）：
        * Microsoft Research Swin Transformer (GitHub): https://github.com/microsoft/Swin-Transformer
        * timm (部分模块/初始化/DropPath等在社区中常用): https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base import BaseModel
from ..registry import register_model


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将特征图分割为窗口

    Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (window_partition)
          https://github.com/microsoft/Swin-Transformer
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """将窗口合并回特征图

    Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (window_reverse)
          https://github.com/microsoft/Swin-Transformer
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力机制

    Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        - 相对位置偏置（relative position bias）细节亦可参考 Swin V2：
          https://arxiv.org/abs/2111.09883
        - 参考实现：microsoft/Swin-Transformer (WindowAttention)
          https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 相对位置偏置表
        # Reference (出处):
        # - Swin Transformer (relative_position_bias_table / relative_position_index)
        #   https://arxiv.org/abs/2103.14030
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 获取每个token对的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        # Reference (出处):
        # - Swin Transformer官方实现中使用trunc_normal_初始化
        #   https://github.com/microsoft/Swin-Transformer
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B*num_windows, N, C] where N = window_size * window_size
            mask: [num_windows, N, N] or None

        Reference (出处):
            - Swin Transformer窗口注意力（W-MSA / SW-MSA）
              https://arxiv.org/abs/2103.14030
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # Reference (出处):
            # - Shifted window attention mask 机制
            #   https://arxiv.org/abs/2103.14030
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块

    Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (SwinTransformerBlock)
          https://github.com/microsoft/Swin-Transformer
        - DropPath / Stochastic Depth 常用实现可参考 timm：
          https://github.com/huggingface/pytorch-image-models
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        H, W = self.input_resolution
        self.H_pad = (H + self.window_size - 1) // self.window_size * self.window_size
        self.W_pad = (W + self.window_size - 1) // self.window_size * self.window_size

        if isinstance(norm_layer, str):
            if norm_layer == "LayerNorm":
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
            else:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Reference (出处):
        # - Stochastic Depth / DropPath (亦称“随机深度”)
        #   原始思想：Deep Networks with Stochastic Depth (Huang et al., 2016)
        #   https://arxiv.org/abs/1603.09382
        # - Swin中常用timm实现风格
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # 计算注意力掩码（shifted window mask）
            # Reference (出处):
            # - Swin Transformer shifted window attention mask
            #   https://arxiv.org/abs/2103.14030
            img_mask = torch.zeros((1, self.H_pad, self.W_pad, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference (出处):
        # - Swin Transformer Block forward (W-MSA / SW-MSA + MLP)
        #   https://arxiv.org/abs/2103.14030
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_b = self.H_pad - H
        pad_r = self.W_pad - W
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, self.H_pad, self.W_pad
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch合并层

    Reference (出处):
        - Swin Transformer (Patch Merging)
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (PatchMerging)
          https://github.com/microsoft/Swin-Transformer
    """

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
        if isinstance(norm_layer, str):
            if norm_layer == "LayerNorm":
                self.norm = nn.LayerNorm(4 * dim)
            else:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
        else:
            self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """Patch扩展层（上采样）

    Reference (出处):
        - Swin-UNet 的 patch expanding / upsampling 结构（若你的decoder来自Swin-UNet思路）
          Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
          https://arxiv.org/abs/2105.05537
        - 同类实现常见于开源 Swin-UNet 代码（注意保留LICENSE并注明来源）
    """

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        if isinstance(norm_layer, str):
            if norm_layer == "LayerNorm":
                self.norm = nn.LayerNorm(dim // dim_scale)
            else:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
        else:
            self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class Mlp(nn.Module):
    """MLP层

    Reference (出处):
        - Transformer FFN / MLP block (Vaswani et al., 2017)
          https://arxiv.org/abs/1706.03762
        - Swin Transformer中同构的MLP实现
          https://arxiv.org/abs/2103.14030
    """

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


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）

    Reference (出处):
        - Deep Networks with Stochastic Depth (Huang et al., 2016)
          https://arxiv.org/abs/1603.09382
        - timm中常用DropPath实现风格（工程参考）
          https://github.com/huggingface/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class BasicLayer(nn.Module):
    """基础Swin Transformer层

    Reference (出处):
        - Swin Transformer hierarchical stages (BasicLayer)
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (BasicLayer)
          https://github.com/microsoft/Swin-Transformer
    """

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
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
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
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """图像到Patch嵌入

    Reference (出处):
        - Vision Transformer patch embedding via Conv2d (ViT)
          https://arxiv.org/abs/2010.11929
        - Swin Transformer PatchEmbed (Conv stem for non-overlapping patches)
          https://arxiv.org/abs/2103.14030
        - 参考实现：microsoft/Swin-Transformer (PatchEmbed)
          https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            if isinstance(norm_layer, str):
                if norm_layer == "LayerNorm":
                    self.norm = nn.LayerNorm(embed_dim)
                else:
                    raise ValueError(f"Unknown norm_layer: {norm_layer}")
            else:
                self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


@register_model(name="swin_t", aliases=["SwinT", "SwinTransformerTiny"])
class SwinTransformerTiny(BaseModel):
    """Swin Transformer Tiny模型

    基于Swin Transformer Tiny架构，适配PDEBench稀疏观测重建任务。
    严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]

    Reference (出处):
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          https://arxiv.org/abs/2103.14030
        - Swin-Unet（如果你采用了U-Net式对称编码器-解码器 + PatchExpand的解码思路）
          https://arxiv.org/abs/2105.05537
        - 参考实现（如为改写/借鉴来源请在仓库保留LICENSE并注明commit/URL）：
            * Microsoft Swin Transformer: https://github.com/microsoft/Swin-Transformer
            * Swin-Unet implementations (various forks; choose the one you actually used)
            * timm utilities: https://github.com/huggingface/pytorch-image-models
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 128,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = "expand_first",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)

        # Reference (出处):
        # - Swin Transformer Tiny config (embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24])
        #   https://arxiv.org/abs/2103.14030

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        if isinstance(norm_layer, str):
            if norm_layer == "LayerNorm":
                norm_layer_func = nn.LayerNorm
            else:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
        else:
            norm_layer_func = norm_layer

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer_func if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入（APE）
        # Reference (出处):
        # - APE 作为可选项在ViT/部分Transformer中常见；Swin原版默认不开启
        #   Swin: https://arxiv.org/abs/2103.14030
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度（Stochastic Depth）
        # Reference (出处):
        # - Huang et al., 2016: https://arxiv.org/abs/1603.09382
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建编码器层（hierarchical stages）
        # Reference (出处):
        # - Swin Transformer encoder stages + PatchMerging
        #   https://arxiv.org/abs/2103.14030
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # 构建解码器层（U-Net式对称上采样 + skip连接）
        # Reference (出处):
        # - Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
        #   https://arxiv.org/abs/2105.05537
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer_up = BasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[: (self.num_layers - 1 - i_layer)]) : sum(
                            depths[: (self.num_layers - 1 - i_layer) + 1]
                        )
                    ],
                    norm_layer=norm_layer,
                    downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        if isinstance(norm_layer, str):
            if norm_layer == "LayerNorm":
                self.norm = nn.LayerNorm(self.num_features)
                self.norm_up = nn.LayerNorm(self.embed_dim)
            else:
                raise ValueError(f"Unknown norm_layer: {norm_layer}")
        else:
            self.norm = norm_layer(self.num_features)
            self.norm_up = norm_layer(self.embed_dim)

        # 最终上采样到像素域
        # Reference (出处):
        # - Swin-Unet / SwinIR 类工作中常用的 PatchExpand / pixel-shuffle / conv 输出头
        #   Swin-Unet: https://arxiv.org/abs/2105.05537
        #   SwinIR: Image Restoration Using Swin Transformer (若你受其重建头部启发可并列引用)
        #   https://arxiv.org/abs/2108.10257
        if final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4,
                dim=embed_dim,
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )
        elif final_upsample == "bilinear":
            print("---final upsample bilinear---")
            self.up = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 16, bias=False),
                nn.PixelShuffle(4),
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Reference (出处):
        # - Swin Transformer官方实现常用trunc_normal_初始化Linear/pos_embed
        #   https://github.com/microsoft/Swin-Transformer
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Reference (出处):
        # - Swin Transformer forward through hierarchical encoder stages
        #   https://arxiv.org/abs/2103.14030
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(
        self, x: torch.Tensor, x_downsample: list[torch.Tensor]
    ) -> torch.Tensor:
        # Reference (出处):
        # - Swin-Unet style decoder with skip connections
        #   https://arxiv.org/abs/2105.05537
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_x4(self, x: torch.Tensor) -> torch.Tensor:
        # Reference (出处):
        # - PatchExpand / PixelShuffle based upsampling heads used in Swin-Unet/SwinIR-like restoration pipelines
        #   Swin-Unet: https://arxiv.org/abs/2105.05537
        #   SwinIR: https://arxiv.org/abs/2108.10257
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        elif self.final_upsample == "bilinear":
            x = self.up[0](x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            x = self.up[1](x)
            x = self.output(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FinalPatchExpand_X4(nn.Module):
    """最终4倍Patch扩展

    Reference (出处):
        - Swin-Unet / SwinIR 等Transformer重建中常见的线性展开 + rearrange 上采样策略
          Swin-Unet: https://arxiv.org/abs/2105.05537
          SwinIR: https://arxiv.org/abs/2108.10257
        - einops rearrange:
          https://github.com/arogozhnikov/einops
    """

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale**2),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


# 别名，保持一致性
SwinT = SwinTransformerTiny
