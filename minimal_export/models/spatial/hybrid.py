"""
Hybrid 模型：Attention ∥ FNO ∥ UNet（多分支 + 融合）

实现要点：
- 统一接口：forward(x[B, C_in, H, W]) -> y[B, C_out, H, W]
- Attention 分支：窗口自注意力 + 可选 shifted-window（带 attention mask，避免 roll 回环混注意力）
- FNO 分支：频域低模态卷积（双权重处理正/负频率切片）
- UNet 分支：经典 encoder-decoder + skip connections
- 融合：concat / add / attention

References (primary):
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", arXiv:1505.04597
  https://arxiv.org/abs/1505.04597
- FNO: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", arXiv:2010.08895
  https://arxiv.org/abs/2010.08895

Related (inspiration for window/shifted-window attention masking strategy):
- Swin Transformer: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", arXiv:2103.14030
  https://arxiv.org/abs/2103.14030
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model


@register_model(name="hybrid", aliases=["HybridModel"])
class HybridModel(BaseModel):
    """混合模型架构：Attention + FNO + UNet"""

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        # 分支配置
        use_attention_branch: bool = True,
        use_fno_branch: bool = True,
        use_unet_branch: bool = True,
        # Attention分支参数
        attn_embed_dim: int = 256,
        attn_num_heads: int = 8,
        attn_num_layers: int = 6,
        attn_window_size: int = 8,
        # FNO分支参数
        fno_modes: int = 16,
        fno_width: int = 64,
        fno_num_layers: int = 4,
        # UNet分支参数
        unet_base_channels: int = 64,
        unet_num_layers: int = 4,
        # 融合参数
        fusion_method: str = "concat",  # 'concat', 'add', 'attention'
        fusion_channels: int = 256,
        **kwargs,
    ):
        if in_channels is None:
            in_channels = kwargs.pop("in_ch", 1)
        if out_channels is None:
            out_channels = kwargs.pop("out_ch", 1)
        if img_size is None:
            img_size = kwargs.get("img_size", 128)

        super().__init__(in_channels, out_channels, img_size, **kwargs)

        assert fusion_method in {
            "concat",
            "add",
            "attention",
        }, f"Unsupported fusion_method={fusion_method}"
        self.use_attention_branch = use_attention_branch
        self.use_fno_branch = use_fno_branch
        self.use_unet_branch = use_unet_branch
        self.fusion_method = fusion_method

        # 输入投影：统一到 fusion_channels
        self.input_proj = nn.Conv2d(self.in_channels, fusion_channels, kernel_size=1)

        self.branches = nn.ModuleDict()
        branch_out_channels: list[int] = []

        if use_attention_branch:
            self.branches["attention"] = AttentionBranch(
                in_channels=fusion_channels,
                embed_dim=attn_embed_dim,
                num_heads=attn_num_heads,
                num_layers=attn_num_layers,
                window_size=attn_window_size,
                img_size=img_size,
            )
            branch_out_channels.append(attn_embed_dim)

        if use_fno_branch:
            self.branches["fno"] = FNOBranch(
                in_channels=fusion_channels,
                modes=fno_modes,
                width=fno_width,
                num_layers=fno_num_layers,
            )
            branch_out_channels.append(fno_width)

        if use_unet_branch:
            self.branches["unet"] = UNetBranch(
                in_channels=fusion_channels,
                base_channels=unet_base_channels,
                num_layers=unet_num_layers,
            )
            # UNetBranch 输出通道为 base_channels（保持轻量）
            branch_out_channels.append(unet_base_channels)

        # 融合
        if fusion_method == "concat":
            fusion_in_channels = sum(branch_out_channels)
        elif fusion_method == "add":
            fusion_in_channels = branch_out_channels[0]
            self.branch_align = nn.ModuleDict()
            for name, out_ch in zip(self.branches.keys(), branch_out_channels):
                if out_ch != fusion_in_channels:
                    self.branch_align[name] = nn.Conv2d(
                        out_ch, fusion_in_channels, kernel_size=1
                    )
        else:  # 'attention'
            fusion_in_channels = fusion_channels
            self.fusion_attention = CrossBranchAttention(
                branch_channels=branch_out_channels,
                fusion_channels=fusion_channels,
            )

        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv2d(fusion_in_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.input_proj(x)  # [B, fusion_channels, H, W]

        outputs: dict[str, torch.Tensor] = {}
        for name, branch in self.branches.items():
            outputs[name] = branch(x)

        if self.fusion_method == "concat":
            fused = torch.cat(list(outputs.values()), dim=1)
        elif self.fusion_method == "add":
            aligned = []
            for name, y in outputs.items():
                if hasattr(self, "branch_align") and name in self.branch_align:
                    y = self.branch_align[name](y)
                aligned.append(y)
            fused = torch.stack(aligned, dim=0).sum(dim=0)
        else:  # 'attention'
            fused = self.fusion_attention(outputs)

        return self.output_head(fused)


class AttentionBranch(nn.Module):
    """窗口注意力分支（带 shifted-window mask，避免回环混注意力）"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        window_size: int = 8,
        img_size: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.img_size = img_size

        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # 注意：这里使用“绝对位置参数图”，会随 img_size 增大显著增加参数量；
        # 若你想更接近 Swin，可改为相对位置偏置（relative position bias）。
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, img_size, img_size) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                WindowAttentionLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.input_proj(x)

        if (H, W) != (self.img_size, self.img_size):
            pos = F.interpolate(
                self.pos_embed, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            pos = self.pos_embed
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


class WindowAttentionLayer(nn.Module):
    """窗口注意力层：Shifted Window 时计算 attention mask（Swin 风格）"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMultiHeadAttention(dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    @torch.no_grad()
    def _build_attn_mask(
        self, H: int, W: int, pad_h: int, pad_w: int, device: torch.device
    ) -> torch.Tensor:
        """
        构造 shifted-window attention mask，避免 roll 后窗口跨区相互注意力（Swin 典型做法）
        返回: [nW, ws*ws, ws*ws]，其中 nW = (H_pad/ws)*(W_pad/ws)
        """
        ws = self.window_size
        ss = self.shift_size
        H_pad, W_pad = H + pad_h, W + pad_w

        img_mask = torch.zeros((1, H_pad, W_pad, 1), device=device)  # 1, H, W, 1
        cnt = 0
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = img_mask.view(1, H_pad // ws, ws, W_pad // ws, ws, 1)
        mask_windows = (
            mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
        )  # [nW, ws*ws]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # [nW, ws*ws, ws*ws]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        # [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).transpose(1, 2).contiguous()

        # shift
        attn_mask = None
        if self.shift_size > 0:
            x_img = x_seq.view(B, H, W, C)
            x_img = torch.roll(
                x_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            x_seq = x_img.view(B, H * W, C)

            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            attn_mask = self._build_attn_mask(H, W, pad_h, pad_w, x.device)

        # Attention
        shortcut = x_seq
        x_seq = self.norm1(x_seq)
        x_seq = self.attn(x_seq, H, W, attn_mask=attn_mask)
        x_seq = shortcut + x_seq

        # MLP
        shortcut = x_seq
        x_seq = self.norm2(x_seq)
        x_seq = self.mlp(x_seq)
        x_seq = shortcut + x_seq

        # reverse shift
        if self.shift_size > 0:
            x_img = x_seq.view(B, H, W, C)
            x_img = torch.roll(
                x_img, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            x_seq = x_img.view(B, H * W, C)

        # [B, H*W, C] -> [B, C, H, W]
        return x_seq.transpose(1, 2).contiguous().view(B, C, H, W)


class WindowMultiHeadAttention(nn.Module):
    """窗口多头注意力（支持 shifted-window mask）"""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, H: int, W: int, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: [B, N, C], N=H*W
        attn_mask: [nW, ws*ws, ws*ws] or None
        """
        B, N, C = x.shape
        ws = self.window_size

        x_img = x.view(B, H, W, C)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_img = F.pad(
                x_img, (0, 0, 0, pad_w, 0, pad_h)
            )  # pad C(0,0), W(0,pad_w), H(0,pad_h)

        H_pad, W_pad = H + pad_h, W + pad_w
        nW = (H_pad // ws) * (W_pad // ws)

        # partition windows: [B*nW, ws*ws, C]
        x_win = x_img.view(B, H_pad // ws, ws, W_pad // ws, ws, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * nW, ws * ws, C)

        # qkv: [B*nW, ws*ws, 3, heads, head_dim] -> [3, B*nW, heads, ws*ws, head_dim]
        qkv = self.qkv(x_win).reshape(B * nW, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*nW, heads, ws*ws, ws*ws]

        if attn_mask is not None:
            # reshape to [B, nW, heads, ws*ws, ws*ws] then add mask [1, nW, 1, ws*ws, ws*ws]
            attn = attn.view(B, nW, self.num_heads, ws * ws, ws * ws)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(B * nW, self.num_heads, ws * ws, ws * ws)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nW, ws * ws, C)
        out = self.proj(out)

        # merge windows back: [B, H_pad, W_pad, C]
        out = out.view(B, H_pad // ws, W_pad // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)

        # unpad
        out = out[:, :H, :W, :].contiguous().view(B, H * W, C)
        return out


class FNOBranch(nn.Module):
    """FNO 分支（参考 FNO: arXiv:2010.08895）"""

    def __init__(
        self, in_channels: int, modes: int = 16, width: int = 64, num_layers: int = 4
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([FNOLayer(width, modes) for _ in range(num_layers)])
        self.output_proj = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


class FNOLayer(nn.Module):
    """
    简化 FNO 2D layer（保留低频模态）：
    - rfft2 得到 [H, W//2+1] 频谱
    - 使用 weights1 处理 [:m1, :m2]，weights2 处理 [-m1:, :m2]（高度维的负频）
    """

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = channels
        self.modes = modes

        # 频域权重（复数）
        self.weights1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(channels, channels, modes, modes, 2) * 0.02
            )
        )
        self.weights2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(channels, channels, modes, modes, 2) * 0.02
            )
        )

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m1 = min(self.modes, H)
        m2 = min(self.modes, W // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]
        out_ft = torch.zeros(
            B, C, H, W // 2 + 1, device=x.device, dtype=torch.complex64
        )

        w1 = self.weights1[:, :, :m1, :m2].to(torch.complex64)
        w2 = self.weights2[:, :, :m1, :m2].to(torch.complex64)

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2].to(torch.complex64), w1
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2].to(torch.complex64), w2
        )

        x_spec = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        x_loc = self.conv1x1(x)

        return self.act(x + x_loc + x_spec)


class UNetBranch(nn.Module):
    """UNet 分支（参考 U-Net: arXiv:1505.04597）"""

    def __init__(self, in_channels: int, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        chs = [in_channels] + [base_channels * (2**i) for i in range(num_layers)]

        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chs[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(chs[i + 1], chs[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(chs[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
            if i < num_layers - 1:
                self.pools.append(nn.MaxPool2d(2))

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1):
            self.upsamples.append(
                nn.ConvTranspose2d(
                    chs[num_layers - i],
                    chs[num_layers - i - 1],
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        chs[num_layers - i - 1] * 2,
                        chs[num_layers - i - 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(chs[num_layers - i - 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        chs[num_layers - i - 1],
                        chs[num_layers - i - 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(chs[num_layers - i - 1]),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.pools):
                skips.append(x)
                x = self.pools[i](x)

        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x


class CrossBranchAttention(nn.Module):
    """跨分支注意力融合：对不同分支特征做像素级权重融合"""

    def __init__(self, branch_channels: list[int], fusion_channels: int):
        super().__init__()
        self.branch_projs = nn.ModuleList(
            [nn.Conv2d(ch, fusion_channels, kernel_size=1) for ch in branch_channels]
        )
        self.attention = nn.Sequential(
            nn.Conv2d(
                fusion_channels * len(branch_channels), fusion_channels, kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, len(branch_channels), kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, branch_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for i, (_, feat) in enumerate(branch_outputs.items()):
            projected.append(self.branch_projs[i](feat))

        concat = torch.cat(projected, dim=1)
        weights = self.attention(concat)  # [B, n_branch, H, W]

        fused = torch.zeros_like(projected[0])
        for i, feat in enumerate(projected):
            fused = fused + weights[:, i : i + 1] * feat
        return fused
