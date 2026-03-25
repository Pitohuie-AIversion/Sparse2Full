import torch
import torch.nn as nn

from ..base import BaseModel
from ..encoders.sparse_input_encoder import SparseInputEncoder
from ..registry import register_model
from .swin_t import SwinTransformerTiny as SwinT

# References (出处):
# - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#   https://arxiv.org/abs/2103.14030
# - LIIF: Learning Continuous Image Representation with Local Implicit Image Function
#   https://arxiv.org/abs/2012.09161
# - “SparseInputEncoder” 属于工程内自定义模块（请在你的论文/代码仓库中给出其来源：自研/改写自哪篇或哪份实现）


@register_model(name="swin_t_with_encoder", aliases=["SwinTWithEncoder"])
class SwinTWithEncoder(BaseModel):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        img_size: int = 128,
        encoder_out_channels: int = 4,
        use_coords: bool = True,
        use_mask: bool = True,
        use_pe: bool = False,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        patch_size: int = 4,
        window_size: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: str = "LayerNorm",
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = "expand_first",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        name: str | None = None,
        post_conv3x3: bool = False,
        use_liif_decoder: bool = False,
        liif_mlp_hidden: int = 64,
        **kwargs,  # Accept extra arguments
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            encoder_out_channels=encoder_out_channels,
            use_coords=use_coords,
            use_mask=use_mask,
            use_pe=use_pe,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample=final_upsample,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            name=name,
            post_conv3x3=post_conv3x3,
            use_liif_decoder=use_liif_decoder,
            liif_mlp_hidden=liif_mlp_hidden,
            **kwargs,
        )
        # 通用编码器消费 img/coords/mask/(pe)
        # References (出处):
        # - 将稀疏观测的“值 + 坐标 + 掩码”拼接/编码为特征的思想在稀疏重建/神经场中很常见；
        #   若你的 SparseInputEncoder 受某篇工作启发（例如 Senseiver / 神经场坐标注入），请在其文件处注明并在此引用。
        self.use_coords = use_coords
        self.use_mask = use_mask
        self.use_pe = use_pe
        self.in_img_channels = 1
        self.encoder = SparseInputEncoder(
            in_img_channels=self.in_img_channels,
            out_channels=encoder_out_channels,
            use_coords=use_coords,
            use_mask=use_mask,
            use_pe=use_pe,
        )

        # SwinT 主干接收编码器输出通道
        # References (出处):
        # - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        #   https://arxiv.org/abs/2103.14030
        self.backbone = SwinT(
            in_channels=encoder_out_channels,
            out_channels=out_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            patch_size=patch_size,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            final_upsample=final_upsample,
        )

        # 可选后处理卷积（重建/复原任务常见的3x3 refinement）
        # Reference (出处):
        # - CNN-based restoration 中常见的后处理卷积（通用工程策略）
        self.post_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if post_conv3x3
            else nn.Identity()
        )

        # 可选 LIIF 解码器：坐标+邻域特征查询（简化版）
        # References (出处):
        # - LIIF: Learning Continuous Image Representation with Local Implicit Image Function
        #   https://arxiv.org/abs/2012.09161
        # - 这里使用“特征 + 2D坐标 -> MLP -> 像素值”的基本范式（简化实现，未显式做局部邻域采样）
        self.use_liif = bool(use_liif_decoder)
        self.liif_mlp = (
            nn.Sequential(
                nn.Linear(out_channels + 2, liif_mlp_hidden),
                nn.GELU(),
                nn.Linear(liif_mlp_hidden, out_channels),
            )
            if self.use_liif
            else None
        )

        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        # 约定通道顺序：img(1) | coords(2) | mask(1) | (pe: optional P)
        # Reference (出处):
        # - “输入中显式携带坐标与mask通道”的接口设计属于工程约定（通用做法）
        B, C, H, W = x.shape
        x_img = x[:, : self.in_img_channels]
        offset = self.in_img_channels
        coords = None
        mask = None
        pe = None

        if self.use_coords and (offset + 2) <= C:
            coords = x[:, offset : offset + 2]
            offset += 2
        if self.use_mask and (offset + 1) <= C:
            mask = x[:, offset : offset + 1]
            offset += 1
        if self.use_pe and offset < C:
            pe = x[:, offset:]
            # Reference (出处):
            # - 将多通道PE做简单聚合（mean）为工程简化策略；若你有参考实现，请补充引用
            if pe is not None and pe.shape[1] > 1:
                pe = pe.mean(dim=1, keepdim=True)

        # 编码器输出特征
        x_enc = self.encoder(x_img, coords=coords, mask=mask, fourier_pe=pe)

        # Swin 主干预测
        y = self.backbone(x_enc)
        y = self.post_conv(y)

        if not self.use_liif:
            return y

        # LIIF：对 HR 坐标点查询像素值
        # References (出处):
        # - LIIF (坐标查询连续表示)
        #   https://arxiv.org/abs/2012.09161
        B, C, Hh, Wh = y.shape
        device = y.device

        # 优先使用输入的真实坐标作为查询坐标
        if coords is not None:
            # Reference (出处):
            # - 坐标对齐/插值为多尺度网络常见处理（通用工程策略）
            if coords.shape[-2:] != (Hh, Wh):
                coord = torch.nn.functional.interpolate(
                    coords, size=(Hh, Wh), mode="bilinear", align_corners=False
                )
            else:
                coord = coords
        else:
            # 回退：生成默认 HR 归一化坐标 [-1,1]
            # Reference (出处):
            # - 归一化坐标网格为隐式表示/坐标网络常用做法（通用）
            ys = torch.linspace(-1, 1, Hh, device=device)
            xs = torch.linspace(-1, 1, Wh, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, Hh, Wh)

        # 简化邻域特征：直接使用 y 的当前像素特征（可扩展为 grid_sample 邻域）
        # Reference (出处):
        # - LIIF 原始实现通常使用局部邻域特征采样；此处为简化版（请在论文/注释中说明差异）
        feat = y

        # 拼接并送入 MLP（逐像素）
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B * Hh * Wh, C)
        coord_flat = coord.permute(0, 2, 3, 1).reshape(B * Hh * Wh, 2)
        mlp_in = torch.cat([feat_flat, coord_flat], dim=1)
        out_flat = self.liif_mlp(mlp_in)
        out = out_flat.reshape(B, Hh, Wh, C).permute(0, 3, 1, 2)
        return out
