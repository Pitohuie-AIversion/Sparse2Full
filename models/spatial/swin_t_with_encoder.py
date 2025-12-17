import torch
import torch.nn as nn
from ..encoders.sparse_input_encoder import SparseInputEncoder
from .swin_t import SwinTransformerTiny as SwinT

class SwinTWithEncoder(nn.Module):
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
        norm_layer: str = 'LayerNorm',
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = 'expand_first',
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        name: str | None = None,
        post_conv3x3: bool = False,
        use_liif_decoder: bool = False,
        liif_mlp_hidden: int = 64,
    ):
        super().__init__()
        # 通用编码器消费 img/coords/mask/(pe)
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
        self.post_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            if post_conv3x3 else nn.Identity()
        )
        # 可选 LIIF 解码器：坐标+邻域特征查询（简化版）
        self.use_liif = bool(use_liif_decoder)
        self.liif_mlp = nn.Sequential(
            nn.Linear(out_channels + 2, liif_mlp_hidden),
            nn.GELU(),
            nn.Linear(liif_mlp_hidden, out_channels)
        ) if self.use_liif else None
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        # 约定通道顺序：img(1) | coords(2) | mask(1) | (pe: optional P)
        B, C, H, W = x.shape
        x_img = x[:, :self.in_img_channels]
        offset = self.in_img_channels
        coords = None
        mask = None
        pe = None
        if self.use_coords and (offset + 2) <= C:
            coords = x[:, offset:offset+2]
            offset += 2
        if self.use_mask and (offset + 1) <= C:
            mask = x[:, offset:offset+1]
            offset += 1
        if self.use_pe and offset < C:
            pe = x[:, offset:]
            if pe is not None and pe.shape[1] > 1:
                pe = pe.mean(dim=1, keepdim=True)
        x_enc = self.encoder(x_img, coords=coords, mask=mask, fourier_pe=pe)
        y = self.backbone(x_enc)
        y = self.post_conv(y)
        if not self.use_liif:
            return y
            
        # LIIF：对 HR 坐标点查询像素值
        B, C, Hh, Wh = y.shape
        device = y.device
        
        # 优先使用输入的真实坐标作为查询坐标
        if coords is not None:
            # 确保坐标分辨率匹配
            if coords.shape[-2:] != (Hh, Wh):
                # 如果输入坐标分辨率与输出不一致（例如输入是LR坐标），则需要插值到HR
                # 注意：在当前架构中，coords通常是HR网格
                coord = torch.nn.functional.interpolate(
                    coords, size=(Hh, Wh), mode='bilinear', align_corners=False
                )
            else:
                coord = coords
        else:
            # 回退：生成默认 HR 归一化坐标 [-1,1]
            ys = torch.linspace(-1, 1, Hh, device=device)
            xs = torch.linspace(-1, 1, Wh, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, Hh, Wh)  # [B,2,H,W]
            
        # 简化邻域特征：直接使用 y 的当前像素特征（可扩展为 grid_sample 邻域）
        feat = y  # [B,C,H,W]
        # 拼接并送入 MLP（逐像素）
        feat_flat = feat.permute(0,2,3,1).reshape(B*Hh*Wh, C)
        coord_flat = coord.permute(0,2,3,1).reshape(B*Hh*Wh, 2)
        mlp_in = torch.cat([feat_flat, coord_flat], dim=1)
        out_flat = self.liif_mlp(mlp_in)
        out = out_flat.reshape(B, Hh, Wh, C).permute(0,3,1,2)
        return out
