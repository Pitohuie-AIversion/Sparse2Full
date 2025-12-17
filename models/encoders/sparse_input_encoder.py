import torch
import torch.nn as nn

class SparseInputEncoder(nn.Module):
    def __init__(
        self,
        in_img_channels: int,
        out_channels: int,
        use_coords: bool = True,
        use_mask: bool = True,
        use_pe: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_mask = use_mask
        self.use_pe = use_pe

        img_mid = min(out_channels, max(8, out_channels // 2))
        coord_mid = min(out_channels, max(8, out_channels // 2))
        mask_mid = min(out_channels, max(4, out_channels // 4))
        pe_mid = min(out_channels, max(4, out_channels // 4))

        self.img_proj = nn.Sequential(
            nn.Conv2d(in_img_channels, img_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_mid),
            nn.GELU(),
        )

        self.coord_proj = (
            nn.Sequential(
                nn.Conv2d(2, coord_mid, kernel_size=1),
                nn.BatchNorm2d(coord_mid),
                nn.GELU(),
                nn.Conv2d(coord_mid, coord_mid, kernel_size=3, padding=1),
                nn.BatchNorm2d(coord_mid),
                nn.GELU(),
            )
            if use_coords
            else None
        )

        self.mask_proj = (
            nn.Sequential(
                nn.Conv2d(1, mask_mid, kernel_size=1),
                nn.BatchNorm2d(mask_mid),
                nn.GELU(),
            )
            if use_mask
            else None
        )

        self.pe_proj = (
            nn.Sequential(
                nn.Conv2d(1, pe_mid, kernel_size=1),
                nn.BatchNorm2d(pe_mid),
                nn.GELU(),
            )
            if use_pe
            else None
        )

        total_in_channels = img_mid
        if use_coords:
            total_in_channels += coord_mid
        if use_mask:
            total_in_channels += mask_mid
        if use_pe:
            total_in_channels += pe_mid

        self.fusion = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        x_img: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        fourier_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = [self.img_proj(x_img)]

        if self.use_coords and self.coord_proj is not None:
            if coords is None:
                b, _, h, w = x_img.shape
                coords = torch.zeros(b, 2, h, w, dtype=x_img.dtype, device=x_img.device)
            feats.append(self.coord_proj(coords))

        if self.use_mask and self.mask_proj is not None:
            if mask is None:
                b, _, h, w = x_img.shape
                mask = torch.zeros(b, 1, h, w, dtype=x_img.dtype, device=x_img.device)
            feats.append(self.mask_proj(mask))

        if self.use_pe and self.pe_proj is not None:
            if fourier_pe is None:
                b, _, h, w = x_img.shape
                fourier_pe = torch.zeros(b, 1, h, w, dtype=x_img.dtype, device=x_img.device)
            feats.append(self.pe_proj(fourier_pe))

        x_cat = torch.cat(feats, dim=1)
        x = self.fusion(x_cat)
        return x
