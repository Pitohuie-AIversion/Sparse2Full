import torch
import torch.nn as nn

class SparseInputEncoder(nn.Module):
    """
    A simple encoder to handle sparse inputs including images, coordinates, masks, and positional encodings.
    """
    def __init__(self, in_img_channels=1, out_channels=4, use_coords=True, use_mask=True, use_pe=False):
        super().__init__()
        self.use_coords = use_coords
        self.use_mask = use_mask
        self.use_pe = use_pe
        
        in_channels = in_img_channels
        if use_coords:
            in_channels += 2
        if use_mask:
            in_channels += 1
        if use_pe:
            in_channels += 1
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x_img, coords=None, mask=None, fourier_pe=None):
        tensors = [x_img]
        if self.use_coords and coords is not None:
            tensors.append(coords)
        if self.use_mask and mask is not None:
            tensors.append(mask)
        if self.use_pe and fourier_pe is not None:
            tensors.append(fourier_pe)
            
        x = torch.cat(tensors, dim=1)
        return self.act(self.conv(x))
