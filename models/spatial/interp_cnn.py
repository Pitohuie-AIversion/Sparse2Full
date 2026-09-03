import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..registry import register_model

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

@register_model(name="BicubicCNN", aliases=["bicubic_cnn", "bicubiccnn"])
class BicubicCNN(BaseModel):
    def __init__(self, in_channels, out_channels, img_size, upscale=1, **kwargs):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        self.upscale = upscale
        self.cnn = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, 1, 1),
            nn.GELU(),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )
        
    def forward(self, x, **kwargs):
        physical_x = x[:, :self.out_channels, :, :]
        
        H, W = physical_x.shape[2:]
        target_H, target_W = self.img_size if isinstance(self.img_size, (list, tuple)) else (self.img_size, self.img_size)
        
        if H != target_H or W != target_W:
            u_tilde = F.interpolate(physical_x, size=(target_H, target_W), mode='bicubic', align_corners=False)
        else:
            u_tilde = physical_x
            
        res = self.cnn(u_tilde)
        return u_tilde + res

@register_model(name="RBFCNN", aliases=["rbf_cnn", "rbfcnn"])
class RBFCNN(BaseModel):
    def __init__(self, in_channels, out_channels, img_size, epsilon=1.0, **kwargs):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        self.epsilon = epsilon
        self.cnn = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, 1, 1),
            nn.GELU(),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )
        
        H, W = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        y_grid, x_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
        self.register_buffer('grid', torch.stack([x_grid / (W-1), y_grid / (H-1)], dim=-1))
        self.register_buffer('cached_W', None)
        self.register_buffer('cached_mask', None)

    def _rbf(self, dist_sq):
        return torch.exp(- (self.epsilon ** 2) * dist_sq)
        
    def forward(self, x, **kwargs):
        physical_x = x[:, :self.out_channels, :, :]
        
        if self.in_channels > self.out_channels:
            mask = x[:, -1, :, :] > 0.5
        else:
            mask = torch.ones_like(physical_x[:, 0, :, :], dtype=torch.bool)
            
        B, C, H, W = physical_x.shape
        u_tilde = torch.zeros_like(physical_x)
        
        # Check target size
        target_H, target_W = self.img_size if isinstance(self.img_size, (list, tuple)) else (self.img_size, self.img_size)
        
        # The mask corresponds to the input physical_x (e.g. 32x32 for SRx4 or 16x16 for Crop).
        # We need to map the observed points to the target grid (e.g. 128x128).
        # We assume the grid we created is for the target size (H_target x W_target).
        # So we need a grid for the input size to get the relative coordinates of the observations.
        
        mask_0 = mask[0]
        if not mask_0.any():
            if H != target_H or W != target_W:
                u_tilde = F.interpolate(physical_x, size=(target_H, target_W), mode='bilinear', align_corners=False)
            else:
                u_tilde = physical_x
            return u_tilde + self.cnn(u_tilde) # fallback
            
        if self.cached_mask is None or not torch.equal(self.cached_mask, mask_0):
            # Compute interpolation matrix W
            # Create a grid for the input shape
            y_in, x_in = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=x.device), torch.arange(W, dtype=torch.float32, device=x.device), indexing='ij')
            grid_in = torch.stack([x_in / (W-1) if W > 1 else x_in, y_in / (H-1) if H > 1 else y_in], dim=-1)
            
            pts_obs = grid_in[mask_0] # [N_obs, 2]
            pts_all = self.grid.view(target_H * target_W, 2) # [target_H * target_W, 2]
            
            dist_sq_obs = torch.cdist(pts_obs, pts_obs, p=2).pow(2)
            K = self._rbf(dist_sq_obs)
            K = K + torch.eye(K.shape[0], device=K.device) * 1e-5
            
            dist_sq_all = torch.cdist(pts_all, pts_obs, p=2).pow(2)
            K_star = self._rbf(dist_sq_all)
            
            try:
                # W = K_star @ K^{-1}
                K_inv = torch.linalg.inv(K)
                W_mat = torch.matmul(K_star, K_inv)
            except RuntimeError:
                # Fallback to pseudoinverse
                K_pinv = torch.linalg.pinv(K)
                W_mat = torch.matmul(K_star, K_pinv)
                
            self.cached_mask = mask_0.clone()
            self.cached_W = W_mat
            
        # Fast interpolation using cached W
        # W: [target_H * target_W, N_obs]
        # vals_obs: [B, C, N_obs]
        vals_obs = physical_x[:, :, mask_0]
        # vals_all: [B, C, target_H * target_W] = vals_obs @ W.T
        vals_all = torch.matmul(vals_obs, self.cached_W.T)
        u_tilde = vals_all.view(B, C, target_H, target_W)
            
        res = self.cnn(u_tilde)
        return u_tilde + res
