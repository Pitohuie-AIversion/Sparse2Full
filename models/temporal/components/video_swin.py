"""
Video Swin Transformer for Spatiotemporal Prediction.

This module implements a 3D Swin Transformer that processes video (spatiotemporal) data
using 3D Shifted Window Attention. It preserves the 5D tensor structure (B, C, T, H, W)
throughout the network, enabling it to capture local motion and spatiotemporal dependencies
efficiently without flattening.

References:
    - Video Swin Transformer (Liu et al., 2021)
    - Swin Transformer (Liu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from models.registry import register_model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, T, H, W)
        window_size: (Wt, Wh, Ww)
    Returns:
        windows: (B*num_windows, Wt*Wh*Ww, C)
    """
    B, C, T, H, W = x.shape
    wt, wh, ww = window_size
    
    x = x.view(B, C, T // wt, wt, H // wh, wh, W // ww, ww)
    # Permute to (B, T//wt, H//wh, W//ww, wt, wh, ww, C)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    # Merge windows
    windows = windows.view(-1, wt * wh * ww, C)
    return windows

def window_reverse(windows, window_size, B, T, H, W):
    """
    Args:
        windows: (B*num_windows, Wt*Wh*Ww, C)
        window_size: (Wt, Wh, Ww)
        B: Batch size
        T: Total time steps
        H: Height
        W: Width
    Returns:
        x: (B, C, T, H, W)
    """
    wt, wh, ww = window_size
    C = windows.shape[-1]
    
    # Reshape to (B, T//wt, H//wh, W//ww, wt, wh, ww, C)
    x = windows.view(B, T // wt, H // wh, W // ww, wt, wh, ww, C)
    # Permute to (B, C, T//wt, wt, H//wh, wh, W//ww, ww)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    # Merge dimensions
    x = x.view(B, C, T, H, W)
    return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wt, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wt*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() # Placeholder for DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix=None, mask=None):
        """
        x: (B, C, T, H, W)
        """
        if mask_matrix is None and mask is not None:
            mask_matrix = mask
        B, C, T, H, W = x.shape
        shortcut = x
        
        # Reshape for LayerNorm: (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Permute back to (B, C, T, H, W) for partitioning
        shifted_x = shifted_x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size) # (B*nW, Wt*Wh*Ww, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, B, T, H, W) # (B, C, T, H, W)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            # Reshape for roll: (B, T, H, W, C)
            shifted_x = shifted_x.permute(0, 2, 3, 4, 1).contiguous()
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            x = x.permute(0, 4, 1, 2, 3).contiguous() # (B, C, T, H, W)
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        
        # Reshape for Norm2: (B, T, H, W, C)
        shortcut = x
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (B, C, T, H, W)
        x = shortcut + self.drop_path(x)

        return x

@register_model(name="SWVT", aliases=["VideoSwin", "video_swin", "swvt", "VideoSwinPredictor"])
class VideoSwinPredictor(nn.Module):
    """
    Video Swin Transformer (SWVT) for Spatiotemporal Prediction.
    
    基于 3D Shifted Window Attention 的时空联合建模 Transformer，
    支持保持 5D 张量结构并在时空三维窗口内捕获动力学演化与空间依赖。
    支持全场真值输入，原生支持单步推演与多步自回归展开（AR Rollout）。
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 hidden_dim: int = 96, 
                 out_channels: int = 1,
                 num_layers: int = 2, 
                 num_heads: int = 4,
                 window_size: Tuple[int, int, int] = (2, 8, 8),
                 dropout: float = 0.0,
                 use_checkpoint: bool = False,
                 img_size: Optional[Union[int, Tuple[int, int]]] = None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.window_size = tuple(window_size)
        self.use_checkpoint = use_checkpoint
        self.img_size = img_size
        
        # 输入投影 (C -> hidden_dim)
        self.patch_embed = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        
        # 3D Swin 块堆叠
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            shift_size = (0, 0, 0) if i % 2 == 0 else (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2)
            self.layers.append(
                SwinTransformerBlock3D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    drop=dropout,
                    attn_drop=dropout
                )
            )
            
        # 输出投影 (hidden_dim -> out_channels)
        self.output_proj = nn.Conv3d(hidden_dim, out_channels, kernel_size=1)
        
        # 通道适配层（用于多步自回归将 out_channels 反馈回 in_channels）
        if self.in_channels != self.out_channels:
            self.feedback_proj = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        else:
            self.feedback_proj = nn.Identity()

    def _forward_step(self, x_5d: torch.Tensor) -> torch.Tensor:
        """单步 3D Swin 特征处理与预测帧输出
        
        Args:
            x_5d: [B, C, T, H, W]
        Returns:
            pred_next: [B, C_out, 1, H, W]
        """
        B, C, T, H, W = x_5d.shape
        
        # 空间与时间维度 pad 以被 window_size 整除
        pad_t = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x_5d, (0, pad_w, 0, pad_h, 0, pad_t))
        else:
            x_padded = x_5d
            
        # 3D Patch Embedding
        feat = self.patch_embed(x_padded)
        
        # 逐层 3D Shifted Window Attention
        for layer in self.layers:
            feat = layer(feat, mask=None)
            
        # 投影到输出物理场
        out = self.output_proj(feat)
        
        # 移除 Padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :T, :H, :W]
            
        # 取最新时间步特征预测下一时间步 [B, C_out, 1, H, W]
        pred_next = out[:, :, -1:, :, :]
        return pred_next
        
    def forward(
        self, 
        x: torch.Tensor, 
        T_out: int = 1, 
        teacher_seq: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """前向预测
        
        Args:
            x: 输入序列 [B, T_in, C, H, W] 或 [B, C, H, W]
            T_out: 未来推演步数
            teacher_seq: 教师指导信号 [B, T_out, C, H, W]（训练时可选）
            teacher_forcing_ratio: 教师强制比率 [0.0, 1.0]
        Returns:
            predictions: [B, T_out, out_channels, H, W]
        """
        # 统一规范输入为 5D: [B, T_in, C, H, W]
        if x.dim() == 4:
            x_seq = x.unsqueeze(1)  # [B, 1, C, H, W]
        elif x.dim() == 5:
            # 如果输入格式是 [B, C, T, H, W] 且 C==in_channels，但 T != in_channels
            if x.shape[1] == self.in_channels and x.shape[2] != self.in_channels and x.shape[2] <= 64:
                x_seq = x.permute(0, 2, 1, 3, 4).contiguous()
            else:
                x_seq = x
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")
            
        B, T_in, C, H, W = x_seq.shape
        
        if T_out <= 1:
            # 单步预测
            in_tensor = x_seq.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T_in, H, W]
            pred = self._forward_step(in_tensor)  # [B, C_out, 1, H, W]
            return pred.permute(0, 2, 1, 3, 4).contiguous()  # [B, 1, C_out, H, W]

        # 多步自回归推演循环
        curr_seq = x_seq  # [B, T_curr, C, H, W]
        preds = []
        
        for step in range(T_out):
            in_tensor = curr_seq.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T_curr, H, W]
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                step_pred = checkpoint(self._forward_step, in_tensor, use_reentrant=False)
            else:
                step_pred = self._forward_step(in_tensor)  # [B, C_out, 1, H, W]
            step_pred_t = step_pred.permute(0, 2, 1, 3, 4).contiguous()  # [B, 1, C_out, H, W]
            preds.append(step_pred_t)
            
            if step < T_out - 1:
                # 教师强制选择 (使用 PyTorch 统一随机数流)
                if self.training and teacher_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    feed_frame = teacher_seq[:, step:step+1]
                else:
                    feed_frame = step_pred_t

                # 若输出通道与输入通道不一致，投影至输入通道以供下步自回归使用
                if self.in_channels != self.out_channels:
                    feed_frame = self.feedback_proj(feed_frame.squeeze(1)).unsqueeze(1)
                
                # 滑动窗口机制：拼接最新物理场帧
                if curr_seq.shape[1] >= T_in:
                    curr_seq = torch.cat([curr_seq[:, 1:], feed_frame], dim=1)
                else:
                    curr_seq = torch.cat([curr_seq, feed_frame], dim=1)
                    
        return torch.cat(preds, dim=1)  # [B, T_out, out_channels, H, W]

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型架构信息"""
        return {
            "model_type": "SWVT",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_dim": self.hidden_dim,
            "window_size": self.window_size,
            "num_layers": len(self.layers),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
