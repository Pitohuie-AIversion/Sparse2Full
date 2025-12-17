"""Swin-UNet模型实现

基于Swin Transformer的UNet架构，支持可选的FNO瓶颈层。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - 依赖存在时直接使用
    from einops import rearrange
except ModuleNotFoundError as err:  # pragma: no cover - 测试环境缺失einops时使用
    if err.name != "einops":
        raise

    def rearrange(x: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
        """用于测试环境的极简rearrange实现，仅支持Swin-UNet所需的模式。"""

        expected_pattern = "b h w (p1 p2 c)-> b (h p1) (w p2) c"
        if pattern != expected_pattern:
            raise NotImplementedError(
                "Fallback rearrange only supports pattern 'b h w (p1 p2 c)-> b (h p1) (w p2) c'. "
                "Install einops for full functionality."
            )

        p1 = axes_lengths.get("p1")
        p2 = axes_lengths.get("p2")
        c = axes_lengths.get("c")
        if None in (p1, p2, c):
            missing = [
                name for name, value in (("p1", p1), ("p2", p2), ("c", c))
                if value is None
            ]
            raise ValueError(f"Missing axes lengths for fallback rearrange: {missing}")

        b, h, w, _ = x.shape
        # 使用reshape而不是view以兼容channels_last等非连续内存格式
        x = x.reshape(b, h, w, p1, p2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * p1, w * p2, c)
        return x

try:  # pragma: no cover - 简单的兼容性分支（兼容timm>=0.9）
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ModuleNotFoundError as err:  # pragma: no cover - timm 未安装时的回退实现
    # 在部分环境下，ModuleNotFoundError.name 可能为 'timm.layers' 或 'timm'
    if not str(getattr(err, "name", "")).startswith("timm"):
        raise

    class DropPath(nn.Module):
        """timm中的DropPath的轻量级实现。

        仅依赖于PyTorch, 用于在测试环境缺失timm库时保持模型可用。
        """

        def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)
            self.scale_by_keep = scale_by_keep

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x

            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()

            if self.scale_by_keep and keep_prob > 0.0:
                x = x / keep_prob

            return x * random_tensor

    def to_2tuple(value):
        """将输入转换为长度为2的tuple。"""

        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value, value)

    def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.,
                      a: float = -2., b: float = 2.) -> torch.Tensor:
        """PyTorch中的截断正态初始化，用于兼容timm依赖。"""

        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..base import BaseModel
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from base import BaseModel


class WindowAttention(nn.Module):
    """窗口多头自注意力机制"""
    
    def __init__(
        self, 
        dim: int, 
        window_size: Tuple[int, int], 
        num_heads: int, 
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None, 
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto"
    ):
        super().__init__()
        
        # 确保dim能被num_heads整除
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 获取相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算相对位置偏置 [num_heads, N, N]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H, N, N]

        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            # SDPA路径：内部使用1/sqrt(d)缩放，需按qk_scale做调整
            head_dim = C // self.num_heads
            scale_adjust = self.scale * (head_dim ** 0.5)  # 使SDPA的1/sqrt(d)得到等效于self.scale的缩放
            if scale_adjust != 1.0:
                q = q * scale_adjust

            # 组合掩码与相对位置偏置为加性mask
            if mask is not None:
                nW = mask.shape[0]
                # [nW, N, N] + [1, H, N, N] -> [nW, H, N, N]
                combined = mask.unsqueeze(1) + relative_position_bias.unsqueeze(0)
                # 扩展到批次维度 [B_//nW, nW, H, N, N] → [B_, H, N, N]
                combined = combined.unsqueeze(0).expand(B_ // nW, -1, -1, -1, -1).reshape(-1, self.num_heads, N, N)
            else:
                # 仅相对位置偏置 [1, H, N, N] → [B_, H, N, N]
                combined = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)

            def _sdpa(q, k, v, attn_mask):
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False
                )

            # 根据sdpa_kernel选择后端（flash/mem_efficient/math/auto）
            use_ctx = hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel')
            if use_ctx and self.sdpa_kernel in ("flash", "flash_attention", "fa"):
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    attn_out = _sdpa(q, k, v, combined)
            elif use_ctx and self.sdpa_kernel in ("mem_efficient", "memory_efficient", "me"):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                    attn_out = _sdpa(q, k, v, combined)
            elif use_ctx and self.sdpa_kernel in ("math", "naive"):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    attn_out = _sdpa(q, k, v, combined)
            else:
                attn_out = _sdpa(q, k, v, combined)
            x = attn_out.transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            # 传统路径
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
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
    """Swin Transformer块"""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        num_heads: int,
        window_size: int = 7, 
        shift_size: int = 0, 
        mlp_ratio: float = 4.,
        qkv_bias: bool = True, 
        qk_scale: Optional[float] = None, 
        drop: float = 0., 
        attn_drop: float = 0.,
        drop_path: float = 0., 
        act_layer: nn.Module = nn.GELU, 
        norm_layer: nn.Module = nn.LayerNorm,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto"
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 确保dim能被num_heads整除
        if dim % num_heads != 0:
            adjusted_dim = (dim // num_heads) * num_heads
            print(f"Warning: Adjusted SwinTransformerBlock dim from {dim} to {adjusted_dim} to be divisible by {num_heads} heads")
            dim = adjusted_dim
        
        # 更新实例变量
        self.dim = dim
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_sdpa=self.use_sdpa, sdpa_kernel=self.sdpa_kernel
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_expected, W_expected = self.input_resolution
        B, L, C = x.shape
        # 允许在分辨率不匹配时进行自适应推断（避免因DataParallel/reshape导致的L不等于H*W）
        if L != H_expected * W_expected:
            H = int(math.sqrt(L))
            W = int(L // H)
            if H * W != L:
                # 回退到近似方形分辨率
                H = int(math.sqrt(L))
                W = H
        else:
            H, W = H_expected, W_expected

        # 如果输入维度与期望维度不匹配，进行调整
        if C != self.dim:
            # 使用线性层调整维度
            if not hasattr(self, 'dim_adjust'):
                self.dim_adjust = nn.Linear(C, self.dim).to(x.device)
            x = self.dim_adjust(x)
            C = self.dim

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 计算填充以便窗口划分
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size
        pad_right = (self.window_size - W % self.window_size) % self.window_size
        Hp, Wp = H + pad_bottom, W + pad_right

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 进行必要的填充（确保Hp、Wp能被window_size整除）
        if pad_bottom > 0 or pad_right > 0:
            shifted_x = shifted_x.permute(0, 3, 1, 2)
            shifted_x = F.pad(shifted_x, (0, pad_right, 0, pad_bottom))
            shifted_x = shifted_x.permute(0, 2, 3, 1)

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        if self.shift_size > 0:
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_windows = self.attn(x_windows, mask=attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 去除填充
        if pad_bottom > 0 or pad_right > 0:
            shifted_x = shifted_x[:, :H, :W, :]

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch合并层"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """Patch扩展层 - 对称于PatchMerging的上采样操作"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        B, L, C = x.shape
        assert C == 2 * self.dim, "expand dimension wrong"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """基础Swin Transformer层"""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        depth: int, 
        num_heads: int,
        window_size: int, 
        mlp_ratio: float = 4., 
        qkv_bias: bool = True, 
        qk_scale: Optional[float] = None,
        drop: float = 0., 
        attn_drop: float = 0., 
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm, 
        downsample: Optional[nn.Module] = None, 
        use_checkpoint: bool = False,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto"
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()

        # 确保dim能被num_heads整除
        if dim % num_heads != 0:
            adjusted_dim = (dim // num_heads) * num_heads
            print(f"Warning: Adjusted BasicLayer dim from {dim} to {adjusted_dim} to be divisible by {num_heads} heads")
            dim = adjusted_dim
        
        # 更新实例变量
        self.dim = dim
        
        # 构建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_sdpa=self.use_sdpa,
                sdpa_kernel=self.sdpa_kernel
            )
            for i in range(depth)
        ])

        # patch合并层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            blk.input_resolution = self.input_resolution
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            self.downsample.input_resolution = self.input_resolution
            x = self.downsample(x)
            # 更新分辨率为下采样后的尺寸
            self.input_resolution = (self.input_resolution[0] // 2, self.input_resolution[1] // 2)
        return x


class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 允许任意尺寸，Conv2d 会产生对应的patch栅格
        x = self.proj(x)
        # 记录当前patch分辨率，供位置嵌入与解码器使用
        self.current_resolution = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP层"""
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.
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


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将特征图分割为窗口（自动对非整除尺寸进行填充）"""
    B, H, W, C = x.shape
    pad_bottom = (window_size - H % window_size) % window_size
    pad_right = (window_size - W % window_size) % window_size
    if pad_bottom > 0 or pad_right > 0:
        # 转为NCHW进行填充
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, pad_right, 0, pad_bottom))
        x = x.permute(0, 2, 3, 1)
        H = H + pad_bottom
        W = W + pad_right
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """将窗口合并回特征图（支持填充后的尺寸）"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinUNetDecoder(nn.Module):
    """对称的Swin-UNet解码器 - 使用PatchExpanding和Swin Transformer块"""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        depths: List[int],
        num_heads: List[int],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_connections: bool = True,
        patches_resolution: Tuple[int, int] = (64, 64),
        use_checkpoint: bool = False,
        use_sdpa: bool = False,
        sdpa_kernel: str = "auto"
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.depths = depths
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.patches_resolution = patches_resolution
        self.use_sdpa = bool(use_sdpa)
        self.sdpa_kernel = str(sdpa_kernel).lower()
        
        # 计算解码器的分辨率序列（从小到大）
        num_layers = len(depths)
        self.decoder_resolutions = []
        for i in range(num_layers):
            # 解码器分辨率：从最小开始逐步增大
            scale = 2 ** (num_layers - 1 - i)
            res = (patches_resolution[0] // scale, patches_resolution[1] // scale)
            self.decoder_resolutions.append(res)
        
        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建解码器层
        self.decoder_layers = nn.ModuleList()
        
        # 从最深层开始构建解码器
        for i in range(num_layers):
            # 当前层的输入通道数（来自编码器或上一个解码器层）
            if i == 0:
                # 第一层：来自编码器最深层
                in_dim = encoder_channels[-1]
            else:
                # 后续层：来自上一个解码器层
                in_dim = decoder_channels[i-1] if i-1 < len(decoder_channels) else encoder_channels[-(i)]
            
            # 输出通道数
            out_dim = decoder_channels[i] if i < len(decoder_channels) else in_dim // 2
            
            # 跳跃连接的通道数
            if skip_connections and i > 0:  # 第一层（最深层）没有跳跃连接
                skip_idx = num_layers - 1 - i  # 对应的编码器层索引
                skip_dim = encoder_channels[skip_idx] if skip_idx >= 0 and skip_idx < len(encoder_channels) else 0
            else:
                skip_dim = 0
            
            # 如果有跳跃连接，需要融合层
            if skip_dim > 0:
                # 跳跃连接融合层
                fuse_layer = nn.Sequential(
                    nn.Linear(in_dim + skip_dim, out_dim, bias=False),
                    norm_layer(out_dim)
                )
            else:
                fuse_layer = None
            
            # 确保解码器维度能被注意力头数整除
            decoder_num_heads = num_heads[i]
            if out_dim % decoder_num_heads != 0:
                out_dim = (out_dim // decoder_num_heads) * decoder_num_heads
                print(f"Warning: Adjusted decoder layer {i} dim to {out_dim} to be divisible by {decoder_num_heads} heads")
            
            # Swin Transformer块
            swin_layer = BasicLayer(
                dim=out_dim,
                input_resolution=self.decoder_resolutions[i],
                depth=depths[i],
                num_heads=decoder_num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=None,  # 解码器不需要下采样
                use_checkpoint=use_checkpoint,
                use_sdpa=self.use_sdpa,
                sdpa_kernel=self.sdpa_kernel
            )
            
            # PatchExpanding层（除了最后一层）
            if i < num_layers - 1:
                patch_expand = PatchExpanding(
                    input_resolution=self.decoder_resolutions[i],
                    dim=out_dim,
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                patch_expand = None
            
            # 创建解码器层字典，只包含nn.Module对象
            layer_dict = nn.ModuleDict({
                'swin': swin_layer,
            })
            
            if fuse_layer is not None:
                layer_dict['fuse'] = fuse_layer
            if patch_expand is not None:
                layer_dict['expand'] = patch_expand
            
            # 将非Module属性存储为普通属性
            layer_dict.skip_dim = skip_dim
            layer_dict.out_dim = out_dim
            
            self.decoder_layers.append(layer_dict)
    
    def forward(self, x: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """解码器前向传播
        
        Args:
            x: 编码器输出 [B, N, C]
            skip_features: 跳跃连接特征列表（从浅到深）
            
        Returns:
            解码后的特征图 [B, C, H, W]
        """
        B, N, C = x.shape
        
        # 逐层解码
        for i, layer_dict in enumerate(self.decoder_layers):
            fuse_layer = layer_dict['fuse'] if 'fuse' in layer_dict else None
            swin_layer = layer_dict['swin']
            expand_layer = layer_dict['expand'] if 'expand' in layer_dict else None
            skip_dim = layer_dict.skip_dim
            
            # 处理跳跃连接
            if (skip_features is not None and 
                fuse_layer is not None and
                skip_dim > 0 and
                i > 0):  # 第一层（最深层）没有跳跃连接
                # 跳跃连接索引：对应的编码器层
                skip_idx = len(skip_features) - i
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip_feat = skip_features[skip_idx]
                    
                    # 确保跳跃连接特征的分辨率匹配
                    B_skip, N_skip, C_skip = skip_feat.shape
                    B_x, N_x, C_x = x.shape
                    
                    if N_skip != N_x:
                        # 需要调整跳跃连接特征的分辨率
                        H_skip = W_skip = int(math.sqrt(N_skip))
                        H_x = W_x = int(math.sqrt(N_x))
                        
                        # 转换为图像格式进行插值
                        skip_feat = skip_feat.transpose(1, 2).view(B_skip, C_skip, H_skip, W_skip)
                        skip_feat = F.interpolate(skip_feat, size=(H_x, W_x), mode='bilinear', align_corners=False)
                        skip_feat = skip_feat.view(B_skip, C_skip, -1).transpose(1, 2)  # [B, N_x, C_skip]
                    
                    # 融合跳跃连接
                    x = torch.cat([x, skip_feat], dim=-1)  # [B, N, C + skip_C]
                    x = fuse_layer(x)  # [B, N, out_dim]
            
            # Swin Transformer块
            # 设置当前层的分辨率
            if hasattr(self, 'current_input_resolution') and self.current_input_resolution is not None:
                swin_layer.input_resolution = self.current_input_resolution
            x = swin_layer(x)
            
            # PatchExpanding（除了最后一层）
            if expand_layer is not None:
                expand_layer.input_resolution = self.current_input_resolution
                x = expand_layer(x)
                # 上采样后更新分辨率
                self.current_input_resolution = (self.current_input_resolution[0] * 2, self.current_input_resolution[1] * 2)
        
        # 转换为图像格式
        B, N, C = x.shape
        if hasattr(self, 'current_input_resolution') and self.current_input_resolution is not None:
            H, W = self.current_input_resolution
        else:
            # 近似方形回退
            H = int(math.sqrt(N))
            W = int(N // H)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        # 确保输出尺寸与目标匹配
        target_size = self.patches_resolution
        if x.shape[-2:] != target_size:
            # 使用双线性插值调整到目标尺寸
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class SwinUNet(BaseModel):
    """完全对称的Swin-UNet模型
    
    基于Swin Transformer的UNet架构，编码器和解码器都使用Swin Transformer块。
    严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        # 解码器参数
        decoder_depths: Optional[List[int]] = None,
        decoder_num_heads: Optional[List[int]] = None,
        skip_connections: bool = True,
        # FNO瓶颈参数（可选）
        use_fno_bottleneck: bool = False,
        fno_modes: int = 16,
        # 最终激活函数
        final_activation: Optional[str] = None,  # None, 'tanh', 'sigmoid'
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        self.skip_connections = skip_connections
        self.use_fno_bottleneck = use_fno_bottleneck
        # SDPA/Flash Attention 配置
        self.use_sdpa = bool(kwargs.get('use_sdpa', False))
        self.sdpa_kernel = str(kwargs.get('sdpa_kernel', 'auto')).lower()
        
        # 解码器参数（默认与编码器对称）
        if decoder_depths is None:
            decoder_depths = depths[::-1]  # 倒序
        if decoder_num_heads is None:
            decoder_num_heads = num_heads[::-1]  # 倒序
        
        self.decoder_depths = decoder_depths
        self.decoder_num_heads = decoder_num_heads
        
        # 分割图像为patch并嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建编码器层
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_num_heads = num_heads[i_layer]
            
            # 确保维度能被注意力头数整除
            if layer_dim % layer_num_heads != 0:
                # 调整维度使其能被注意力头数整除
                layer_dim = (layer_dim // layer_num_heads) * layer_num_heads
                print(f"Warning: Adjusted layer {i_layer} dim from {int(embed_dim * 2 ** i_layer)} to {layer_dim} to be divisible by {layer_num_heads} heads")
            
            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=layer_num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < len(depths) - 1) else None,
                use_checkpoint=use_checkpoint,
                use_sdpa=self.use_sdpa,
                sdpa_kernel=self.sdpa_kernel
            )
            self.encoder_layers.append(layer)

        # 计算最后一层的实际维度（考虑维度调整）
        final_layer_dim = int(embed_dim * 2 ** (len(depths) - 1))
        final_layer_num_heads = num_heads[-1]
        if final_layer_dim % final_layer_num_heads != 0:
            final_layer_dim = (final_layer_dim // final_layer_num_heads) * final_layer_num_heads
        
        self.norm = norm_layer(final_layer_dim)

        # FNO瓶颈层（可选）
        if use_fno_bottleneck:
            # 使用实际调整后的最终层维度
            self.fno_bottleneck = FNOBottleneck(final_layer_dim, fno_modes)
        else:
            self.fno_bottleneck = None

        # 对称的Swin-UNet解码器
        encoder_channels = []
        for i in range(len(depths)):
            layer_dim = int(embed_dim * 2 ** i)
            layer_num_heads = num_heads[i]
            
            # 确保维度能被注意力头数整除
            if layer_dim % layer_num_heads != 0:
                layer_dim = (layer_dim // layer_num_heads) * layer_num_heads
            
            encoder_channels.append(layer_dim)
        
        # 解码器通道数：从最深层开始逐步减少
        decoder_channels = []
        for i in range(len(depths)):
            if i == 0:
                # 第一层：保持最深层的通道数
                decoder_channels.append(encoder_channels[-1])
            else:
                # 后续层：逐步减少通道数
                decoder_channels.append(encoder_channels[-(i+1)])
        
        # 最后一层输出通道数应该与输入图像通道数匹配
        # 确保最后一层的维度也能被对应的注意力头数整除
        final_decoder_dim = embed_dim
        final_decoder_num_heads = self.decoder_num_heads[-1] if self.decoder_num_heads else 3
        if final_decoder_dim % final_decoder_num_heads != 0:
            final_decoder_dim = (final_decoder_dim // final_decoder_num_heads) * final_decoder_num_heads
        decoder_channels[-1] = final_decoder_dim
        
        self.decoder = SwinUNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            depths=self.decoder_depths,
            num_heads=self.decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            skip_connections=skip_connections,
            patches_resolution=patches_resolution,
            use_checkpoint=use_checkpoint,
            use_sdpa=self.use_sdpa,
            sdpa_kernel=self.sdpa_kernel
        )

        # 最终输出层 - 使用实际的解码器输出通道数
        final_decoder_channels = decoder_channels[-1]
        self.final_conv = nn.Conv2d(final_decoder_channels, out_channels, kernel_size=1)
        
        # 最终激活函数
        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

        # 初始化权重
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """初始化权重 - 使用更保守的初始化策略"""
        if isinstance(m, nn.Linear):
            # 使用更小的标准差进行初始化，提高数值稳定性
            trunc_normal_(m.weight, std=.005)  # 进一步减少到0.005
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 使用更保守的Xavier初始化
            nn.init.xavier_uniform_(m.weight, gain=0.3)  # 进一步减少gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # 批归一化层的保守初始化
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（coords, mask等）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, N, embed_dim]
        # 自适应位置嵌入到当前patch栅格尺寸
        if hasattr(self.patch_embed, 'current_resolution') and self.patch_embed.current_resolution is not None:
            cur_h, cur_w = self.patch_embed.current_resolution
            base_h = self.patches_resolution[0]
            base_w = self.patches_resolution[1]
            pos = self.absolute_pos_embed.view(1, base_h, base_w, self.embed_dim).permute(0, 3, 1, 2)  # [1,C,H,W]
            pos = F.interpolate(pos, size=(cur_h, cur_w), mode='bilinear', align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, cur_h * cur_w, self.embed_dim)
        else:
            pos = self.absolute_pos_embed
        x = x + pos
        x = self.pos_drop(x)

        # 编码器前向传播，保存跳跃连接
        skip_connections = []
        cur_h, cur_w = self.patch_embed.current_resolution if hasattr(self.patch_embed, 'current_resolution') else self.patches_resolution
        for layer in self.encoder_layers:
            layer.input_resolution = (cur_h, cur_w)
            if self.skip_connections:
                # 保存当前特征用于跳跃连接
                skip_connections.append(x)
            x = layer(x)
            # 下采样后更新分辨率
            cur_h, cur_w = (cur_h // 2, cur_w // 2) if layer.downsample is not None else (cur_h, cur_w)

        x = self.norm(x)  # [B, N, C]

        # FNO瓶颈层（可选）
        if self.fno_bottleneck is not None:
            x = self.fno_bottleneck(x)

        # 对称的Swin解码器
        # 传递当前分辨率给解码器
        self.decoder.current_input_resolution = (cur_h, cur_w)
        x = self.decoder(x, skip_connections if self.skip_connections else None)

        # 最终输出层
        x = self.final_conv(x)
        
        # 保持与输入一致的空间尺寸（依据当前patch栅格推测尺寸）
        if hasattr(self.patch_embed, 'current_resolution') and self.patch_embed.current_resolution is not None:
            cur_h, cur_w = self.patch_embed.current_resolution
            out_h = cur_h * self.patch_size
            out_w = cur_w * self.patch_size
            if x.shape[-2:] != (out_h, out_w):
                x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
        
        x = self.final_activation(x)

        return x


class UNetDecoder(nn.Module):
    """UNet解码器"""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        skip_connections: bool = True,
        upsampling_mode: str = 'bilinear',
        patches_resolution: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.skip_connections = skip_connections
        self.upsampling_mode = upsampling_mode
        self.patches_resolution = patches_resolution
        
        # 构建解码器层
        self.decoder_blocks = nn.ModuleList()
        
        # 从最深层开始
        in_channels = encoder_channels[-1]
        
        for i, out_channels in enumerate(decoder_channels):
            # 跳跃连接的通道数 - 修正索引逻辑
            if skip_connections and i < len(encoder_channels) - 1:
                # 跳跃连接来自对应的编码器层（倒序）
                skip_idx = len(encoder_channels) - 2 - i
                skip_channels = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            else:
                skip_channels = 0
            
            # 调试信息
            print(f"Decoder layer {i}: in_channels={in_channels}, skip_channels={skip_channels}, out_channels={out_channels}")
            
            block = DecoderBlock(
                in_channels=in_channels,  # 上采样后的通道数
                out_channels=out_channels,
                skip_channels=skip_channels,  # 跳跃连接通道数
                upsampling_mode=upsampling_mode
            )
            self.decoder_blocks.append(block)
            in_channels = out_channels
    
    def forward(self, x: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """解码器前向传播
        
        Args:
            x: 编码器输出 [B, N, C]
            skip_features: 跳跃连接特征列表
            
        Returns:
            解码后的特征图 [B, C, H, W]
        """
        # 将序列格式转换为图像格式
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        print(f"Decoder input: {x.shape}")
        if skip_features is not None:
            print(f"Skip features shapes: {[sf.shape for sf in skip_features]}")
        
        # 逐层解码
        for i, decoder_block in enumerate(self.decoder_blocks):
            print(f"\nDecoder layer {i}:")
            print(f"  Input shape: {x.shape}")
            
            # 先上采样
            x = decoder_block.upsample(x)
            print(f"  After upsample: {x.shape}")
            
            # 添加跳跃连接
            skip_feat = None
            if (skip_features is not None and 
                self.skip_connections and 
                i < len(skip_features)):
                # 跳跃连接索引：从最新的开始（倒序）
                skip_idx = len(skip_features) - 1 - i
                if skip_idx >= 0 and skip_idx < len(skip_features):
                    skip_feat = skip_features[skip_idx]
                    print(f"  Skip feature {skip_idx} shape: {skip_feat.shape}")
                    
                    # 将跳跃连接特征转换为图像格式
                    B_skip, N_skip, C_skip = skip_feat.shape
                    if hasattr(self, 'current_input_resolution') and self.current_input_resolution is not None:
                        H_skip, W_skip = self.current_input_resolution
                    else:
                        H_skip = int(math.sqrt(N_skip))
                        W_skip = int(N_skip // H_skip)
                    skip_feat = skip_feat.transpose(1, 2).view(B_skip, C_skip, H_skip, W_skip)
                    print(f"  Skip feature reshaped: {skip_feat.shape}")
                    
                    # 调整尺寸匹配
                    if skip_feat.shape[-2:] != x.shape[-2:]:
                        skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                        print(f"  Skip feature after resize: {skip_feat.shape}")
                    
                    # 检查通道数是否匹配预期的skip_channels
                    expected_skip_channels = decoder_block.skip_channels
                    print(f"  Expected skip channels: {expected_skip_channels}, Actual: {C_skip}")
                    
                    # 如果通道数不匹配，使用1x1卷积调整
                    if C_skip != expected_skip_channels and expected_skip_channels > 0:
                        if not hasattr(decoder_block, 'skip_proj'):
                            decoder_block.skip_proj = nn.Conv2d(C_skip, expected_skip_channels, 1).to(skip_feat.device)
                            # 初始化权重
                            nn.init.xavier_uniform_(decoder_block.skip_proj.weight)
                            nn.init.zeros_(decoder_block.skip_proj.bias)
                        skip_feat = decoder_block.skip_proj(skip_feat)
                        print(f"  Skip feature after projection: {skip_feat.shape}")
                    
                    # 如果不需要跳跃连接，直接设为None
                    if expected_skip_channels == 0:
                        skip_feat = None
                        print(f"  Skip connection disabled for this layer")
            
            # 应用解码器块的卷积层
            if skip_feat is not None:
                # 拼接跳跃连接
                x = torch.cat([x, skip_feat], dim=1)
                print(f"  After concatenation: {x.shape}")
            
            # 直接调用卷积层，而不是整个decoder_block
            x = decoder_block.conv(x)
            print(f"  After conv: {x.shape}")
        
        # 确保输出尺寸与原始输入匹配
        target_size = self.patches_resolution
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class DecoderBlock(nn.Module):
    """解码器块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        upsampling_mode: str = 'bilinear'
    ):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.skip_channels = skip_channels
        
        # 上采样层
        if upsampling_mode == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        elif upsampling_mode == 'pixel_shuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
                nn.PixelShuffle(2)
            )
        else:  # bilinear
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 卷积层（考虑跳跃连接）
        conv_in_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：这个forward方法不应该被直接调用
        # 上采样和卷积应该在UNetDecoder中分别调用
        x = self.upsample(x)
        x = self.conv(x)
        return x


class FNOBottleneck(nn.Module):
    """FNO瓶颈层（可选）"""
    
    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.channels = channels
        self.modes = modes
        
        # 傅里叶权重
        self.weights1 = nn.Parameter(torch.view_as_complex(torch.randn(channels, channels, modes, modes, 2)))
        self.weights2 = nn.Parameter(torch.view_as_complex(torch.randn(channels, channels, modes, modes, 2)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FNO前向传播
        
        Args:
            x: 输入特征 [B, N, C]
            
        Returns:
            输出特征 [B, N, C]
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # 转换为图像格式
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # 频域卷积
        out_ft = torch.zeros_like(x_ft)
        
        # 计算实际可用的模式数（考虑FFT输出的实际尺寸）
        actual_modes_h = min(self.modes, x_ft.shape[2])
        actual_modes_w = min(self.modes, x_ft.shape[3])
        
        # 频域切片
        x_ft_slice = x_ft[:, :, :actual_modes_h, :actual_modes_w]  # [B, C, actual_modes_h, actual_modes_w]
        
        # 逐批次处理以避免维度问题
        for b in range(B):
            for c_out in range(self.channels):
                for c_in in range(self.channels):
                    out_ft[b, c_out, :actual_modes_h, :actual_modes_w] += (
                        x_ft_slice[b, c_in, :actual_modes_h, :actual_modes_w] * 
                        self.weights1[c_in, c_out, :actual_modes_h, :actual_modes_w]
                    )
        
        # IFFT
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        # 转换回序列格式
        x = x.view(B, C, -1).transpose(1, 2)
        
        return x
