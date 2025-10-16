"""MLP模型架构

基于MLP的模型，支持坐标编码和patch处理。
严格遵循统一接口：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseModel


class MLPModel(BaseModel):
    """基于MLP的模型
    
    支持两种模式：
    1. 坐标模式：将每个像素位置作为坐标输入MLP
    2. Patch模式：将图像分割为patch，每个patch通过MLP处理
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 256,
        # MLP配置
        hidden_dims: List[int] = [256, 512, 512, 256],
        activation: str = 'relu',  # 'relu', 'gelu', 'swish'
        dropout: float = 0.1,
        # 模式配置
        mode: str = 'patch',  # 'coord', 'patch'
        # 坐标模式参数
        coord_encoding: str = 'positional',  # 'positional', 'fourier', 'none'
        coord_encoding_dim: int = 64,
        # Patch模式参数
        patch_size: int = 8,
        overlap: float = 0.0,  # patch重叠比例
        # 位置编码参数
        use_positional_encoding: bool = True,
        max_freq: float = 10.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.mode = mode
        self.patch_size = patch_size
        self.overlap = overlap
        self.coord_encoding = coord_encoding
        self.use_positional_encoding = use_positional_encoding
        
        # 计算输入维度
        if mode == 'coord':
            # 坐标模式：像素值 + 坐标编码
            coord_dim = self._get_coord_encoding_dim(coord_encoding, coord_encoding_dim)
            mlp_input_dim = in_channels + coord_dim
            mlp_output_dim = out_channels
        elif mode == 'patch':
            # Patch模式：patch像素 + 位置编码
            patch_pixels = patch_size * patch_size * in_channels
            pos_dim = coord_encoding_dim if use_positional_encoding else 0
            mlp_input_dim = patch_pixels + pos_dim
            mlp_output_dim = patch_size * patch_size * out_channels
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 构建MLP
        self.mlp = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dims=hidden_dims,
            output_dim=mlp_output_dim,
            activation=activation,
            dropout=dropout
        )
        
        # 坐标编码器
        if mode == 'coord' and coord_encoding != 'none':
            self.coord_encoder = CoordinateEncoder(
                encoding_type=coord_encoding,
                encoding_dim=coord_encoding_dim,
                max_freq=max_freq
            )
        else:
            self.coord_encoder = None
        
        # Patch位置编码器
        if mode == 'patch' and use_positional_encoding:
            self.patch_pos_encoder = PatchPositionalEncoder(
                patch_size=patch_size,
                img_size=img_size,
                encoding_dim=coord_encoding_dim
            )
        else:
            self.patch_pos_encoder = None
        
        # Patch处理器
        if mode == 'patch':
            self.patch_processor = PatchProcessor(
                patch_size=patch_size,
                overlap=overlap,
                img_size=img_size
            )
    
    def _get_coord_encoding_dim(self, encoding_type: str, encoding_dim: int) -> int:
        """获取坐标编码维度"""
        if encoding_type == 'none':
            return 2  # 只有x, y坐标
        elif encoding_type == 'positional':
            # 位置编码：x和y坐标各有sin/cos编码，总共4倍频率数
            # 但实际输出维度由encoding_dim控制
            return encoding_dim
        elif encoding_type == 'fourier':
            # 傅里叶编码：x和y坐标各有sin/cos编码
            return encoding_dim
        else:
            return 2
    
    def _build_mlp(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int,
        activation: str,
        dropout: float
    ) -> nn.Module:
        """构建MLP网络"""
        layers = []
        
        # 激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'swish':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 输入层
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（coords等）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        if self.mode == 'coord':
            return self._forward_coord_mode(x, **kwargs)
        elif self.mode == 'patch':
            return self._forward_patch_mode(x, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _forward_coord_mode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """坐标模式前向传播"""
        B, C, H, W = x.shape
        
        # 生成坐标网格
        coords = kwargs.get('coords')
        if coords is None:
            y_coords = torch.linspace(-1, 1, H, device=x.device)
            x_coords = torch.linspace(-1, 1, W, device=x.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
            coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 坐标编码
        if self.coord_encoder is not None:
            coord_encoding = self.coord_encoder(coords)  # [B, coord_dim, H, W]
        else:
            coord_encoding = coords  # [B, 2, H, W]
        
        # 检查维度匹配
        if coord_encoding.shape[-2:] != x.shape[-2:]:
            coord_encoding = F.interpolate(coord_encoding, size=(H, W), mode='bilinear', align_corners=False)
        
        # 拼接像素值和坐标编码
        mlp_input = torch.cat([x, coord_encoding], dim=1)  # [B, C+coord_dim, H, W]
        
        # 重塑为序列格式
        mlp_input = mlp_input.permute(0, 2, 3, 1).reshape(B * H * W, -1)  # [B*H*W, C+coord_dim]
        
        # MLP前向传播
        mlp_output = self.mlp(mlp_input)  # [B*H*W, output_dim]
        
        # 检查输出维度是否匹配
        if mlp_output.shape[-1] != self.out_channels:
            # 添加一个线性层来调整维度
            if not hasattr(self, 'output_proj'):
                self.output_proj = nn.Linear(mlp_output.shape[-1], self.out_channels).to(mlp_output.device)
            mlp_output = self.output_proj(mlp_output)
        
        # 重塑回图像格式
        output = mlp_output.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)  # [B, C_out, H, W]
        
        return output
    
    def _forward_patch_mode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Patch模式前向传播"""
        B, C, H, W = x.shape
        
        # 提取patches
        patches, patch_coords = self.patch_processor.extract_patches(x)
        # patches: [B, num_patches, patch_size*patch_size*C]
        # patch_coords: [B, num_patches, 2]
        
        B, num_patches, patch_dim = patches.shape
        
        # Patch位置编码
        if self.patch_pos_encoder is not None:
            pos_encoding = self.patch_pos_encoder(patch_coords)  # [B, num_patches, pos_dim]
            mlp_input = torch.cat([patches, pos_encoding], dim=-1)
        else:
            mlp_input = patches
        
        # 重塑为批次格式
        mlp_input = mlp_input.reshape(B * num_patches, -1)  # [B*num_patches, input_dim]
        
        # MLP前向传播
        mlp_output = self.mlp(mlp_input)  # [B*num_patches, output_dim]
        
        # 计算正确的patch输出维度
        patch_output_dim = self.patch_size * self.patch_size * self.out_channels
        
        # 如果MLP输出维度与期望的patch输出维度不匹配，需要调整
        if mlp_output.shape[-1] != patch_output_dim:
            # 添加一个线性层来调整维度
            if not hasattr(self, 'patch_output_proj'):
                self.patch_output_proj = nn.Linear(mlp_output.shape[-1], patch_output_dim).to(mlp_output.device)
            mlp_output = self.patch_output_proj(mlp_output)
        
        # 重塑为patch格式
        mlp_output = mlp_output.reshape(B, num_patches, patch_output_dim)
        
        # 重建图像
        output = self.patch_processor.reconstruct_from_patches(
            mlp_output, patch_coords, (H, W), self.out_channels
        )
        
        return output


class CoordinateEncoder(nn.Module):
    """坐标编码器"""
    
    def __init__(
        self, 
        encoding_type: str = 'positional',
        encoding_dim: int = 64,
        max_freq: float = 10.0
    ):
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding_dim = encoding_dim
        self.max_freq = max_freq
        
        if encoding_type == 'fourier':
            # 傅里叶特征的频率
            self.register_buffer('frequencies', 
                torch.exp(torch.linspace(0, math.log(max_freq), encoding_dim // 4)))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """编码坐标
        
        Args:
            coords: 坐标张量 [B, 2, H, W]
            
        Returns:
            编码后的坐标 [B, encoding_dim, H, W]
        """
        B, _, H, W = coords.shape
        
        if self.encoding_type == 'positional':
            return self._positional_encoding(coords)
        elif self.encoding_type == 'fourier':
            return self._fourier_encoding(coords)
        else:
            return coords  # 直接返回原始坐标
    
    def _positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """位置编码"""
        B, _, H, W = coords.shape
        
        # 生成频率
        freqs = torch.exp(torch.linspace(0, math.log(self.max_freq), 
                                       self.encoding_dim // 4, device=coords.device))
        
        # 对x和y坐标分别编码
        x_coords = coords[:, 0:1]  # [B, 1, H, W]
        y_coords = coords[:, 1:2]  # [B, 1, H, W]
        
        # 计算sin和cos编码
        x_freqs = x_coords.unsqueeze(-1) * freqs.view(1, 1, 1, 1, -1)  # [B, 1, H, W, freq_dim]
        y_freqs = y_coords.unsqueeze(-1) * freqs.view(1, 1, 1, 1, -1)  # [B, 1, H, W, freq_dim]
        
        x_sin = torch.sin(x_freqs).flatten(-2)  # [B, 1, H, W*freq_dim]
        x_cos = torch.cos(x_freqs).flatten(-2)
        y_sin = torch.sin(y_freqs).flatten(-2)
        y_cos = torch.cos(y_freqs).flatten(-2)
        
        # 拼接所有编码
        encoding = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=1)  # [B, 4*freq_dim, H, W]
        
        # 确保输出维度正确
        if encoding.shape[1] != self.encoding_dim:
            # 使用1x1卷积调整到目标维度
            if not hasattr(self, 'dim_proj'):
                self.dim_proj = nn.Conv2d(encoding.shape[1], self.encoding_dim, 1).to(encoding.device)
            encoding = self.dim_proj(encoding)
        
        return encoding
    
    def _fourier_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """傅里叶特征编码"""
        B, _, H, W = coords.shape
        
        # 展平坐标
        coords_flat = coords.permute(0, 2, 3, 1).reshape(B * H * W, 2)  # [B*H*W, 2]
        
        # 计算傅里叶特征
        freqs = self.frequencies.unsqueeze(0)  # [1, freq_dim]
        
        # 对每个坐标维度计算
        x_proj = coords_flat[:, 0:1] * freqs  # [B*H*W, freq_dim]
        y_proj = coords_flat[:, 1:2] * freqs  # [B*H*W, freq_dim]
        
        # sin和cos编码
        encoding = torch.cat([
            torch.sin(x_proj), torch.cos(x_proj),
            torch.sin(y_proj), torch.cos(y_proj)
        ], dim=-1)  # [B*H*W, encoding_dim]
        
        # 重塑回图像格式
        encoding = encoding.reshape(B, H, W, self.encoding_dim).permute(0, 3, 1, 2)
        
        return encoding


class PatchPositionalEncoder(nn.Module):
    """Patch位置编码器"""
    
    def __init__(self, patch_size: int, img_size: int, encoding_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.encoding_dim = encoding_dim
        
        # 位置编码MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim)
        )
    
    def forward(self, patch_coords: torch.Tensor) -> torch.Tensor:
        """编码patch位置
        
        Args:
            patch_coords: patch中心坐标 [B, num_patches, 2]
            
        Returns:
            位置编码 [B, num_patches, encoding_dim]
        """
        # 归一化坐标到[-1, 1]
        normalized_coords = patch_coords / (self.img_size / 2) - 1
        
        # MLP编码
        pos_encoding = self.pos_mlp(normalized_coords)
        
        return pos_encoding


class PatchProcessor(nn.Module):
    """Patch处理器"""
    
    def __init__(self, patch_size: int, overlap: float, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.img_size = img_size
        
        # 计算步长
        self.stride = int(patch_size * (1 - overlap))
        
        # 计算patch网格
        self.num_patches_h = (img_size - patch_size) // self.stride + 1
        self.num_patches_w = (img_size - patch_size) // self.stride + 1
        self.total_patches = self.num_patches_h * self.num_patches_w
    
    def extract_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取patches
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            patches: [B, num_patches, patch_size*patch_size*C]
            patch_coords: [B, num_patches, 2] (patch中心坐标)
        """
        B, C, H, W = x.shape
        
        patches = []
        coords = []
        
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # 计算patch位置
                y_start = i * self.stride
                x_start = j * self.stride
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                
                # 提取patch
                patch = x[:, :, y_start:y_end, x_start:x_end]  # [B, C, patch_size, patch_size]
                patch = patch.flatten(1)  # [B, C*patch_size*patch_size]
                patches.append(patch)
                
                # 计算patch中心坐标
                center_y = (y_start + y_end) / 2
                center_x = (x_start + x_end) / 2
                coords.append([center_x, center_y])
        
        patches = torch.stack(patches, dim=1)  # [B, num_patches, patch_dim]
        coords = torch.tensor(coords, device=x.device).unsqueeze(0).expand(B, -1, -1)  # [B, num_patches, 2]
        
        return patches, coords
    
    def reconstruct_from_patches(
        self, 
        patches: torch.Tensor, 
        patch_coords: torch.Tensor,
        output_size: Tuple[int, int],
        out_channels: int
    ) -> torch.Tensor:
        """从patches重建图像
        
        Args:
            patches: [B, num_patches, patch_size*patch_size*out_channels]
            patch_coords: [B, num_patches, 2]
            output_size: (H, W)
            out_channels: 输出通道数
            
        Returns:
            重建的图像 [B, out_channels, H, W]
        """
        B, num_patches, _ = patches.shape
        H, W = output_size
        
        # 初始化输出图像和权重图
        output = torch.zeros(B, out_channels, H, W, device=patches.device)
        weight_map = torch.zeros(B, 1, H, W, device=patches.device)
        
        # 重塑patches
        patches = patches.reshape(B, num_patches, out_channels, self.patch_size, self.patch_size)
        
        patch_idx = 0
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # 计算patch位置
                y_start = i * self.stride
                x_start = j * self.stride
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                
                # 确保不超出边界
                y_end = min(y_end, H)
                x_end = min(x_end, W)
                patch_h = y_end - y_start
                patch_w = x_end - x_start
                
                # 添加patch到输出
                patch = patches[:, patch_idx, :, :patch_h, :patch_w]
                output[:, :, y_start:y_end, x_start:x_end] += patch
                weight_map[:, :, y_start:y_end, x_start:x_end] += 1
                
                patch_idx += 1
        
        # 归一化重叠区域
        output = output / (weight_map + 1e-8)
        
        return output