"""
SwinTemporalNAR模型
集成时间编码的Swin Transformer用于非自回归预测
基于技术方案实现时间感知的空间-时间特征学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import math
import logging
from pathlib import Path

# 导入时间编码Transformer
from src.models.temporal_transformer import (
    TemporalTransformerEncoder, TemporalEncodingConfig, create_temporal_encoder
)

logger = logging.getLogger(__name__)

@dataclass
class SwinTemporalConfig:
    """SwinTemporalNAR配置"""
    # 基础配置
    img_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 96
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    
    # 时间编码配置
    temporal_config: TemporalEncodingConfig = field(default_factory=TemporalEncodingConfig)
    
    # NAR预测配置
    future_steps: int = 10
    prediction_type: str = "direct"  # "direct", "residual", "hierarchical"
    
    # 输出配置
    out_channels: int = 3
    output_activation: str = "identity"  # "identity", "tanh", "sigmoid"
    
    # 优化配置
    use_checkpoint: bool = False
    fused_window_process: bool = True

    # 兼容测试的别名参数（不会影响内部实现字段）
    # 注意：这些别名仅用于配置初始化兼容，实际模型使用规范字段
    input_channels: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    heads: Optional[int] = None  # 单值num_heads别名
    time_steps: Optional[int] = None
    prediction_steps: Optional[int] = None
    spatial_resolution: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """别名参数规范化到内部字段。
        满足测试约定：SwinTemporalConfig(input_channels, hidden_dim, num_layers, num_heads,
        time_steps, prediction_steps, spatial_resolution)
        """
        # in_channels 映射
        if self.input_channels is not None:
            self.in_channels = int(self.input_channels)

        # embed_dim 映射
        if self.hidden_dim is not None:
            self.embed_dim = int(self.hidden_dim)

        # depths 与 num_layers 映射
        if self.num_layers is not None:
            layers = int(self.num_layers)
            self.depths = [2] * layers  # 采用均匀深度的保守默认
            # 将 num_heads 映射为列表形式
            if self.heads is not None:
                self.num_heads = [int(self.heads)] * layers
            elif isinstance(self.num_heads, int):
                self.num_heads = [int(self.num_heads)] * layers
            elif isinstance(self.num_heads, list) and len(self.num_heads) != layers:
                # 对齐长度
                self.num_heads = [self.num_heads[0]] * layers
        else:
            # 如果未提供 num_layers，但提供了 heads（int），也做广播
            if self.heads is not None:
                # 若已有 depths 列表，按其长度广播
                L = len(self.depths) if isinstance(self.depths, list) else 4
                self.num_heads = [int(self.heads)] * L

        # spatial_resolution 映射到 img_size（使用高宽中的高作为基准）
        if self.spatial_resolution is not None:
            try:
                h, w = int(self.spatial_resolution[0]), int(self.spatial_resolution[1])
                # 使用较小边作为 img_size 基准，以保持窗口与patch对齐
                self.img_size = min(h, w)
            except Exception:
                pass

        # prediction_steps 映射到 future_steps
        if self.prediction_steps is not None:
            self.future_steps = int(self.prediction_steps)

        # 若未显式设置 out_channels，默认与输入通道一致以满足测试期望
        if not isinstance(self.out_channels, int) or self.out_channels <= 0:
            self.out_channels = self.in_channels

class PatchEmbedding(nn.Module):
    """
    图像块嵌入
    将输入图像分割成patch并进行嵌入
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, 
                 in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # 投影层
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"PatchEmbedding初始化: img_size={img_size}, patch_size={patch_size}, "
                   f"patches={self.num_patches}, embed_dim={embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            patch嵌入 [B, N, D] 其中N是patch数量，D是嵌入维度
        """
        B, C, H, W = x.shape
        
        # 投影到嵌入空间
        x = self.proj(x)  # [B, D, H//patch_size, W//patch_size]
        
        # 展平空间维度
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 归一化
        x = self.norm(x)
        
        return x

class WindowAttention(nn.Module):
    """
    窗口注意力机制
    基于窗口的注意力计算，支持时间编码
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int, 
                 qkv_bias: bool = True, attention_dropout: float = 0.0, 
                 projection_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(projection_dropout)
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 相对位置索引
        self.register_buffer("relative_position_index", self._get_relative_position_index())
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        logger.info(f"WindowAttention初始化: dim={dim}, window_size={window_size}, "
                   f"num_heads={num_heads}")
    
    def _get_relative_position_index(self):
        """获取相对位置索引"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B*num_windows, window_size*window_size, C]
            mask: 可选的注意力掩码
            
        Returns:
            注意力输出 [B*num_windows, window_size*window_size, C]
        """
        B_, N, C = x.shape
        
        # QKV变换
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attention = (q @ k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)
        
        # 应用掩码
        if mask is not None:
            nW = mask.shape[0]
            attention = attention.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, N, N)
        
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        
        # 输出计算
        x = (attention @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer块
    包含窗口注意力和前馈网络
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attention_dropout: float = 0.0,
                 drop_path: float = 0.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attention = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attention_dropout=attention_dropout, projection_dropout=dropout
        )
        
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        logger.info(f"SwinTransformerBlock初始化: dim={dim}, num_heads={num_heads}, "
                   f"window_size={window_size}, shift_size={shift_size}")
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, H*W, C]
            H: 高度
            W: 宽度
            
        Returns:
            输出特征 [B, H*W, C]
        """
        B, L, C = x.shape
        
        # 残差连接
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位（如果需要）
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口分割
        x_windows = self._window_partition(shifted_x, self.window_size)
        
        # 窗口注意力
        attention_windows = self.attention(x_windows)
        
        # 窗口合并
        shifted_x = self._window_reverse(attention_windows, self.window_size, H, W)
        
        # 循环移位恢复
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # 残差连接
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """窗口分割"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows.view(-1, window_size * window_size, C)
    
    def _window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """窗口合并"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class PatchMerging(nn.Module):
    """
    图像块合并
    将相邻的patch合并以减少空间分辨率
    """
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
        logger.info(f"PatchMerging初始化: dim={dim}")
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, H*W, C]
            H: 高度
            W: 宽度
            
        Returns:
            合并后的特征 [B, H//2 * W//2, 2*C], 新高度, 新宽度
        """
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        # 空间下采样
        x0 = x[:, 0::2, 0::2, :]  # [B, H//2, W//2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H//2, W//2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H//2, W//2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H//2, W//2, C]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H//2, W//2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H//2*W//2, 4*C]
        
        x = self.norm(x)
        x = self.reduction(x)  # [B, H//2*W//2, 2*C]
        
        return x, H // 2, W // 2

class TemporalSwinBlock(nn.Module):
    """
    时间感知的Swin Transformer块
    集成时间编码的Swin Transformer块
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attention_dropout: float = 0.0,
                 drop_path: float = 0.0, temporal_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.temporal_dim = temporal_dim
        
        # 标准Swin块
        self.swin_block = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift_size, mlp_ratio=mlp_ratio,
            dropout=dropout, attention_dropout=attention_dropout,
            drop_path=drop_path
        )
        
        # 时间编码融合
        self.temporal_proj = nn.Linear(temporal_dim, dim)
        self.temporal_gate = nn.Parameter(torch.zeros(1))
        
        logger.info(f"TemporalSwinBlock初始化: dim={dim}, num_heads={num_heads}, "
                   f"window_size={window_size}, temporal_dim={temporal_dim}")
    
    def forward(self, x: torch.Tensor, H: int, W: int, 
                temporal_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 空间特征 [B, H*W, C]
            H: 高度
            W: 宽度
            temporal_features: 时间特征 [B, C_temporal]
            
        Returns:
            融合特征 [B, H*W, C]
        """
        # 标准Swin块处理
        x = self.swin_block(x, H, W)
        
        # 融合时间特征
        if temporal_features is not None:
            # 投影时间特征
            temporal_proj = self.temporal_proj(temporal_features)  # [B, C]
            
            # 扩展空间维度
            B, L, C = x.shape
            temporal_proj = temporal_proj.unsqueeze(1).expand(-1, L, -1)  # [B, L, C]
            
            # 门控融合
            x = x + self.temporal_gate * temporal_proj
        
        return x

class SwinTemporalStage(nn.Module):
    """
    SwinTemporal阶段
    包含多个时间感知的Swin块和patch合并
    """
    
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 attention_dropout: float = 0.0, drop_path: float = 0.0,
                 downsample: bool = True, temporal_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # 构建时间感知的Swin块
        self.blocks = nn.ModuleList([
            TemporalSwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path * i / (depth - 1) if depth > 1 else 0,
                temporal_dim=temporal_dim
            )
            for i in range(depth)
        ])
        
        # Patch合并
        self.downsample = PatchMerging(dim) if downsample else nn.Identity()
        
        logger.info(f"SwinTemporalStage初始化: dim={dim}, depth={depth}, num_heads={num_heads}")
    
    def forward(self, x: torch.Tensor, H: int, W: int,
                temporal_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int, int]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, H*W, C]
            H: 高度
            W: 宽度
            temporal_features: 时间特征
            
        Returns:
            输出特征 [B, H'*W', C'], 新高度, 新宽度
        """
        # 通过所有块
        for block in self.blocks:
            x = block(x, H, W, temporal_features)
        
        # 下采样
        x, H, W = self.downsample(x, H, W)
        
        return x, H, W

class SwinTemporalEncoder(nn.Module):
    """
    SwinTemporal编码器
    基于Swin Transformer的时间-空间特征编码器
    """
    
    def __init__(self, config: SwinTemporalConfig):
        super().__init__()
        self.config = config
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size, patch_size=config.patch_size,
            in_channels=config.in_channels, embed_dim=config.embed_dim
        )
        
        # 绝对位置嵌入
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)
        
        # 时间编码器
        self.temporal_encoder = create_temporal_encoder(config.temporal_config, num_layers=2)
        
        # 构建阶段
        self.stages = nn.ModuleList()
        dim = config.embed_dim
        H = W = config.img_size // config.patch_size
        
        for i, (depth, num_heads) in enumerate(zip(config.depths, config.num_heads)):
            stage = SwinTemporalStage(
                dim=dim, depth=depth, num_heads=num_heads,
                window_size=config.window_size, mlp_ratio=config.mlp_ratio,
                dropout=config.dropout, attention_dropout=config.attention_dropout,
                drop_path=config.drop_path_rate * i / len(config.depths),
                downsample=(i < len(config.depths) - 1),
                temporal_dim=config.temporal_config.embedding_dim
            )
            self.stages.append(stage)
            
            # 更新维度
            if i < len(config.depths) - 1:
                dim *= 2
                H //= 2
                W //= 2
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"SwinTemporalEncoder初始化完成，阶段数: {len(self.stages)}")
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, T, C, H, W] 或 [B, C, H, W]
            time_steps: 时间步长 [B, T] 或 [B]
            
        Returns:
            多尺度特征列表
        """
        # 处理输入维度
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            # 合并时间维度到批次维度
            x = x.reshape(B * T, C, H, W)
            if time_steps is not None:
                time_steps = time_steps.reshape(B * T)
        else:  # [B, C, H, W]
            B, C, H, W = x.shape
            T = 1
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B*T, N, D]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # 时间特征编码
        temporal_features = None
        if time_steps is not None:
            # 创建时间特征
            time_features = torch.zeros(B * T, self.config.temporal_config.embedding_dim, device=x.device)
            time_features[:, 0] = time_steps.float()
            temporal_features = self.temporal_encoder(time_features.unsqueeze(1)).squeeze(1)
        
        # 存储多尺度特征
        features = []
        H_patches = self.config.img_size // self.config.patch_size
        W_patches = self.config.img_size // self.config.patch_size
        
        # 通过各个阶段
        for i, stage in enumerate(self.stages):
            x, H_patches, W_patches = stage(x, H_patches, W_patches, temporal_features)
            features.append(x)
        
        return features

class TemporalNARHead(nn.Module):
    """
    非自回归预测头
    集成时间信息的预测头
    """
    
    def __init__(self, config: SwinTemporalConfig):
        super().__init__()
        self.config = config

        # 计算输入特征维度
        self.input_dim = config.embed_dim * (2 ** (len(config.depths) - 1))

        # 每个时间步的输出大小（按照完整图像重建）
        self.output_size_per_step = int(config.out_channels) * int(config.img_size) * int(config.img_size)

        # 时间融合层
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.input_dim + config.temporal_config.embedding_dim, self.input_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 预测头
        if config.prediction_type == "direct":
            self.predictor = self._build_direct_predictor()
        elif config.prediction_type == "residual":
            self.predictor = self._build_residual_predictor()
        elif config.prediction_type == "hierarchical":
            self.predictor = self._build_hierarchical_predictor()
        else:
            raise ValueError(f"不支持的预测类型: {config.prediction_type}")
        
        # 输出激活函数
        if config.output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif config.output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        logger.info(f"TemporalNARHead初始化: 输入维度={self.input_dim}, "
                   f"预测类型={config.prediction_type}")
    
    def _build_direct_predictor(self) -> nn.Module:
        """构建直接预测器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.input_dim // 2, self.output_size_per_step * self.config.future_steps)
        )
    
    def _build_residual_predictor(self) -> nn.Module:
        """构建残差预测器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.input_dim // 2, self.output_size_per_step * self.config.future_steps)
        )
    
    def _build_hierarchical_predictor(self) -> nn.Module:
        """构建分层预测器"""
        layers = []
        current_dim = self.input_dim

        # 逐步减少维度
        target_dim = self.output_size_per_step * self.config.future_steps
        while current_dim > target_dim * 4:
            next_dim = current_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout)
            ])
            current_dim = next_dim

        # 最终输出层
        layers.append(nn.Linear(current_dim, target_dim))

        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, 
                temporal_features: Optional[torch.Tensor] = None,
                target_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 空间特征 [B, N, C]
            temporal_features: 时间特征 [B, C_temporal]
            target_time: 目标时间 [B]
            
        Returns:
            预测结果 [B, future_steps, out_channels]
        """
        B, N, C = features.shape
        
        # 全局平均池化
        global_features = features.mean(dim=1)  # [B, C]
        
        # 时间融合（简化：若提供时间特征则融合，否则使用空间全局特征）
        fused_features = global_features
        if temporal_features is not None:
            fused_features = torch.cat([global_features, temporal_features], dim=1)
            fused_features = self.temporal_fusion(fused_features)
        
        # 预测
        predictions = self.predictor(fused_features)  # [B, future_steps * (C * H * W)]

        # 重塑成 [B, T_out, C, H, W]
        C = int(self.config.out_channels)
        H = int(self.config.img_size)
        W = int(self.config.img_size)
        predictions = predictions.view(B, self.config.future_steps, C, H, W)
        
        # 应用输出激活
        predictions = self.output_activation(predictions)
        
        return predictions

class SwinTemporalNAR(nn.Module):
    """
    SwinTemporalNAR模型
    集成时间编码的Swin Transformer用于非自回归预测
    """
    
    def __init__(self, config: SwinTemporalConfig):
        super().__init__()
        self.config = config
        
        # 编码器
        self.encoder = SwinTemporalEncoder(config)
        
        # 预测头
        self.predictor = TemporalNARHead(config)
        
        logger.info(f"SwinTemporalNAR模型初始化完成: img_size={config.img_size}, "
                   f"embed_dim={config.embed_dim}, future_steps={config.future_steps}")
    
    def forward(self, x: torch.Tensor, 
                time_steps: Optional[torch.Tensor] = None,
                target_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, T, C, H, W] 或 [B, C, H, W]
            time_steps: 输入时间步长 [B, T] 或 [B]
            target_times: 目标时间步长 [B, future_steps]
            
        Returns:
            预测结果 [B, future_steps, out_channels, H, W]
        """
        # 编码多尺度特征
        features = self.encoder(x, time_steps)
        
        # 使用最深层的特征进行预测
        deepest_features = features[-1]
        
        # 预测未来时间步
        predictions = []
        for i in range(self.config.future_steps):
            if target_times is not None:
                target_time = target_times[:, i]
            else:
                # 如果没有提供目标时间，使用相对时间
                if time_steps is not None:
                    target_time = time_steps[:, -1] + i + 1
                else:
                    target_time = None
            
            # 单次预测
            pred = self.predictor(deepest_features, target_time=target_time)
            predictions.append(pred)
        
        # 合并所有时间步的预测
        predictions = torch.stack(predictions, dim=1)  # [B, future_steps, out_channels, H, W]
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2),  # 假设FP32
            "config": {
                "img_size": self.config.img_size,
                "patch_size": self.config.patch_size,
                "embed_dim": self.config.embed_dim,
                "depths": self.config.depths,
                "future_steps": self.config.future_steps,
                "prediction_type": self.config.prediction_type,
            }
        }
    
    def save_model(self, filepath: Union[str, Path]):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'model_info': self.get_model_info()
            }, filepath)
            logger.info(f"SwinTemporalNAR模型已保存到: {filepath}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def load_model(self, filepath: Union[str, Path], map_location: Optional[str] = None):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=map_location)
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"SwinTemporalNAR模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")

# 模型工厂函数
def create_swin_temporal_nar(config: Optional[SwinTemporalConfig] = None, **kwargs) -> SwinTemporalNAR:
    """
    创建SwinTemporalNAR模型
    
    Args:
        config: 模型配置，如果为None则使用默认配置
        **kwargs: 配置参数
        
    Returns:
        SwinTemporalNAR模型实例
    """
    if config is None:
        config = SwinTemporalConfig(**kwargs)
    
    return SwinTemporalNAR(config)