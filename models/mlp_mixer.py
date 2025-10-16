"""MLP-Mixer模型

基于MLP的视觉架构，使用两种类型的MLP层：
1. Token-mixing MLP：在空间维度上混合信息
2. Channel-mixing MLP：在通道维度上混合信息

Reference:
    MLP-Mixer: An all-MLP Architecture for Vision
    https://arxiv.org/abs/2105.01601
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class MLP(nn.Module):
    """多层感知机
    
    标准的MLP块，包含两个线性层和激活函数
    """
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """Mixer块
    
    包含Token-mixing MLP和Channel-mixing MLP的基本构建块
    """
    
    def __init__(
        self, 
        dim: int, 
        seq_len: int, 
        mlp_ratio: Tuple[int, int] = (0.5, 4.0), 
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.0, 
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # 修复mlp_ratio类型问题
        if isinstance(mlp_ratio, (list, tuple)):
            tokens_dim, channels_dim = [int(x * dim) for x in mlp_ratio]
        else:
            # 如果是单个float值，使用默认比例
            tokens_dim = int(0.5 * dim)
            channels_dim = int(mlp_ratio * dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.mlp_tokens = MLP(seq_len, tokens_dim, seq_len, act_layer, drop)
        
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_channels = MLP(dim, channels_dim, dim, act_layer, drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token-mixing
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        
        # Channel-mixing
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        
        return x


class PatchEmbed(nn.Module):
    """图像到补丁嵌入
    
    将输入图像分割成补丁并嵌入到高维空间
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class PatchRestore(nn.Module):
    """补丁恢复
    
    将补丁序列恢复为图像格式
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        embed_dim: int = 768, 
        out_chans: int = 3
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.proj(x)  # [B, N, out_chans * patch_size^2]
        
        # 重塑为图像格式
        x = x.reshape(B, self.grid_size, self.grid_size, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x


class MLPMixer(BaseModel):
    """MLP-Mixer模型
    
    基于MLP的视觉架构，完全使用MLP进行特征提取和混合。
    适用于图像分类、分割等任务。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        patch_size: 补丁大小，默认16
        embed_dim: 嵌入维度，默认512
        depth: Mixer块的数量，默认8
        mlp_ratio: MLP扩展比例 (token_mixing_ratio, channel_mixing_ratio)，默认(0.5, 4.0)
        drop_rate: Dropout概率，默认0.0
        drop_path_rate: DropPath概率，默认0.0
        **kwargs: 其他参数
    
    Examples:
        >>> model = MLPMixer(in_channels=3, out_channels=1, img_size=256, patch_size=16)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        mlp_ratio: Tuple[float, float] = (0.5, 4.0),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        
        # 确保图像尺寸能被补丁大小整除
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # 补丁嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )
        
        # Mixer块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减
        self.blocks = nn.ModuleList([
            MixerBlock(
                dim=embed_dim,
                seq_len=self.num_patches,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 补丁恢复
        self.patch_restore = PatchRestore(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=out_channels
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            特征张量 [B, num_patches, embed_dim]
        """
        # 补丁嵌入
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Mixer块
        for blk in self.blocks:
            x = blk(x)
        
        # 层归一化
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（忽略，保持接口一致性）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 特征提取
        x = self.forward_features(x)
        
        # 补丁恢复
        x = self.patch_restore(x)
        
        return x
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs
        
        Args:
            input_shape: 输入形状，默认为(1, in_channels, img_size, img_size)
            
        Returns:
            FLOPs数量
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        batch_size, _, height, width = input_shape
        
        # 补丁嵌入FLOPs
        patch_embed_flops = (
            self.in_channels * self.embed_dim * 
            self.patch_size * self.patch_size * 
            self.num_patches
        )
        
        # Mixer块FLOPs
        mixer_flops = 0
        for _ in range(self.depth):
            # Token-mixing MLP
            if isinstance(self.mlp_ratio, (list, tuple)):
                tokens_dim = int(self.mlp_ratio[0] * self.embed_dim)
            else:
                tokens_dim = int(0.5 * self.embed_dim)
            token_mixing_flops = (
                # 第一个线性层
                self.num_patches * tokens_dim * self.num_patches +
                # 第二个线性层
                tokens_dim * self.num_patches * self.num_patches
            ) * self.embed_dim
            
            # Channel-mixing MLP
            if isinstance(self.mlp_ratio, (list, tuple)):
                channels_dim = int(self.mlp_ratio[1] * self.embed_dim)
            else:
                channels_dim = int(self.mlp_ratio * self.embed_dim)
            channel_mixing_flops = (
                # 第一个线性层
                self.embed_dim * channels_dim +
                # 第二个线性层
                channels_dim * self.embed_dim
            ) * self.num_patches
            
            mixer_flops += token_mixing_flops + channel_mixing_flops
        
        # 补丁恢复FLOPs
        patch_restore_flops = (
            self.embed_dim * self.out_channels * 
            self.patch_size * self.patch_size * 
            self.num_patches
        )
        
        total_flops = patch_embed_flops + mixer_flops + patch_restore_flops
        self._flops = total_flops * batch_size
        return self._flops
    
    def freeze_patch_embed(self) -> None:
        """冻结补丁嵌入层参数"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
    
    def freeze_mixer_blocks(self, start_layer: int = 0, end_layer: Optional[int] = None) -> None:
        """冻结指定范围的Mixer块参数
        
        Args:
            start_layer: 开始层索引
            end_layer: 结束层索引，None表示到最后一层
        """
        if end_layer is None:
            end_layer = len(self.blocks)
        
        for i in range(start_layer, min(end_layer, len(self.blocks))):
            for param in self.blocks[i].parameters():
                param.requires_grad = False
    
    def get_intermediate_features(self, x: torch.Tensor, layer_indices: list = None) -> dict:
        """获取中间层特征（用于分析和可视化）
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            layer_indices: 要提取特征的层索引列表，None表示提取所有层
            
        Returns:
            包含中间特征的字典
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))
        
        features = {}
        
        # 补丁嵌入
        x = self.patch_embed(x)
        features['patch_embed'] = x.clone()
        
        # Mixer块
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features[f'block_{i}'] = x.clone()
        
        # 层归一化
        x = self.norm(x)
        features['norm'] = x.clone()
        
        return features
    
    def interpolate_pos_encoding(self, new_size: int):
        """插值位置编码（如果需要处理不同尺寸的输入）
        
        Args:
            new_size: 新的图像尺寸
        """
        # MLP-Mixer不使用位置编码，但保留接口以保持一致性
        # 实际上需要重新初始化patch_embed和patch_restore
        if new_size != self.img_size:
            self.img_size = new_size
            self.grid_size = new_size // self.patch_size
            self.num_patches = self.grid_size * self.grid_size
            
            # 重新初始化相关层
            self.patch_embed = PatchEmbed(
                img_size=new_size,
                patch_size=self.patch_size,
                in_chans=self.in_channels,
                embed_dim=self.embed_dim
            )
            
            self.patch_restore = PatchRestore(
                img_size=new_size,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                out_chans=self.out_channels
            )
            
            # 重新初始化Mixer块（因为序列长度改变了）
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
            self.blocks = nn.ModuleList([
                MixerBlock(
                    dim=self.embed_dim,
                    seq_len=self.num_patches,
                    mlp_ratio=self.mlp_ratio,
                    drop=self.drop_rate,
                    drop_path=dpr[i]
                )
                for i in range(self.depth)
            ])


# 别名，保持向后兼容
MixerModel = MLPMixer