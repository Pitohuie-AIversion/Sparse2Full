"""U-Net++模型实现

实现嵌套U-Net架构（U-Net++），通过密集跳跃连接和深度监督提升性能。
严格遵循统一接口：forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]

Reference:
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    https://arxiv.org/abs/1807.10165
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class ConvBlock(nn.Module):
    """卷积块：Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(BaseModel):
    """U-Net++模型
    
    实现嵌套U-Net架构，通过密集跳跃连接提升特征重用。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 输入图像尺寸
        features: 各层特征通道数，默认[32, 64, 128, 256]
        deep_supervision: 是否启用深度监督
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        features: List[int] = [32, 64, 128, 256],
        deep_supervision: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.features = features
        self.deep_supervision = deep_supervision
        self.num_layers = len(features)
        
        # 编码器路径
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, features[0], dropout))
        for i in range(1, self.num_layers):
            self.encoders.append(ConvBlock(features[i-1], features[i], dropout))
        
        # 下采样
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.num_layers - 1)])
        
        # 上采样
        self.upsamples = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.upsamples.append(
                nn.ConvTranspose2d(features[i+1], features[i], kernel_size=2, stride=2)
            )
        
        # 嵌套卷积块
        self.nested_convs = nn.ModuleDict()
        
        # 第1层嵌套
        for i in range(self.num_layers - 1):
            in_ch = features[i] * 2  # 上采样 + 编码器特征
            self.nested_convs[f'{i}_1'] = ConvBlock(in_ch, features[i], dropout)
        
        # 第2层嵌套
        for i in range(self.num_layers - 2):
            in_ch = features[i] * 3  # 上采样 + 编码器 + 第1层嵌套
            self.nested_convs[f'{i}_2'] = ConvBlock(in_ch, features[i], dropout)
        
        # 第3层嵌套
        if self.num_layers >= 4:
            in_ch = features[0] * 4  # 上采样 + 编码器 + 第1层 + 第2层
            self.nested_convs['0_3'] = ConvBlock(in_ch, features[0], dropout)
        
        # 输出头
        if deep_supervision:
            # 多个输出头用于深度监督
            self.output_heads = nn.ModuleList()
            for i in range(1, min(4, self.num_layers)):
                self.output_heads.append(
                    nn.Conv2d(features[0], out_channels, kernel_size=1)
                )
        else:
            # 单一输出头
            self.output_head = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（忽略，保持接口一致性）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 存储各层特征
        features = {}
        
        # 编码器路径
        features['0_0'] = self.encoders[0](x)
        for i in range(1, self.num_layers):
            x_pool = self.pools[i-1](features[f'{i-1}_0'])
            features[f'{i}_0'] = self.encoders[i](x_pool)
        
        # 第1层嵌套
        for i in range(self.num_layers - 1):
            up = self.upsamples[i](features[f'{i+1}_0'])
            concat = torch.cat([up, features[f'{i}_0']], dim=1)
            features[f'{i}_1'] = self.nested_convs[f'{i}_1'](concat)
        
        # 第2层嵌套
        for i in range(self.num_layers - 2):
            up = self.upsamples[i](features[f'{i+1}_1'])
            concat = torch.cat([up, features[f'{i}_0'], features[f'{i}_1']], dim=1)
            features[f'{i}_2'] = self.nested_convs[f'{i}_2'](concat)
        
        # 第3层嵌套
        if self.num_layers >= 4:
            up = self.upsamples[0](features['1_2'])
            concat = torch.cat([up, features['0_0'], features['0_1'], features['0_2']], dim=1)
            features['0_3'] = self.nested_convs['0_3'](concat)
        
        # 输出
        if self.deep_supervision:
            # 深度监督：返回最深层的输出
            if '0_3' in features:
                final_feature = features['0_3']
                output_idx = 2
            elif '0_2' in features:
                final_feature = features['0_2']
                output_idx = 1
            else:
                final_feature = features['0_1']
                output_idx = 0
            return self.output_heads[output_idx](final_feature)
        else:
            # 单一输出：使用最深层的特征
            if '0_3' in features:
                final_feature = features['0_3']
            elif '0_2' in features:
                final_feature = features['0_2']
            else:
                final_feature = features['0_1']
            return self.output_head(final_feature)
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """获取特征图（用于可视化）"""
        features = {}
        
        # 编码器路径
        features['0_0'] = self.encoders[0](x)
        for i in range(1, self.num_layers):
            x_pool = self.pools[i-1](features[f'{i-1}_0'])
            features[f'{i}_0'] = self.encoders[i](x_pool)
        
        # 第1层嵌套
        for i in range(self.num_layers - 1):
            up = self.upsamples[i](features[f'{i+1}_0'])
            concat = torch.cat([up, features[f'{i}_0']], dim=1)
            features[f'{i}_1'] = self.nested_convs[f'{i}_1'](concat)
        
        return features
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs"""
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        B, C, H, W = input_shape
        flops = 0
        
        # 编码器FLOPs
        current_h, current_w = H, W
        for i, feature in enumerate(self.features):
            if i == 0:
                in_ch = C
            else:
                in_ch = self.features[i-1]
                current_h //= 2
                current_w //= 2
            
            # 两个3x3卷积
            flops += 2 * in_ch * feature * 9 * current_h * current_w
        
        # 嵌套路径FLOPs（简化估算）
        flops *= 2  # 大约是编码器的2倍
        
        self._flops = flops * B
        return self._flops