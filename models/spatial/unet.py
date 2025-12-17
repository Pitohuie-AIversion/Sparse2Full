"""经典U-Net基线模型

实现标准的U-Net架构，用作基线对比模型。
遵循统一接口：forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel


class DoubleConv(nn.Module):
    """双卷积块：Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块：MaxPool2d -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块：Upsample -> Conv -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # 使用双线性插值或转置卷积进行上采样
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.dynamic_conv = None
        self.out_channels = out_channels
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        
        # 动态构建卷积块以匹配通道数
        if self.dynamic_conv is None or getattr(self.dynamic_conv, 'in_channels', None) != x.shape[1]:
            mid_ch = max(x.shape[1] // 2, 1)
            self.dynamic_conv = DoubleConv(x.shape[1], self.out_channels, mid_ch).to(x.device)
        return self.dynamic_conv(x)


class OutConv(nn.Module):
    """输出卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(BaseModel):
    """经典U-Net模型
    
    标准的U-Net架构，包含编码器-解码器结构和跳跃连接。
    适用于图像到图像的转换任务，如超分辨率和图像修复。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        features: 特征通道数列表，默认[64, 128, 256, 512]
        bilinear: 是否使用双线性插值上采样，默认True
        dropout: Dropout概率，默认0.0
        **kwargs: 其他参数
    
    Examples:
        >>> model = UNet(in_channels=3, out_channels=1, img_size=256)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        img_size: int | None = None,
        features: List[int] = None,
        bilinear: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        if in_channels is None:
            in_channels = kwargs.pop('in_ch', kwargs.pop('in_chans', 1))
        if out_channels is None:
            out_channels = kwargs.pop('out_ch', kwargs.pop('num_classes', 1))
        if img_size is None:
            img_size = kwargs.get('img_size', 128)
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.features = features
        self.bilinear = bilinear
        self.dropout = dropout
        
        # 输入卷积
        self.inc = DoubleConv(in_channels, features[0])
        
        # 编码器（下采样路径）
        # 根据features长度自适应构建编码器
        self.down_blocks = nn.ModuleList()
        for i in range(1, len(features)):
            self.down_blocks.append(Down(features[i-1], features[i]))
        
        # 瓶颈层
        factor = 2 if bilinear else 1
        # 瓶颈层（当层数>=4），否则用最后一层作为瓶颈
        if len(features) >= 4:
            self.down_bottleneck = Down(features[3], features[3] * 2 // factor)
            bottleneck_out = features[3] * 2 // factor
        else:
            self.down_bottleneck = None
            bottleneck_out = features[-1]
        
        # 解码器（上采样路径）
        # 动态解码器路径
        self.up_blocks = nn.ModuleList()
        up_in = bottleneck_out if self.down_bottleneck is not None else features[-1]
        for i in range(len(features)-1, 0, -1):
            out_ch = features[i-1]
            self.up_blocks.append(Up(up_in, out_ch, bilinear))
            up_in = out_ch
        
        # 输出卷积
        self.outc = OutConv(features[0], out_channels)
        
        # Dropout层（可选）
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)
        else:
            self.dropout_layer = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重 - 使用更保守的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier初始化替代Kaiming初始化，减少初始权重幅度
                nn.init.xavier_uniform_(m.weight, gain=0.5)
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
        # 编码器路径
        x1 = self.inc(x)
        downs = []
        prev = x1
        for down in self.down_blocks:
            prev = down(prev)
            downs.append(prev)
        if self.down_bottleneck is not None:
            x_bottleneck = self.down_bottleneck(prev)
        else:
            x_bottleneck = prev
        
        # 应用Dropout（如果启用）
        if self.dropout_layer is not None:
            x_bottleneck = self.dropout_layer(x_bottleneck)
        
        # 解码器路径
        up_input = x_bottleneck
        # 对齐跳跃连接从最后一层开始
        skip_list = [x1] + downs[:-1]
        for i, up in enumerate(self.up_blocks):
            skip = skip_list[-(i+1)] if i < len(skip_list) else None
            up_input = up(up_input, skip if skip is not None else up_input)
        x = up_input
        
        # 输出
        logits = self.outc(x)
        
        return logits
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs（更精确的估算）
        
        Args:
            input_shape: 输入形状，默认为(1, in_channels, img_size, img_size)
            
        Returns:
            FLOPs数量
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        batch_size, _, height, width = input_shape
        
        # 粗略估算：每个卷积层的FLOPs
        flops = 0
        
        # 编码器路径
        h, w = height, width
        for i, feat in enumerate(self.features):
            if i == 0:
                # 输入卷积：2个3x3卷积
                flops += 2 * (self.in_channels * feat * 3 * 3 * h * w)
            else:
                # 下采样：MaxPool + 2个3x3卷积
                h, w = h // 2, w // 2
                prev_feat = self.features[i-1]
                flops += 2 * (prev_feat * feat * 3 * 3 * h * w)
        
        # 瓶颈层
        h, w = h // 2, w // 2
        bottleneck_feat = self.features[-1] * 2 if not self.bilinear else self.features[-1]
        flops += 2 * (self.features[-1] * bottleneck_feat * 3 * 3 * h * w)
        
        # 解码器路径（对称）
        for i in range(len(self.features)):
            h, w = h * 2, w * 2
            # 上采样 + 卷积
            if i == 0:
                in_feat = bottleneck_feat + self.features[-(i+1)]
                out_feat = self.features[-(i+1)]
            else:
                in_feat = self.features[-(i)] + self.features[-(i+1)]
                out_feat = self.features[-(i+1)]
            
            flops += 2 * (in_feat * out_feat * 3 * 3 * h * w)
        
        # 输出卷积
        flops += self.features[0] * self.out_channels * 1 * 1 * height * width
        
        self._flops = flops * batch_size
        return self._flops
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        # 冻结编码器部分
        for param in self.inc.parameters():
            param.requires_grad = False
        for param in self.down1.parameters():
            param.requires_grad = False
        for param in self.down2.parameters():
            param.requires_grad = False
        for param in self.down3.parameters():
            param.requires_grad = False
        for param in self.down4.parameters():
            param.requires_grad = False
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取中间特征图（用于可视化和分析）
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            
        Returns:
            特征图列表
        """
        features = []
        
        # 编码器路径
        x1 = self.inc(x)
        features.append(x1)
        
        x2 = self.down1(x1)
        features.append(x2)
        
        x3 = self.down2(x2)
        features.append(x3)
        
        x4 = self.down3(x3)
        features.append(x4)
        
        x5 = self.down4(x4)
        features.append(x5)
        
        return features


# 别名，保持向后兼容
UNetModel = UNet
