"""U-FNO瓶颈模型

结合U-Net和FNO的混合架构，在U-Net的瓶颈层使用FNO进行全局建模。
这种设计结合了U-Net的局部特征提取能力和FNO的全局依赖建模能力。

Reference:
    U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow
    https://doi.org/10.1016/j.advwatres.2022.104180
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from .fno2d import SpectralConv2d


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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 上采样后的通道数保持不变，然后与跳跃连接拼接
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FNOBottleneck(nn.Module):
    """FNO瓶颈层
    
    在U-Net的瓶颈处使用FNO进行全局特征建模，
    结合频域卷积的全局感受野和空间域卷积的局部特征。
    """
    
    def __init__(
        self, 
        channels: int, 
        modes1: int = 16, 
        modes2: int = 16, 
        n_layers: int = 2,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers
        
        # 激活函数
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # FNO层
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(channels, channels, modes1, modes2))
            self.conv_layers.append(nn.Conv2d(channels, channels, 1))
        
        # 归一化层
        self.norm_layers = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(n_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C, H, W]
        """
        for i in range(self.n_layers):
            # FNO分支
            x1 = self.spectral_layers[i](x)
            
            # 1x1卷积分支
            x2 = self.conv_layers[i](x)
            
            # 残差连接
            x = x1 + x2 + x
            
            # 归一化和激活
            x = self.norm_layers[i](x)
            if i < self.n_layers - 1:  # 最后一层不加激活
                x = self.activation(x)
        
        return x


class UFNOUNet(BaseModel):
    """U-FNO瓶颈模型
    
    结合U-Net和FNO的混合架构：
    - 编码器：标准U-Net下采样路径
    - 瓶颈层：FNO进行全局特征建模
    - 解码器：标准U-Net上采样路径
    
    这种设计充分利用了U-Net的多尺度特征提取和FNO的全局建模能力。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        features: U-Net特征通道数列表，默认[64, 128, 256, 512]
        fno_modes1: FNO第一个维度的频率模态数，默认16
        fno_modes2: FNO第二个维度的频率模态数，默认16
        fno_layers: FNO层数，默认2
        bilinear: 是否使用双线性插值上采样，默认True
        dropout: Dropout概率，默认0.0
        **kwargs: 其他参数
    
    Examples:
        >>> model = UFNOUNet(in_channels=3, out_channels=1, img_size=256)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        features: List[int] = None,
        fno_modes1: int = 16,
        fno_modes2: int = 16,
        fno_layers: int = 2,
        bilinear: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.features = features
        self.fno_modes1 = fno_modes1
        self.fno_modes2 = fno_modes2
        self.fno_layers = fno_layers
        self.bilinear = bilinear
        self.dropout = dropout
        
        # 输入卷积
        self.inc = DoubleConv(in_channels, features[0])
        
        # 编码器（下采样路径）
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # FNO瓶颈层
        factor = 2 if bilinear else 1
        bottleneck_channels = features[3] * 2 // factor
        
        self.down4 = Down(features[3], bottleneck_channels)
        self.fno_bottleneck = FNOBottleneck(
            channels=bottleneck_channels,
            modes1=fno_modes1,
            modes2=fno_modes2,
            n_layers=fno_layers
        )
        
        # 解码器（上采样路径）
        # 修正通道数计算，考虑跳跃连接
        self.up1 = Up(bottleneck_channels + features[3], features[3] // factor, bilinear)
        self.up2 = Up((features[3] // factor) + features[2], features[2] // factor, bilinear)
        self.up3 = Up((features[2] // factor) + features[1], features[1] // factor, bilinear)
        self.up4 = Up((features[1] // factor) + features[0], features[0], bilinear)
        
        # 输出卷积
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Dropout层（可选）
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)
        else:
            self.dropout_layer = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
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
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # FNO瓶颈层
        x5 = self.fno_bottleneck(x5)
        
        # 应用Dropout（如果启用）
        if self.dropout_layer is not None:
            x5 = self.dropout_layer(x5)
        
        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
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
        
        # U-Net部分的FLOPs（简化估算）
        unet_flops = 0
        
        # 编码器路径
        h, w = height, width
        for i, feat in enumerate(self.features):
            if i == 0:
                # 输入卷积：2个3x3卷积
                unet_flops += 2 * (self.in_channels * feat * 3 * 3 * h * w)
            else:
                # 下采样：MaxPool + 2个3x3卷积
                h, w = h // 2, w // 2
                prev_feat = self.features[i-1]
                unet_flops += 2 * (prev_feat * feat * 3 * 3 * h * w)
        
        # 瓶颈层
        h, w = h // 2, w // 2
        bottleneck_feat = self.features[-1] * 2 if not self.bilinear else self.features[-1]
        unet_flops += 2 * (self.features[-1] * bottleneck_feat * 3 * 3 * h * w)
        
        # FNO瓶颈层的FLOPs
        fno_flops = 0
        for _ in range(self.fno_layers):
            # FFT + 复数乘法 + IFFT
            fft_flops = h * w * torch.log2(torch.tensor(h * w, dtype=torch.float)).item()
            spectral_flops = bottleneck_feat * bottleneck_feat * self.fno_modes1 * self.fno_modes2 * 2
            conv_flops = bottleneck_feat * bottleneck_feat * h * w
            fno_flops += (fft_flops + spectral_flops + conv_flops)
        
        # 解码器路径（对称）
        decoder_flops = 0
        for i in range(len(self.features)):
            h, w = h * 2, w * 2
            if i == 0:
                in_feat = bottleneck_feat + self.features[-(i+1)]
                out_feat = self.features[-(i+1)]
            else:
                in_feat = self.features[-(i)] + self.features[-(i+1)]
                out_feat = self.features[-(i+1)]
            
            decoder_flops += 2 * (in_feat * out_feat * 3 * 3 * h * w)
        
        # 输出卷积
        output_flops = self.features[0] * self.out_channels * 1 * 1 * height * width
        
        total_flops = unet_flops + fno_flops + decoder_flops + output_flops
        self._flops = total_flops * batch_size
        return self._flops
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        # 冻结U-Net编码器部分
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
    
    def freeze_fno(self) -> None:
        """冻结FNO瓶颈层参数"""
        for param in self.fno_bottleneck.parameters():
            param.requires_grad = False
    
    def get_fno_weights(self) -> dict:
        """获取FNO权重（用于分析和可视化）
        
        Returns:
            包含FNO瓶颈层权重的字典
        """
        weights = {}
        for i, layer in enumerate(self.fno_bottleneck.spectral_layers):
            weights[f'fno_layer_{i}_weights1'] = layer.weights1.detach().cpu()
            weights[f'fno_layer_{i}_weights2'] = layer.weights2.detach().cpu()
        
        return weights
    
    def set_fno_modes(self, modes1: int, modes2: int):
        """动态设置FNO频率模态数
        
        Args:
            modes1: 第一个维度的频率模态数
            modes2: 第二个维度的频率模态数
        """
        self.fno_modes1 = modes1
        self.fno_modes2 = modes2
        
        for layer in self.fno_bottleneck.spectral_layers:
            layer.modes1 = min(modes1, layer.modes1)
            layer.modes2 = min(modes2, layer.modes2)


# 别名，保持向后兼容
UFNOModel = UFNOUNet