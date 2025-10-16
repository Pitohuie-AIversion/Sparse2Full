"""FNO 2D基线模型

实现Fourier Neural Operator (FNO) 2D版本，用于求解偏微分方程。
基于频域卷积的神经算子，适用于物理场重建任务。

Reference:
    Fourier Neural Operator for Parametric Partial Differential Equations
    https://arxiv.org/abs/2010.08895
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class SpectralConv2d(nn.Module):
    """2D频谱卷积层
    
    在频域中进行卷积操作，通过学习频域权重来捕获全局依赖关系。
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            modes1: 第一个维度保留的频率模态数
            modes2: 第二个维度保留的频率模态数
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        
        # 频域权重参数（复数）
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """复数矩阵乘法"""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, in_channels, height, width]
            
        Returns:
            输出张量 [batch, out_channels, height, width]
        """
        batchsize = x.shape[0]
        
        # 禁用AMP进行复数操作
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为float32以避免AMP问题
            x = x.float()
            
            # 计算2D FFT
            x_ft = torch.fft.rfft2(x)
            
            # 初始化输出
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                                dtype=torch.complex64, device=x.device)
            
            # 确保modes不超过实际频域大小
            modes1 = min(self.modes1, x.size(-2))
            modes2 = min(self.modes2, x.size(-1)//2 + 1)
            
            # 频域卷积 - 确保数据类型一致
            x_ft_slice = x_ft[:, :, :modes1, :modes2].to(torch.complex64)
            weights1_slice = self.weights1[:, :, :modes1, :modes2].to(torch.complex64)
            out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(x_ft_slice, weights1_slice)
            
            if modes1 < x.size(-2):
                x_ft_slice2 = x_ft[:, :, -modes1:, :modes2].to(torch.complex64)
                weights2_slice = self.weights2[:, :, :modes1, :modes2].to(torch.complex64)
                out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(x_ft_slice2, weights2_slice)
            
            # 逆FFT回到空间域
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(BaseModel):
    """Fourier Neural Operator 2D模型
    
    基于频域卷积的神经算子，通过学习频域权重来建模算子映射。
    特别适用于求解偏微分方程和物理场重建任务。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        modes1: 第一个维度的频率模态数，默认12
        modes2: 第二个维度的频率模态数，默认12
        width: 隐藏层宽度，默认64
        n_layers: FNO层数，默认4
        activation: 激活函数，默认'gelu'
        **kwargs: 其他参数
    
    Examples:
        >>> model = FNO2d(in_channels=3, out_channels=1, img_size=256)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        activation: str = 'gelu',
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
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
        
        # 输入投影层
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # +2 for coordinates
        
        # FNO层
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
        
        # 输出投影层
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_grid(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        """生成坐标网格
        
        Args:
            shape: (batch_size, channels, height, width)
            device: 设备
            
        Returns:
            坐标网格 [batch_size, height, width, 2]
        """
        batchsize, _, size_x, size_y = shape
        
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（忽略，保持接口一致性）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        batch_size = x.shape[0]
        
        # 生成坐标网格
        grid = self.get_grid(x.shape, x.device)
        
        # 重排维度：[B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        # 拼接坐标信息
        x = torch.cat((x, grid), dim=-1)
        
        # 输入投影
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # FNO层
        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:  # 最后一层不加激活
                x = self.activation(x)
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        # 重排回原始维度
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        return x
    
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
        
        flops = 0
        
        # 输入投影层
        flops += (self.in_channels + 2) * self.width * height * width
        
        # FNO层
        for i in range(self.n_layers):
            # 频谱卷积（FFT + 复数乘法 + IFFT）
            # FFT: O(N log N)
            fft_flops = height * width * torch.log2(torch.tensor(height * width, dtype=torch.float)).item()
            
            # 复数乘法在频域
            spectral_flops = self.width * self.width * self.modes1 * self.modes2 * 2  # 2 for complex
            
            # 1x1卷积
            conv_flops = self.width * self.width * height * width
            
            flops += (fft_flops + spectral_flops + conv_flops)
        
        # 输出投影层
        flops += self.width * 128 * height * width
        flops += 128 * self.out_channels * height * width
        
        self._flops = flops * batch_size
        return self._flops
    
    def get_spectral_weights(self) -> dict:
        """获取频谱权重（用于分析和可视化）
        
        Returns:
            包含所有频谱卷积层权重的字典
        """
        weights = {}
        for i, layer in enumerate(self.conv_layers):
            weights[f'layer_{i}_weights1'] = layer.weights1.detach().cpu()
            weights[f'layer_{i}_weights2'] = layer.weights2.detach().cpu()
        
        return weights
    
    def set_modes(self, modes1: int, modes2: int):
        """动态设置频率模态数（用于消融实验）
        
        Args:
            modes1: 第一个维度的频率模态数
            modes2: 第二个维度的频率模态数
        """
        self.modes1 = min(modes1, self.modes1)
        self.modes2 = min(modes2, self.modes2)
        
        for layer in self.conv_layers:
            layer.modes1 = self.modes1
            layer.modes2 = self.modes2


# 别名，保持向后兼容
FNOModel = FNO2d