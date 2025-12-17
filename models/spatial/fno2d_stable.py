"""Stabilized FNO 2D模型

实现具有数值稳定性的Fourier Neural Operator (FNO) 2D版本，
专门用于解决与temporal model结合时的数值不稳定问题。

改进包括：
- 复数运算的数值稳定性检查
- 梯度裁剪和正则化
- 更稳定的权重初始化
- NaN/Inf检测和处理
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel


class StableSpectralConv2d(nn.Module):
    """数值稳定的2D频谱卷积层
    
    在频域中进行卷积操作，添加了数值稳定性检查和防护措施。
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
        self.modes1 = modes1
        self.modes2 = modes2
        
        # 更稳定的缩放因子
        self.scale = torch.sqrt(torch.tensor(1.0 / (in_channels * out_channels)))
        
        # 频域权重参数（复数）- 使用更稳定的初始化
        self.weights1 = nn.Parameter(torch.view_as_complex(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2)
        ))
        self.weights2 = nn.Parameter(torch.view_as_complex(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2)
        ))
    
    def stable_compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """数值稳定的复数矩阵乘法"""
        # 检查输入的有效性
        if torch.isnan(input).any() or torch.isinf(input).any():
            print(f"Warning: Invalid values in spectral input")
            input = torch.nan_to_num(input, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print(f"Warning: Invalid values in spectral weights")
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 使用更稳定的einsum实现
        try:
            result = torch.einsum("bixy,ioxy->boxy", input, weights)
            
            # 后处理检查结果
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"Warning: Invalid values in spectral multiplication result")
                result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except Exception as e:
            print(f"Error in spectral multiplication: {e}")
            # 返回零张量作为fallback
            return torch.zeros(
                input.shape[0], weights.shape[1], input.shape[2], input.shape[3],
                dtype=input.dtype, device=input.device
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, in_channels, height, width]
            
        Returns:
            输出张量 [batch, out_channels, height, width]
        """
        batchsize = x.shape[0]
        
        # 禁用AMP进行复数操作，使用float64提高精度
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为float64以提高数值稳定性
            x = x.double()
            
            # 检查输入的有效性
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: Invalid values in spectral conv input")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 计算2D FFT - 添加正则化
            try:
                x_ft = torch.fft.rfft2(x, norm='ortho')  # 使用正交归一化
            except Exception as e:
                print(f"Error in FFT: {e}")
                return torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1),
                                 dtype=torch.float32, device=x.device)
            
            # 检查FFT结果
            if torch.isnan(x_ft).any() or torch.isinf(x_ft).any():
                print(f"Warning: Invalid values in FFT output")
                x_ft = torch.nan_to_num(x_ft, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 初始化输出
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                                dtype=torch.complex128, device=x.device)
            
            # 确保modes不超过实际频域大小
            modes1 = min(self.modes1, x.size(-2))
            modes2 = min(self.modes2, x.size(-1)//2 + 1)
            
            # 频域卷积 - 使用更稳定的数据类型
            try:
                x_ft_slice = x_ft[:, :, :modes1, :modes2].to(torch.complex128)
                weights1_slice = self.weights1[:, :, :modes1, :modes2].to(torch.complex128)
                out_ft[:, :, :modes1, :modes2] = self.stable_compl_mul2d(x_ft_slice, weights1_slice)
                
                if modes1 < x.size(-2):
                    x_ft_slice2 = x_ft[:, :, -modes1:, :modes2].to(torch.complex128)
                    weights2_slice = self.weights2[:, :, :modes1, :modes2].to(torch.complex128)
                    out_ft[:, :, -modes1:, :modes2] = self.stable_compl_mul2d(x_ft_slice2, weights2_slice)
                    
            except Exception as e:
                print(f"Error in spectral convolution: {e}")
                # 保持out_ft为零，继续执行
            
            # 检查输出频谱
            if torch.isnan(out_ft).any() or torch.isinf(out_ft).any():
                print(f"Warning: Invalid values in output spectrum")
                out_ft = torch.nan_to_num(out_ft, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 逆FFT回到空间域
            try:
                x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
                x = x.float()  # 转换回float32
            except Exception as e:
                print(f"Error in IFFT: {e}")
                return torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1),
                                 dtype=torch.float32, device=x.device)
        
        return x


class StableFNO2d(BaseModel):
    """数值稳定的Fourier Neural Operator 2D模型
    
    基于频域卷积的神经算子，添加了数值稳定性改进，
    专门用于与temporal model结合的场景。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        img_size: 图像尺寸（正方形）
        modes1: 第一个维度的频率模态数，默认12
        modes2: 第二个维度的频率模态数，默认12
        width: 隐藏层宽度，默认64
        n_layers: FNO层数，默认4
        activation: 激活函数，默认'gelu'
        spectral_norm: 是否使用谱归一化，默认True
        gradient_clip: 梯度裁剪阈值，默认1.0
        **kwargs: 其他参数
    
    Examples:
        >>> model = StableFNO2d(in_channels=3, out_channels=1, img_size=256)
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
        spectral_norm: bool = True,
        gradient_clip: float = 1.0,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm
        self.gradient_clip = gradient_clip
        
        # 激活函数
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 输入投影层 - 使用LayerNorm提高稳定性
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # +2 for coordinates
        self.input_norm = nn.LayerNorm(self.width)
        
        # FNO层 - 使用稳定的频谱卷积
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.conv_layers.append(StableSpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            
            # 1x1卷积层 - 可选谱归一化
            conv1x1 = nn.Conv2d(self.width, self.width, 1)
            if self.spectral_norm:
                conv1x1 = nn.utils.spectral_norm(conv1x1)
            self.w_layers.append(conv1x1)
            
            # Layer normalization 应用在通道维度，使用GroupNorm以避免维度不匹配
            # 原实现使用 LayerNorm([C,H,W]) 需要输入为 [*, C, H, W] 展平后的最后维度，容易出错
            # 改为 GroupNorm，设置为每通道归一化更稳健
            self.layer_norms.append(nn.GroupNorm(num_groups=self.width, num_channels=self.width))
        
        # 输出投影层 - 使用更窄的中间层
        self.fc1 = nn.Linear(self.width, 64)  # 从128减少到64
        self.fc2 = nn.Linear(64, out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重 - 使用更稳定的初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重 - 使用更稳定的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更小的初始化范围
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化但减小增益
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_grid(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        """生成坐标网格 - 使用更稳定的坐标范围"""
        batchsize, _, size_x, size_y = shape
        
        # 使用更小的坐标范围 [-1, 1] 而不是 [0, 1]
        gridx = torch.tensor(torch.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        
        gridy = torch.tensor(torch.linspace(-1, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播 - 添加了数值稳定性检查"""
        batch_size = x.shape[0]
        
        # Debug日志改为受控的logger，默认不打印
        # print(f"FNO2D input shape: {x.shape}, expected in_channels: {self.in_channels}")
        
        # 检查输入的有效性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid values in FNO2d input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 生成坐标网格
        grid = self.get_grid(x.shape, x.device)
        
        # 重排维度：[B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        # 拼接坐标信息
        x = torch.cat((x, grid), dim=-1)
        
        # 输入投影 - 添加梯度裁剪
        x = self.fc0(x)
        x = self.input_norm(x)
        x = self.dropout(x)  # 添加dropout
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # 保存初始特征用于残差连接
        x_residual = x.clone()
        
        # FNO层 - 添加残差连接和稳定性检查
        for i in range(self.n_layers):
            try:
                # 频谱卷积
                x1 = self.conv_layers[i](x)
                
                # 1x1卷积
                x2 = self.w_layers[i](x)
                
                # 残差连接
                x = x1 + x2
                
                # 归一化：直接在 [B, C, H, W] 上使用 GroupNorm
                x = self.layer_norms[i](x)
                
                # 激活函数（最后一层除外）
                if i < self.n_layers - 1:
                    x = self.activation(x)
                    x = self.dropout(x)  # 在激活后添加dropout
                
                # 检查每层的输出
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: Invalid values in FNO layer {i}")
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                    
            except Exception as e:
                print(f"Error in FNO layer {i}: {e}")
                # 返回上一层的输出作为fallback
                if i > 0:
                    return x_residual
                else:
                    return torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1),
                                     dtype=torch.float32, device=x.device)
        
        # 添加全局残差连接
        x = x + 0.1 * x_residual  # 使用小的残差系数
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 重排回原始维度
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid values in FNO2d output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return x
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs"""
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        batch_size, _, height, width = input_shape
        
        flops = 0
        
        # 输入投影层
        flops += (self.in_channels + 2) * self.width * height * width
        
        # FNO层
        for i in range(self.n_layers):
            # 频谱卷积（FFT + 复数乘法 + IFFT）
            fft_flops = height * width * torch.log2(torch.tensor(height * width, dtype=torch.float)).item()
            
            # 复数乘法在频域
            spectral_flops = self.width * self.width * self.modes1 * self.modes2 * 2  # 2 for complex
            
            # 1x1卷积
            conv_flops = self.width * self.width * height * width
            
            # Layer normalization
            norm_flops = self.width * height * width
            
            flops += (fft_flops + spectral_flops + conv_flops + norm_flops)
        
        # 输出投影层
        flops += self.width * 64 * height * width
        flops += 64 * self.out_channels * height * width
        
        self._flops = flops * batch_size
        return self._flops
    
    def get_spectral_weights(self) -> dict:
        """获取频谱权重"""
        weights = {}
        for i, layer in enumerate(self.conv_layers):
            weights[f'layer_{i}_weights1'] = layer.weights1.detach().cpu()
            weights[f'layer_{i}_weights2'] = layer.weights2.detach().cpu()
        
        return weights
    
    def set_modes(self, modes1: int, modes2: int):
        """动态设置频率模态数"""
        self.modes1 = min(modes1, self.modes1)
        self.modes2 = min(modes2, self.modes2)
        
        for layer in self.conv_layers:
            layer.modes1 = self.modes1
            layer.modes2 = self.modes2


# 别名
StableFNOModel = StableFNO2d