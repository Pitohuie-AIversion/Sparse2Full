"""时序编码器模块

实现因果卷积的时序编码器，支持AR/NAR训练模式。
遵循黄金法则：观测算子H与训练DC必须复用同一实现与配置。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


class CausalConv1D(nn.Module):
    """因果1D卷积层
    
    确保时间步t的输出只依赖于t及之前的输入，满足因果性约束。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """初始化因果卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 膨胀率
            groups: 分组卷积数
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, C, T]
        """
        # 应用卷积
        out = self.conv(x)
        
        # 移除右侧填充以保持因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return out


class TemporalConv1D(nn.Module):
    """时序1D卷积编码器
    
    使用因果卷积和残差连接构建时序特征编码器。
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        """初始化时序卷积编码器
        
        Args:
            in_channels: 输入通道数
            hidden_channels: 隐藏层通道数
            num_layers: 层数
            kernel_size: 卷积核大小
            dilation_base: 膨胀率基数
            dropout: Dropout概率
            activation: 激活函数类型
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 输入投影
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # 因果卷积层
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 计算膨胀率
            dilation = dilation_base ** i
            
            # 因果卷积
            conv_layer = CausalConv1D(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            self.conv_layers.append(conv_layer)
            
            # 层归一化
            if use_layer_norm:
                norm_layer = nn.LayerNorm(hidden_channels)
                self.norm_layers.append(norm_layer)
            
            # Dropout
            dropout_layer = nn.Dropout(dropout)
            self.dropout_layers.append(dropout_layer)
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出投影
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, C, T]
        """
        # 输入投影
        residual = x
        x = self.input_proj(x)
        
        # 因果卷积层
        for i in range(self.num_layers):
            # 保存残差
            layer_residual = x
            
            # 因果卷积
            x = self.conv_layers[i](x)
            
            # 层归一化
            if self.use_layer_norm:
                # 转换维度用于LayerNorm: [B, C, T] -> [B, T, C]
                x = x.transpose(1, 2)
                x = self.norm_layers[i](x)
                x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            
            # 激活函数
            x = self.activation(x)
            
            # Dropout
            x = self.dropout_layers[i](x)
            
            # 残差连接
            if self.use_residual:
                x = x + layer_residual
        
        # 输出投影
        x = self.output_proj(x)
        
        # 全局残差连接
        if self.use_residual:
            x = x + residual
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码
    
    为时序数据添加位置信息。
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1
    ):
        """初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, T, C]
            
        Returns:
            添加位置编码的张量 [B, T, C]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """完整的时序编码器
    
    结合因果卷积、位置编码和多尺度特征提取。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_seq_len: int = 1000,
        activation: str = "gelu",
        max_spatial_dim: int = 10000  # 最大空间维度，用于预注册层
    ):
        """初始化时序编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_conv_layers: 卷积层数
            kernel_size: 卷积核大小
            dilation_base: 膨胀率基数
            dropout: Dropout概率
            use_positional_encoding: 是否使用位置编码
            max_seq_len: 最大序列长度
            activation: 激活函数类型
            max_spatial_dim: 最大空间维度，用于预注册动态层
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                d_model=hidden_dim,
                max_len=max_seq_len,
                dropout=dropout
            )
        
        # 时序卷积编码器
        self.temporal_conv = TemporalConv1D(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_conv_layers,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            activation=activation
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 预注册常见维度的投影层（保持向后兼容）
        self.adaptive_input_proj = nn.ModuleDict()
        self.adaptive_output_proj = nn.ModuleDict()
        self.adaptive_layer_norm = nn.ModuleDict()

        common_spatial_dims = [64 * 64 * 2, 128 * 128 * 2]
        for dim in common_spatial_dims:
            if dim != input_dim:
                self.adaptive_input_proj[str(dim)] = nn.Linear(dim, hidden_dim)
                self.adaptive_output_proj[str(dim)] = nn.Linear(hidden_dim, dim)
                self.adaptive_layer_norm[str(dim)] = nn.LayerNorm(dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量 [B, T, C] 或 [B, T, C, H, W]
            mask: 注意力掩码 [B, T] (可选)
            
        Returns:
            包含编码结果的字典
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 处理 5D 时空输入 [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            
            # 模式 1: 通道数 C 与 input_dim 匹配（逐空间像素的时序编码）
            if C == self.input_dim:
                # 重排为 [B*H*W, T, C] 进行时序编码，保持空间局部性与分辨率不变性
                x_reshaped = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
                residual = x_reshaped
                
                h = self.input_proj(x_reshaped)  # [B*H*W, T, hidden_dim]
                if self.use_positional_encoding:
                    h = self.pos_encoding(h)
                
                h = h.transpose(1, 2)  # [B*H*W, hidden_dim, T]
                h = self.temporal_conv(h)  # [B*H*W, hidden_dim, T]
                h = h.transpose(1, 2)  # [B*H*W, T, hidden_dim]
                
                out = self.output_proj(h)  # [B*H*W, T, C]
                out = self.layer_norm(out + residual)
                
                out = out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
                return {
                    'encoded_sequence': out,
                    'sequence_length': seq_len,
                    'batch_size': batch_size
                }
            
            # 模式 2: 全图展平空间维度 [B, T, C*H*W]
            flattened_dim = C * H * W
            x_in = x.reshape(B, T, flattened_dim)
            spatial_shape = (C, H, W)
        elif x.dim() == 3:  # [B, T, C]
            spatial_shape = None
            flattened_dim = x.size(-1)
            x_in = x
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}，期望 3D [B, T, C] 或 5D [B, T, C, H, W]")
        
        residual = x_in
        dim_key = str(flattened_dim)
        
        # 输入投影
        if flattened_dim == self.input_dim:
            h = self.input_proj(x_in)
        elif dim_key in self.adaptive_input_proj:
            h = self.adaptive_input_proj[dim_key](x_in)
        else:
            raise ValueError(
                f"TemporalEncoder 输入维度不匹配: 模型配置 input_dim={self.input_dim}，"
                f"实际输入特征维度为 {flattened_dim} (C={x.shape[2] if x.dim() == 5 else flattened_dim})"
            )
        
        # 位置编码
        if self.use_positional_encoding:
            h = self.pos_encoding(h)
        
        # 时序卷积编码: [B, T, hidden_dim] -> [B, hidden_dim, T] -> [B, T, hidden_dim]
        h = h.transpose(1, 2)
        h = self.temporal_conv(h)
        h = h.transpose(1, 2)
        
        # 输出投影和残差连接
        if flattened_dim == self.input_dim:
            out = self.output_proj(h)
            out = self.layer_norm(out + residual)
        else:
            out = self.adaptive_output_proj[dim_key](h)
            out = self.adaptive_layer_norm[dim_key](out + residual)
        
        # 恢复空间形状
        if spatial_shape is not None:
            C, H, W = spatial_shape
            out = out.reshape(batch_size, seq_len, C, H, W)
        
        return {
            'encoded_sequence': out,
            'sequence_length': seq_len,
            'batch_size': batch_size
        }
    
    def get_receptive_field(self) -> int:
        """计算感受野大小
        
        Returns:
            感受野大小
        """
        # 从temporal_conv获取参数
        if hasattr(self.temporal_conv, 'conv_layers'):
            receptive_field = 1
            for conv_layer in self.temporal_conv.conv_layers:
                # 获取卷积层的参数
                if hasattr(conv_layer, 'dilation'):
                    dilation = conv_layer.dilation[0] if isinstance(conv_layer.dilation, tuple) else conv_layer.dilation
                else:
                    dilation = 1
                if hasattr(conv_layer, 'kernel_size'):
                    kernel_size = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
                else:
                    kernel_size = 3
                receptive_field += (kernel_size - 1) * dilation
            return receptive_field
        else:
            # 简单估计：假设3层，kernel_size=3，指数增长的dilation
            return 1 + (3 - 1) * (1 + 2 + 4)  # 1 + 2*7 = 15


# 工厂函数
def create_temporal_encoder(
    input_dim: int,
    config: Optional[Dict] = None
) -> TemporalEncoder:
    """创建时序编码器
    
    Args:
        input_dim: 输入维度
        config: 配置字典
        
    Returns:
        时序编码器实例
    """
    default_config = {
        'hidden_dim': 128,
        'num_conv_layers': 4,
        'kernel_size': 3,
        'dilation_base': 2,
        'dropout': 0.1,
        'use_positional_encoding': True,
        'max_seq_len': 1000,
        'activation': 'gelu'
    }
    
    if config:
        default_config.update(config)
    
    return TemporalEncoder(input_dim=input_dim, **default_config)


# 单元测试
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试因果卷积
    print("🧪 测试因果卷积...")
    causal_conv = CausalConv1D(in_channels=64, out_channels=64, kernel_size=3)
    x = torch.randn(2, 64, 10)  # [B, C, T]
    out = causal_conv(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
    assert out.shape == x.shape, "因果卷积输出形状不匹配"
    
    # 测试时序卷积编码器
    print("🧪 测试时序卷积编码器...")
    temporal_conv = TemporalConv1D(in_channels=64, hidden_channels=128, num_layers=3)
    out = temporal_conv(x)
    print(f"时序卷积输出形状: {out.shape}")
    assert out.shape == x.shape, "时序卷积输出形状不匹配"
    
    # 测试位置编码
    print("🧪 测试位置编码...")
    pos_enc = PositionalEncoding(d_model=64, max_len=100)
    x_pos = torch.randn(2, 10, 64)  # [B, T, C]
    out_pos = pos_enc(x_pos)
    print(f"位置编码输出形状: {out_pos.shape}")
    assert out_pos.shape == x_pos.shape, "位置编码输出形状不匹配"
    
    # 测试完整时序编码器
    print("🧪 测试完整时序编码器...")
    
    # 测试3D输入 [B, T, C]
    encoder = create_temporal_encoder(input_dim=64)
    x_3d = torch.randn(2, 10, 64)  # [B, T, C]
    result_3d = encoder(x_3d)
    print(f"3D输入形状: {x_3d.shape}")
    print(f"3D输出形状: {result_3d['encoded_sequence'].shape}")
    assert result_3d['encoded_sequence'].shape == x_3d.shape, "3D编码器输出形状不匹配"
    
    # 测试5D逐像素通道输入 [B, T, C, H, W] (C == input_dim)
    x_5d_channel = torch.randn(2, 10, 64, 32, 32)
    result_5d_channel = encoder(x_5d_channel)
    print(f"5D通道模式输入形状: {x_5d_channel.shape}, 输出形状: {result_5d_channel['encoded_sequence'].shape}")
    assert result_5d_channel['encoded_sequence'].shape == x_5d_channel.shape, "5D通道模式输出形状不匹配"
    
    # 测试5D全图展平输入 [B, T, C, H, W] (C*H*W == input_dim)
    encoder_flat = create_temporal_encoder(input_dim=3 * 16 * 16)
    x_5d_flat = torch.randn(2, 10, 3, 16, 16)
    result_5d_flat = encoder_flat(x_5d_flat)
    print(f"5D展平模式输入形状: {x_5d_flat.shape}, 输出形状: {result_5d_flat['encoded_sequence'].shape}")
    assert result_5d_flat['encoded_sequence'].shape == x_5d_flat.shape, "5D展平模式输出形状不匹配"
    
    # 测试感受野计算
    receptive_field = encoder.get_receptive_field()
    print(f"感受野大小: {receptive_field}")
    
    print("✅ 时序编码器模块测试完成！")
