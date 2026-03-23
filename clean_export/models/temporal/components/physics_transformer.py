"""
物理感知Transformer时序模块

专门为PDE求解设计的Transformer架构，结合：
1. 物理信息位置编码
2. 多尺度时序注意力
3. 物理约束机制
4. 因果性保证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from ..base_temporal import BaseTemporalModel

logger = logging.getLogger(__name__)


class PhysicsInformedPositionalEncoding(nn.Module):
    """物理信息位置编码
    
    结合时间、物理状态（速度、加速度）的位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # 基础正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        # 物理信息投影层
        self.physics_proj = nn.Linear(d_model * 3, d_model)  # 位置+速度+加速度
        
    def forward(self, x: torch.Tensor, physics_info: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            x: [seq_len, batch_size, d_model]
            physics_info: {'velocity': [seq_len, batch_size, d_model], 
                         'acceleration': [seq_len, batch_size, d_model]}
        """
        seq_len = x.size(0)
        
        # 基础位置编码
        pos_encoding = self.pe[:seq_len, :].repeat(1, x.size(1), 1)
        
        if physics_info is not None:
            # 添加物理信息
            velocity = physics_info.get('velocity', torch.zeros_like(x))
            acceleration = physics_info.get('acceleration', torch.zeros_like(x))
            
            # 拼接物理信息
            physics_features = torch.cat([pos_encoding, velocity, acceleration], dim=-1)
            pos_encoding = self.physics_proj(physics_features)
        
        return self.dropout(x + pos_encoding)


class MultiScaleAttention(nn.Module):
    """多尺度时序注意力
    
    不同注意力头专门处理不同时间尺度的物理过程
    """
    
    def __init__(self, d_model: int, nhead: int = 8, scales: Optional[list] = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        if scales is None:
            scales = [1, 2, 4, 8, 16, 32, 64, 128]  # 多时间尺度
        self.scales = scales[:nhead]  # 每个头一个尺度
        
        # 为每个尺度创建专门的注意力层
        self.attention_layers = nn.ModuleList([
            self._create_scale_attention(scale) for scale in self.scales
        ])
        
        # 输出融合层
        self.output_proj = nn.Linear(d_model * nhead, d_model)
        
    def _create_scale_attention(self, scale: int):
        """为特定时间尺度创建注意力层"""
        return nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=1,  # 每个尺度单头
            batch_first=True
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            attn_mask: 注意力掩码
        """
        batch_size, seq_len, _ = query.shape
        outputs = []
        attention_weights = []
        
        for i, (scale, attention_layer) in enumerate(zip(self.scales, self.attention_layers)):
            # 根据尺度调整感受野
            if scale > 1:
                # 下采样以扩大感受野
                scale_query = F.avg_pool1d(
                    query.transpose(1, 2), 
                    kernel_size=scale, stride=scale
                ).transpose(1, 2)
                scale_key = F.avg_pool1d(
                    key.transpose(1, 2), 
                    kernel_size=scale, stride=scale
                ).transpose(1, 2)
                scale_value = F.avg_pool1d(
                    value.transpose(1, 2), 
                    kernel_size=scale, stride=scale
                ).transpose(1, 2)
            else:
                scale_query, scale_key, scale_value = query, key, value
            
            # 应用注意力
            scale_output, scale_weights = attention_layer(
                scale_query, scale_key, scale_value, attn_mask=attn_mask
            )
            
            # 上采样回原尺寸
            if scale > 1:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            outputs.append(scale_output)
            attention_weights.append(scale_weights)
            
        # 融合多尺度输出
        multi_scale_output = torch.cat(outputs, dim=-1)
        final_output = self.output_proj(multi_scale_output)
        
        return final_output, attention_weights


class PhysicsConstrainedAttention(nn.Module):
    """物理约束注意力机制
    
    通过物理定律约束注意力权重，确保注意力分布符合物理规律
    """
    
    def __init__(self, d_model: int, nhead: int, physics_weight: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.physics_weight = physics_weight
        self.physics_loss_fn = None  # 将通过hook设置
        
    def set_physics_loss_fn(self, loss_fn):
        """设置物理损失函数钩子"""
        self.physics_loss_fn = loss_fn
        
    def compute_causal_mask(self, seq_len: int, device: torch.device):
        """计算因果掩码，确保时间因果性"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
        
    def compute_locality_mask(self, seq_len: int, locality_radius: int, device: torch.device):
        """计算局部性掩码，鼓励局部注意力"""
        mask = torch.ones(seq_len, seq_len, device=device)
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > locality_radius:
                    mask[i, j] = 0
        return mask.bool()
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                physics_mask: Optional[torch.Tensor] = None, 
                enforce_causal: bool = True,
                enforce_locality: bool = True,
                locality_radius: int = 8):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            physics_mask: 物理约束掩码
            enforce_causal: 是否强制执行因果性
            enforce_locality: 是否强制执行局部性
            locality_radius: 局部性半径
        """
        batch_size, seq_len, _ = query.shape
        
        # 组合物理约束掩码
        combined_mask = None
        if enforce_causal:
            causal_mask = self.compute_causal_mask(seq_len, query.device)
            combined_mask = causal_mask
            
        if enforce_locality:
            locality_mask = self.compute_locality_mask(seq_len, locality_radius, query.device)
            if combined_mask is not None:
                combined_mask = combined_mask | locality_mask
            else:
                combined_mask = locality_mask
                
        if physics_mask is not None:
            if combined_mask is not None:
                combined_mask = combined_mask | physics_mask
            else:
                combined_mask = physics_mask
        
        # 应用注意力
        attn_output, attn_weights = self.attention(
            query, key, value, attn_mask=combined_mask
        )
        
        # 物理约束损失
        if self.training and physics_mask is not None and self.physics_loss_fn is not None:
            physics_loss = F.mse_loss(attn_weights, physics_mask)
            self.physics_loss_fn(physics_loss * self.physics_weight)
            
        return attn_output, attn_weights


class PhysicsTransformerEncoderLayer(nn.Module):
    """物理感知的Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, physics_weight: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        # 多头注意力（带物理约束）
        self.self_attn = PhysicsConstrainedAttention(
            d_model, nhead, physics_weight
        )
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = self._get_activation_fn(activation)
        
    def _get_activation_fn(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish":
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor, physics_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
            physics_mask: 物理约束掩码
            **kwargs: 其他参数（因果性、局部性等）
        """
        # 自注意力
        attn_output, _ = self.self_attn(
            x, x, x, physics_mask=physics_mask, **kwargs
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class PhysicsTransformerTemporal(BaseTemporalModel):
    """物理感知Transformer时序模型
    
    专门为PDE求解设计的Transformer架构，特点：
    1. 物理信息位置编码
    2. 多尺度时序注意力
    3. 物理约束机制
    4. 因果性保证
    5. 与空间模型解耦
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int,
        T_in: int = 1,
        T_out: int = 1,
        mode: str = 'nar',
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        physics_weight: float = 0.1,
        multi_scale: bool = True,
        scales: Optional[list] = None,
        max_seq_len: int = 1000,
        activation: str = "gelu",
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            T_in=T_in,
            T_out=T_out,
            mode=mode,
            **kwargs
        )
        
        self.d_model = d_model
        self.physics_weight = physics_weight
        self.multi_scale = multi_scale
        
        # 输入投影层
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 物理信息位置编码
        self.pos_encoding = PhysicsInformedPositionalEncoding(
            d_model, max_seq_len, dropout
        )
        
        # Transformer编码器层
        encoder_layers = []
        for i in range(num_layers):
            if multi_scale and i == 0:
                # 第一层使用多尺度注意力
                layer = nn.ModuleList([
                    PhysicsTransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout, physics_weight, activation
                    )
                    for _ in range(num_layers)
                ])
                # 使用多尺度注意力替换标准注意力
                for layer_module in layer:
                    layer_module.self_attn = MultiScaleAttention(d_model, nhead, scales)
                encoder_layers.extend(layer)
                break
            else:
                encoder_layers.append(
                    PhysicsTransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout, physics_weight, activation
                    )
                )
        
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, out_channels)
        
        # 物理损失钩子
        self._physics_losses = []
        self._register_physics_loss_hook()
        
        logger.info(f"PhysicsTransformerTemporal: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, physics_weight={physics_weight}")
        
    def _register_physics_loss_hook(self):
        """注册物理损失钩子"""
        def physics_loss_fn(loss):
            self._physics_losses.append(loss)
            
        # 为所有注意力层设置物理损失函数
        for layer in self.transformer_layers:
            if hasattr(layer.self_attn, 'set_physics_loss_fn'):
                layer.self_attn.set_physics_loss_fn(physics_loss_fn)
                
    def get_physics_loss(self) -> torch.Tensor:
        """获取累积的物理损失"""
        if not self._physics_losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(self._physics_losses).mean()
        
    def clear_physics_losses(self):
        """清空物理损失缓存"""
        self._physics_losses.clear()
        
    def forward(
        self,
        x: torch.Tensor,
        T_out: Optional[int] = None,
        teacher_forcing: Optional[torch.Tensor] = None,
        physics_info: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量 [B, T_in, C] 或 [B, T_in, C, H, W]
            T_out: 输出时间步数
            teacher_forcing: 教师信号
            physics_info: 物理信息 {'velocity': tensor, 'acceleration': tensor}
            return_dict: 是否返回字典格式
            
        Returns:
            输出张量 [B, T_out, C_out] 或字典格式结果
        """
        # 验证输入
        self.validate_input(x)
        
        # 清空之前的物理损失
        self.clear_physics_losses()
        
        # 处理输入维度
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            # 重塑为时序序列：每个空间位置独立处理
            x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
            spatial_shape = (H, W)
        else:  # [B, T, C]
            spatial_shape = None
            
        batch_size, seq_len, in_channels = x.shape
        
        # 输入投影
        x = self.input_proj(x)  # [B, T, d_model]
        
        # 位置编码（包含物理信息）
        x = x.transpose(0, 1)  # [T, B, d_model] - Transformer格式
        x = self.pos_encoding(x, physics_info)
        
        # 因果掩码
        causal_mask = self._create_causal_mask(seq_len, x.device)
        
        # 物理约束掩码（可选）
        physics_mask = None
        if physics_info is not None:
            physics_mask = self._create_physics_mask(seq_len, batch_size, x.device, physics_info)
        
        # Transformer层处理
        layer_outputs = []
        for i, layer in enumerate(self.transformer_layers):
            # 不同层使用不同的物理约束强度
            layer_physics_weight = self.physics_weight * (1.0 + 0.1 * i)  # 逐层增强
            
            if hasattr(layer.self_attn, 'physics_weight'):
                layer.self_attn.physics_weight = layer_physics_weight
                
            x = layer(x, physics_mask=physics_mask, enforce_causal=True)
            layer_outputs.append(x.clone())
            
        # 输出投影
        x = self.output_proj(x)  # [T, B, C_out]
        x = x.transpose(0, 1)    # [B, T, C_out]
        
        # 处理输出时间步
        if T_out is not None and T_out != seq_len:
            if T_out < seq_len:
                x = x[:, :T_out]
            else:
                # 需要外推，使用最后一步重复或更复杂的策略
                last_step = x[:, -1:]
                repeat_steps = T_out - seq_len
                x = torch.cat([x, last_step.repeat(1, repeat_steps, 1)], dim=1)
        
        # 恢复空间维度
        if spatial_shape is not None:
            H, W = spatial_shape
            x = x.reshape(batch_size, H, W, x.size(1), x.size(2))
            x = x.permute(0, 3, 4, 1, 2)  # [B, T_out, C_out, H, W]
            
        # 收集物理损失
        physics_loss = self.get_physics_loss()
        
        if return_dict:
            return {
                'output': x,
                'temporal_features': layer_outputs,
                'physics_loss': physics_loss,
                'model_info': self.get_model_info()
            }
        else:
            return x
            
    def _create_causal_mask(self, seq_len: int, device: torch.device):
        """创建因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
        
    def _create_physics_mask(self, seq_len: int, batch_size: int, device: torch.device,
                           physics_info: Dict[str, torch.Tensor]):
        """创建物理约束掩码"""
        # 基于物理信息创建注意力约束
        # 这里可以实现具体的物理约束逻辑
        # 例如：基于能量守恒、动量守恒等
        
        # 简单的局部性约束作为示例
        locality_mask = torch.ones(seq_len, seq_len, device=device)
        locality_radius = min(seq_len // 4, 16)  # 自适应局部性半径
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > locality_radius:
                    locality_mask[i, j] = 0.1  # 降低远距离注意力
                    
        return locality_mask.unsqueeze(0).repeat(batch_size, 1, 1)


# 辅助函数：创建物理感知Transformer时序模块
def create_physics_transformer_temporal(**kwargs) -> PhysicsTransformerTemporal:
    ""”创建物理感知Transformer时序模块的工厂函数"""
    return PhysicsTransformerTemporal(**kwargs)


if __name__ == "__main__":
    # 测试代码
    import torch
    
    print("🧪 测试物理感知Transformer时序模块...")
    
    # 创建模型
    model = PhysicsTransformerTemporal(
        in_channels=2,
        out_channels=2,
        img_size=128,
        T_in=10,
        T_out=10,
        d_model=256,
        nhead=8,
        num_layers=6,
        physics_weight=0.1,
        multi_scale=True
    )
    
    # 测试输入
    batch_size, T, C = 4, 10, 2
    x = torch.randn(batch_size, T, C)
    
    # 物理信息
    physics_info = {
        'velocity': torch.randn(batch_size, T, 256),
        'acceleration': torch.randn(batch_size, T, 256)
    }
    
    # 前向传播
    output = model(x, physics_info=physics_info, return_dict=True)
    
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output['output'].shape}")
    print(f"✅ 物理损失: {output['physics_loss'].item():.6f}")
    print(f"✅ 模型参数量: {model.get_model_info()['total_parameters']:,}")
    
    print("🎉 物理感知Transformer时序模块测试完成！")