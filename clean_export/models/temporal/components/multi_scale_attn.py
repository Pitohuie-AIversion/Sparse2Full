"""
多尺度时序注意力机制
专门为PDE时序预测设计的多尺度注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from einops import rearrange


class MultiScaleTemporalAttention(nn.Module):
    """多尺度时序注意力机制
    
    特点：
    1. 多头部设计，每个头部处理不同时间尺度
    2. 物理约束掩码，确保注意力符合物理因果性
    3. 频率域注意力，捕获周期性模式
    4. 自适应尺度选择
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        scales: List[int] = [1, 2, 4, 8],  # 不同时间尺度
        use_physical_mask: bool = True,
        use_frequency_attn: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scales = scales
        self.use_physical_mask = use_physical_mask
        self.use_frequency_attn = use_frequency_attn
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 多尺度投影
        self.scale_projections = nn.ModuleDict()
        for scale in scales:
            self.scale_projections[f'scale_{scale}'] = nn.Linear(hidden_dim, hidden_dim)
        
        # 注意力参数
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 物理约束参数
        if use_physical_mask:
            self.physical_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
            self.causal_mask = None
        
        # 频率域注意力
        if use_frequency_attn:
            self.freq_proj = nn.Linear(hidden_dim, hidden_dim)
            self.freq_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # 自适应尺度选择
        self.scale_selector = nn.Sequential(
            nn.Linear(hidden_dim, len(scales)),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码，确保时间因果性"""
        if self.causal_mask is None or self.causal_mask.size(-1) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.bool()
            self.causal_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return self.causal_mask
    
    def get_physical_mask(self, seq_len: int, device: torch.device, 
                         physical_info: Optional[Dict] = None) -> torch.Tensor:
        """生成物理约束掩码
        
        Args:
            seq_len: 序列长度
            device: 计算设备
            physical_info: 物理信息字典，包含速度、加速度等
            
        Returns:
            物理约束掩码 [1, num_heads, seq_len, seq_len]
        """
        if not self.use_physical_mask:
            return torch.zeros(1, self.num_heads, seq_len, seq_len, device=device)
        
        # 基础因果掩码
        causal_mask = self.get_causal_mask(seq_len, device)
        
        # 物理约束增强
        physical_mask = torch.zeros(seq_len, seq_len, device=device)
        
        if physical_info is not None:
            # 基于物理速度的时间窗约束
            if 'velocity' in physical_info:
                velocity = physical_info['velocity']  # [B, T, C]
                vel_magnitude = torch.norm(velocity, dim=-1).mean(dim=0)  # [T]
                
                # 速度越大，时间相关性衰减越快
                decay_rate = torch.sigmoid(vel_magnitude * 2)  # [T]
                for i in range(seq_len):
                    for j in range(i+1, seq_len):
                        time_diff = j - i
                        physical_mask[i, j] = -decay_rate[j] * time_diff * 0.1
        
        # 组合掩码
        combined_mask = causal_mask.squeeze(0).squeeze(0).float() + physical_mask
        return combined_mask.unsqueeze(0).unsqueeze(0).expand(1, self.num_heads, -1, -1)
    
    def multi_scale_processing(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """多尺度处理"""
        B, T, C = x.shape
        
        if scale == 1:
            return x
        
        # 下采样
        if T % scale != 0:
            # 填充到可被scale整除
            pad_len = scale - (T % scale)
            x_pad = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            T_pad = T + pad_len
        else:
            x_pad, T_pad = x, T
        
        # 平均池化到下采样尺度
        x_scaled = F.avg_pool1d(
            x_pad.transpose(1, 2), 
            kernel_size=scale, 
            stride=scale
        ).transpose(1, 2)
        
        return x_scaled
    
    def frequency_domain_attention(self, x: torch.Tensor) -> torch.Tensor:
        """频率域注意力，捕获周期性模式"""
        if not self.use_frequency_attn:
            return x
        
        B, T, C = x.shape
        
        # FFT变换到频率域
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag
        
        # 确保频率域特征维度正确
        freq_T = x_freq_real.shape[1]  # 频率域的时间维度
        
        # 创建频率域的投影层（如果维度不匹配）
        if freq_T != T:
            # 创建适配的投影层
            freq_proj = nn.Linear(C, C).to(x.device)
            freq_attention = nn.MultiheadAttention(
                C, self.num_heads, dropout=self.dropout.p, batch_first=True
            ).to(x.device)
            freq_norm = nn.LayerNorm(C).to(x.device)
        else:
            freq_proj = self.freq_proj
            freq_attention = self.freq_attention
            freq_norm = self.layer_norm
        
        # 分别处理实部和虚部
        freq_features_real = freq_proj(x_freq_real)
        freq_features_imag = freq_proj(x_freq_imag)
        
        # 注意力机制（实部）
        freq_features_real_norm = freq_norm(freq_features_real)
        attn_out_real, _ = freq_attention(
            freq_features_real_norm, freq_features_real_norm, freq_features_real_norm
        )
        
        # 注意力机制（虚部）
        freq_features_imag_norm = freq_norm(freq_features_imag)
        attn_out_imag, _ = freq_attention(
            freq_features_imag_norm, freq_features_imag_norm, freq_features_imag_norm
        )
        
        # 转换回时域
        freq_out_real = freq_features_real + self.dropout(attn_out_real)
        freq_out_imag = freq_features_imag + self.dropout(attn_out_imag)
        
        x_freq_enhanced = torch.complex(freq_out_real, freq_out_imag)
        x_time = torch.fft.irfft(x_freq_enhanced, n=T, dim=1, norm='ortho')
        
        return x_time
    
    def forward(
        self, 
        x: torch.Tensor,
        physical_info: Optional[Dict] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入序列 [B, T, C]
            physical_info: 物理信息字典
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            输出序列和注意力权重（可选）
        """
        B, T, C = x.shape
        
        # 自适应尺度选择
        scale_weights = self.scale_selector(x.mean(dim=1))  # [B, num_scales]
        
        # 多尺度处理
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            # 多尺度投影
            x_scaled = self.multi_scale_processing(x, scale)
            x_proj = self.scale_projections[f'scale_{scale}'](x_scaled)
            
            # 上采样回原始尺度
            if scale != 1:
                x_proj = F.interpolate(
                    x_proj.transpose(1, 2), 
                    size=T, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            multi_scale_features.append(x_proj * scale_weights[:, i:i+1, None])
        
        # 融合多尺度特征
        x_multi = sum(multi_scale_features)
        
        # QKV投影
        qkv = self.qkv_proj(x_multi)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # 多头分割
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # 注意力计算
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        
        # 应用物理约束掩码
        if self.use_physical_mask:
            physical_mask = self.get_physical_mask(T, x.device, physical_info)
            attention_scores = attention_scores + physical_mask
        
        # 应用因果掩码
        causal_mask = self.get_causal_mask(T, x.device)
        attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 注意力输出
        attn_out = torch.matmul(attention_weights, v)
        attn_out = rearrange(attn_out, 'b h t d -> b t (h d)')
        
        # 输出投影
        out = self.out_proj(attn_out)
        
        # 频率域增强
        out_freq = self.frequency_domain_attention(out)
        
        # 残差连接和层归一化
        output = self.layer_norm(x + self.dropout(out_freq))
        
        if return_attention_weights:
            return output, attention_weights
        
        return output, None


class AdaptiveTemporalMixer(nn.Module):
    """自适应时序混合器
    
    结合不同时间尺度的特征，自适应地选择最优混合策略
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # 尺度混合权重
        self.scale_mixer = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            multi_scale_features: 多尺度特征列表，每个元素 [B, T, C]
            
        Returns:
            融合后的特征 [B, T, C]
        """
        # 拼接多尺度特征
        concat_features = torch.cat(multi_scale_features, dim=-1)  # [B, T, C*num_scales]
        
        # 计算尺度混合权重
        scale_weights = self.scale_mixer(concat_features)  # [B, T, num_scales]
        
        # 加权融合
        weighted_features = []
        for i, features in enumerate(multi_scale_features):
            weight = scale_weights[:, :, i:i+1]  # [B, T, 1]
            weighted_features.append(features * weight)
        
        # 特征融合
        fused_features = torch.cat(weighted_features, dim=-1)
        output = self.feature_fusion(fused_features)
        
        # 残差连接和归一化
        base_feature = multi_scale_features[0]  # 使用最低尺度作为基准
        output = self.layer_norm(base_feature + output)
        
        return output


class PhysicsAwareAttention(nn.Module):
    """物理感知注意力机制
    
    结合物理方程和约束条件，增强注意力机制
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pde_constraint_weight: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pde_constraint_weight = pde_constraint_weight
        
        # 基础注意力
        self.attention = MultiScaleTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_physical_mask=True,
            use_frequency_attn=True
        )
        
        # PDE约束网络
        self.pde_constraint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def compute_pde_residual(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """计算简化的PDE残差（专注于时间平滑性）"""
        if x.size(1) >= 3:
            # 时间导数（中心差分）
            x_t = (x[:, 2:] - x[:, :-2]) / (2 * dt)
            
            # 简化的时间平滑性约束
            # 理想情况下，时间导数应该相对平滑
            target_smoothness = torch.zeros_like(x_t)
            pde_residual = x_t - target_smoothness
            
            # 填充以匹配原始维度
            pde_residual = F.pad(pde_residual, (0, 0, 1, 1), mode='replicate')
        else:
            pde_residual = torch.zeros_like(x)
        
        return pde_residual
    
    def forward(
        self, 
        x: torch.Tensor,
        physical_info: Optional[Dict] = None,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Returns:
            输出特征和约束损失字典
        """
        # 基础注意力
        attn_out, attention_weights = self.attention(
            x, physical_info, return_attention_weights=True
        )
        
        # PDE约束
        pde_residual = self.compute_pde_residual(x, dt)
        pde_loss = torch.mean(pde_residual ** 2)
        
        # 约束损失
        constraint_loss = self.pde_constraint_weight * pde_loss
        
        # 组合输出
        output = attn_out
        
        loss_dict = {
            'pde_residual': pde_residual,
            'pde_loss': pde_loss,
            'constraint_loss': constraint_loss,
            'attention_weights': attention_weights
        }
        
        return output, loss_dict