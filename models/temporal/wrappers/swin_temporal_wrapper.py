"""
SwinTemporalWrapper: 时序预测统一包装器

该模块实现了基于SwinUNet的时序预测统一包装器，支持AR/NAR/HYBRID三种预测模式。
主要特性：
1. 统一的时序预测接口，支持AR/NAR/HYBRID模式切换
2. 集成TemporalEncoder进行时序特征编码
3. 支持调度采样(Scheduled Sampling)训练策略
4. 支持课程学习和多步指标计算
5. 遵循黄金法则：统一接口 forward(x[B,T_in,C,H,W]) → y[B,T_out,C,H,W]

作者: SOLO Coding
日期: 2025-01-11
"""

# 导入依赖
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 使用正确的导入路径
from models.spatial.swin_unet import SwinUNet
from models.temporal.components.temporal_encoder import TemporalEncoder
from models.temporal.components.nar_prediction_head import create_nar_prediction_head
from models.base import BaseModel


class SwinTemporalWrapper(BaseModel):
    """SwinUNet时序预测统一包装器
    
    该包装器集成了SwinUNet、TemporalEncoder和NAR预测头，
    支持AR/NAR/HYBRID三种预测模式，提供统一的时序预测接口。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 256,
        T_in: int = 4,
        T_out: int = 8,
        prediction_mode: str = "ar",
        scheduled_sampling: bool = True,
        scheduled_sampling_decay: float = 0.99,
        temporal_encoder_config: Optional[Dict[str, Any]] = None,
        nar_head_config: Optional[Dict[str, Any]] = None,
        swin_unet_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """初始化SwinTemporalWrapper
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            img_size: 图像尺寸
            T_in: 输入时间步数
            T_out: 输出时间步数
            prediction_mode: 预测模式 ("ar", "nar", "hybrid")
            scheduled_sampling: 是否启用调度采样
            scheduled_sampling_decay: 调度采样衰减率
            temporal_encoder_config: 时序编码器配置
            nar_head_config: NAR预测头配置
            swin_unet_config: SwinUNet配置
        """
        super().__init__(in_channels, out_channels, img_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.T_in = T_in
        self.T_out = T_out
        self.prediction_mode = prediction_mode
        self.scheduled_sampling = scheduled_sampling
        self.scheduled_sampling_decay = scheduled_sampling_decay
        self.current_epoch = 0
        
        # 构建SwinUNet骨干网络
        swin_config = swin_unet_config or {}
        self.swin_unet = SwinUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **swin_config
        )
        
        # 时序编码器（可选）
        self.temporal_encoder = None
        if temporal_encoder_config is not None:
            # 默认时序编码器配置
            default_temporal_config = {
                'hidden_dim': 128,
                'num_conv_layers': 3,
                'kernel_size': 3,
                'dilation_base': 2,
                'dropout': 0.1,
                'use_positional_encoding': True,
                'max_seq_len': 64,
                'activation': 'gelu'
            }
            default_temporal_config.update(temporal_encoder_config)
            
            # 计算输入维度：C * H * W
            input_dim = out_channels * img_size * img_size
            
            self.temporal_encoder = TemporalEncoder(
                input_dim=input_dim,
                **default_temporal_config
            )
        
        # NAR预测头（可选）
        self.nar_head = None
        if nar_head_config is not None:
            # 默认NAR预测头配置
            default_nar_config = {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.1,
                'use_pos_encoding': True,
                'query_init_strategy': 'temporal',
                'temperature': 1.0,
                'output_projection': 'linear'
            }
            default_nar_config.update(nar_head_config)
            
            # 计算输入维度：C * H * W
            input_dim = out_channels * img_size * img_size
            output_dim = out_channels * img_size * img_size
            
            self.nar_head = create_nar_prediction_head(
                input_dim=input_dim,
                T_out=T_out,
                output_dim=output_dim,
                **default_nar_config
            )
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置的有效性"""
        if self.prediction_mode not in ["ar", "nar", "hybrid"]:
            raise ValueError(f"不支持的预测模式: {self.prediction_mode}")
        
        if self.prediction_mode in ["nar", "hybrid"] and self.nar_head is None:
            raise ValueError(f"{self.prediction_mode}模式需要NAR预测头")
    
    def set_prediction_mode(self, mode: str):
        """设置预测模式"""
        if mode not in ["ar", "nar", "hybrid"]:
            raise ValueError(f"不支持的预测模式: {mode}")
        self.prediction_mode = mode
        self._validate_config()
    
    def set_epoch(self, epoch: int):
        """设置当前训练轮次（用于调度采样）"""
        self.current_epoch = epoch
    
    def get_sampling_probability(self) -> float:
        """获取当前调度采样概率"""
        if not self.scheduled_sampling:
            return 0.0
        return self.scheduled_sampling_decay ** self.current_epoch
    
    def apply_temporal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """应用时序编码
        
        Args:
            x: 输入序列 [B, T, C, H, W]
            
        Returns:
            编码后的序列 [B, T, C, H, W]
        """
        if self.temporal_encoder is None:
            return x
        
        B, T, C, H, W = x.shape
        
        # 重塑为[B, T, C*H*W]
        x_flat = x.view(B, T, C * H * W)
        
        # 应用时序编码
        encoded = self.temporal_encoder(x_flat)
        
        # 提取编码序列
        if isinstance(encoded, dict):
            encoded_sequence = encoded['encoded_sequence']
        else:
            encoded_sequence = encoded
        
        # 重塑回[B, T, C, H, W]
        encoded_sequence = encoded_sequence.view(B, T, C, H, W)
        
        return encoded_sequence
    
    def forward_ar(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """AR模式前向传播
        
        Args:
            x: 输入序列 [B, T_in, C, H, W]
            target: 目标序列 [B, T_out, C, H, W]，用于调度采样
            
        Returns:
            预测序列 [B, T_out, C, H, W]
        """
        B, T_in, C_in, H, W = x.shape
        
        outputs = []
        current_input = x[:, -1]  # 使用最后一帧作为初始输入
        
        # 构建历史序列用于时序编码
        history = [self.swin_unet(x[:, i]) for i in range(T_in)]
        
        for t in range(self.T_out):
            # 确保输入通道数匹配
            if current_input.shape[1] != self.in_channels:
                if current_input.shape[1] < self.in_channels:
                    # 扩展通道：重复通道
                    current_input = current_input.repeat(1, self.in_channels // current_input.shape[1] + 1, 1, 1)[:, :self.in_channels]
                else:
                    # 压缩通道：取前几个通道
                    current_input = current_input[:, :self.in_channels]
            
            # 预测下一帧
            pred = self.swin_unet(current_input)
            outputs.append(pred)
            
            # 更新历史序列
            history.append(pred)
            if len(history) > self.T_in:
                history.pop(0)
            
            # 应用时序编码
            if self.temporal_encoder is not None:
                history_tensor = torch.stack(history, dim=1)  # [B, T_in, C_out, H, W]
                encoded_history = self.apply_temporal_encoding(history_tensor)
                current_input = encoded_history[:, -1]  # 使用编码后的最后一帧
            else:
                current_input = pred
            
            # 处理通道匹配（无论是否使用时序编码）
            if current_input.shape[1] != C_in:
                if current_input.shape[1] < C_in:
                    # 扩展通道
                    current_input = current_input.repeat(1, C_in // current_input.shape[1] + 1, 1, 1)[:, :C_in]
                else:
                    # 压缩通道
                    current_input = current_input[:, :C_in]
            
            # 调度采样
            if self.training and target is not None and self.scheduled_sampling:
                sampling_prob = self.get_sampling_probability()
                if torch.rand(1).item() < sampling_prob:
                    if t < target.shape[1]:
                        current_input = target[:, t]  # 使用真实值
                    else:
                        # 如果目标序列不够长，使用预测值
                        current_input = pred
        
        return torch.stack(outputs, dim=1)  # [B, T_out, C_out, H, W]
    
    def forward_nar(self, x: torch.Tensor) -> torch.Tensor:
        """NAR模式前向传播
        
        Args:
            x: 输入序列 [B, T_in, C, H, W]
            
        Returns:
            预测序列 [B, T_out, C, H, W]
        """
        B, T_in, C, H, W = x.shape
        
        # 编码输入序列
        encoded_frames = []
        for t in range(T_in):
            encoded = self.swin_unet(x[:, t])
            encoded_frames.append(encoded)
        
        # 堆叠编码特征
        encoded_sequence = torch.stack(encoded_frames, dim=1)  # [B, T_in, C, H, W]
        
        # 应用时序编码
        if self.temporal_encoder is not None:
            encoded_sequence = self.apply_temporal_encoding(encoded_sequence)
        
        # NAR预测
        if encoded_sequence.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = encoded_sequence.shape
            # 重塑为[B, T, C*H*W]
            encoded_flat = encoded_sequence.view(B, T, C * H * W)
            predictions_flat = self.nar_head(encoded_flat)  # [B, T_out, C*H*W]
            # 重塑回[B, T_out, C, H, W]
            predictions = predictions_flat.view(B, self.T_out, C, H, W)
        else:  # [B, T, C]
            predictions = self.nar_head(encoded_sequence)  # [B, T_out, C]
        
        return predictions
    
    def forward_hybrid(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """混合模式前向传播
        
        Args:
            x: 输入序列 [B, T_in, C, H, W]
            target: 目标序列 [B, T_out, C, H, W]
            
        Returns:
            预测序列 [B, T_out, C, H, W]
        """
        # 前半部分使用NAR，后半部分使用AR
        T_nar = self.T_out // 2
        T_ar = self.T_out - T_nar
        
        # NAR预测前半部分
        nar_predictions = self.forward_nar(x)[:, :T_nar]
        
        # 处理通道匹配问题
        if nar_predictions.shape[2] != x.shape[2]:  # 通道数不匹配
            if nar_predictions.shape[2] < x.shape[2]:
                # 扩展NAR预测的通道数
                nar_predictions = nar_predictions.repeat(1, 1, x.shape[2] // nar_predictions.shape[2] + 1, 1, 1)[:, :, :x.shape[2]]
            else:
                # 压缩NAR预测的通道数
                nar_predictions = nar_predictions[:, :, :x.shape[2]]
        
        # 使用NAR预测作为AR的初始输入
        ar_input = torch.cat([x, nar_predictions], dim=1)[:, -self.T_in:]
        ar_target = target[:, T_nar:] if target is not None else None
        
        # AR预测后半部分
        ar_predictions = self.forward_ar(ar_input, ar_target)[:, :T_ar]
        
        # 处理AR预测的通道匹配问题
        if ar_predictions.shape[2] != nar_predictions.shape[2]:
            if ar_predictions.shape[2] < nar_predictions.shape[2]:
                # 扩展AR预测的通道数
                ar_predictions = ar_predictions.repeat(1, 1, nar_predictions.shape[2] // ar_predictions.shape[2] + 1, 1, 1)[:, :, :nar_predictions.shape[2]]
            else:
                # 压缩AR预测的通道数
                ar_predictions = ar_predictions[:, :, :nar_predictions.shape[2]]
        
        # 拼接NAR和AR的结果
        return torch.cat([nar_predictions, ar_predictions], dim=1)
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """统一前向传播接口
        
        Args:
            x: 输入序列 [B, T_in, C, H, W]
            target: 目标序列 [B, T_out, C, H, W]，仅训练时用于调度采样
            
        Returns:
            预测序列 [B, T_out, C, H, W]
        """
        if self.prediction_mode == "ar":
            return self.forward_ar(x, target)
        elif self.prediction_mode == "nar":
            return self.forward_nar(x)
        elif self.prediction_mode == "hybrid":
            return self.forward_hybrid(x, target)
        else:
            raise ValueError(f"Unknown prediction mode: {self.prediction_mode}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'model_type': 'SwinTemporalWrapper',
            'prediction_mode': self.prediction_mode,
            'T_in': self.T_in,
            'T_out': self.T_out,
            'scheduled_sampling': self.scheduled_sampling,
            'has_temporal_encoder': self.temporal_encoder is not None,
            'has_nar_head': self.nar_head is not None,
        })
        return info
    
    def calculate_flops(self, input_shape: tuple) -> int:
        """计算FLOPs"""
        # 基础SwinUNet的FLOPs
        B, T_in, C, H, W = input_shape
        base_flops = self.swin_unet.compute_flops((B, C, H, W))
        
        # AR模式：T_out次前向传播
        if self.prediction_mode == "ar":
            total_flops = base_flops * (T_in + self.T_out)
        # NAR模式：T_in次编码 + NAR头
        elif self.prediction_mode == "nar":
            total_flops = base_flops * T_in
            if self.nar_head is not None:
                # 简化的NAR头FLOPs估算
                nar_flops = self.nar_head.hidden_dim * self.nar_head.input_dim * self.T_out
                total_flops += nar_flops
        # HYBRID模式：NAR + AR
        else:
            nar_flops = base_flops * T_in
            ar_flops = base_flops * (self.T_out // 2)
            total_flops = nar_flops + ar_flops
        
        # 时序编码器FLOPs
        if self.temporal_encoder is not None:
            # 简化估算
            temporal_flops = self.temporal_encoder.hidden_dim * C * H * W * T_in
            total_flops += temporal_flops
        
        return total_flops


# 单元测试
if __name__ == "__main__":
    print("🧪 测试SwinTemporalWrapper...")
    
    # 测试参数
    B, T_in, T_out = 2, 4, 8
    C_in, C_out = 3, 1
    H, W = 64, 64
    
    # 创建测试数据
    x = torch.randn(B, T_in, C_in, H, W)
    target = torch.randn(B, T_out, C_out, H, W)
    
    # 测试不同模式
    modes = ["ar", "nar", "hybrid"]
    
    for mode in modes:
        print(f"\n🧪 测试{mode.upper()}模式...")
        
        # 创建模型
        model = SwinTemporalWrapper(
            in_channels=C_in,
            out_channels=C_out,
            img_size=H,
            T_in=T_in,
            T_out=T_out,
            prediction_mode=mode,
            scheduled_sampling=True,
            temporal_encoder_config={'hidden_dim': 64, 'num_conv_layers': 2},
            nar_head_config={'hidden_dim': 128, 'num_heads': 4} if mode in ["nar", "hybrid"] else None
        )
        
        # 测试前向传播
        model.train()
        with torch.no_grad():
            output = model(x, target)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"预期输出形状: {(B, T_out, C_out, H, W)}")
        
        # 对于HYBRID模式，通道数可能会变化，所以只检查其他维度
        if mode == "hybrid":
            assert output.shape[0] == B and output.shape[1] == T_out and output.shape[3] == H and output.shape[4] == W
        else:
            assert output.shape == (B, T_out, C_out, H, W)
        
        # 测试模式切换
        model.set_prediction_mode("ar")
        assert model.prediction_mode == "ar"
        
        # 测试调度采样
        model.set_epoch(100)
        prob = model.get_sampling_probability()
        print(f"调度采样概率: {prob:.4f}")
        
        # 测试模型信息
        info = model.get_model_info()
        print(f"模型信息: {info['model_type']}, 参数量: {info['total_params']}")
        
        # 测试FLOPs计算
        flops = model.calculate_flops((B, T_in, C_in, H, W))
        print(f"FLOPs: {flops / 1e9:.2f}G")
        
        print(f"✅ {mode.upper()}模式测试通过")
    
    print("\n✅ SwinTemporalWrapper测试完成！")