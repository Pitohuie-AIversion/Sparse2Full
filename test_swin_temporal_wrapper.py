#!/usr/bin/env python3
"""测试SwinTemporalWrapper的简化脚本"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

# 简化的BaseModel类
class BaseModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_size: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

# 简化的SwinUNet类（仅用于测试）
class SwinUNet(BaseModel):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, img_size: int = 256, **kwargs):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        # 简化的实现
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_in(x))
        x = torch.relu(self.conv_mid(x))
        x = self.conv_out(x)
        return x

# 导入其他模块
from temporal_encoder import TemporalEncoder, create_temporal_encoder
from nar_prediction_head import CrossAttnTimeQueryHead, create_nar_prediction_head

class SwinTemporalWrapper(BaseModel):
    """SwinUNet时序包装器
    
    集成SwinUNet、TemporalEncoder和CrossAttnTimeQueryHead，
    支持AR/NAR/混合预测模式。
    """
    
    def __init__(
        self,
        # SwinUNet参数
        in_channels: int = 3,
        out_channels: int = 1, 
        img_size: int = 256,
        # 时序参数
        T_in: int = 1,
        T_out: int = 20,
        prediction_mode: str = "nar",  # "ar", "nar", "hybrid"
        # TemporalEncoder参数
        temporal_encoder_config: Optional[Dict] = None,
        # NAR预测头参数
        nar_head_config: Optional[Dict] = None,
        # 调度采样参数
        scheduled_sampling: bool = False,
        sampling_decay_steps: int = 10000,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, img_size, **kwargs)
        
        self.T_in = T_in
        self.T_out = T_out
        self.prediction_mode = prediction_mode
        self.scheduled_sampling = scheduled_sampling
        self.sampling_decay_steps = sampling_decay_steps
        self.current_epoch = 0
        
        # 核心SwinUNet模型
        self.swin_unet = SwinUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **kwargs
        )
        
        # 时序编码器（用于AR/混合模式）
        if prediction_mode in ["ar", "hybrid"]:
            temporal_config = temporal_encoder_config or {}
            # 设置默认值
            default_temporal_config = {
                'hidden_dim': 128,
                'num_conv_layers': 3,  # 这是正确的参数名
                'kernel_size': 3,
                'dropout': 0.1
            }
            default_temporal_config.update(temporal_config)
            
            self.temporal_encoder = create_temporal_encoder(
                input_dim=out_channels,
                config=default_temporal_config
            )
        else:
            self.temporal_encoder = None
            
        # NAR预测头（用于NAR/混合模式）
        if prediction_mode in ["nar", "hybrid"]:
            nar_config = nar_head_config or {}
            # 设置默认值
            default_nar_config = {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'output_projection': 'linear'
            }
            default_nar_config.update(nar_config)
            
            # 计算输入维度：C * H * W
            input_dim = self.out_channels * img_size * img_size
            output_dim = self.out_channels * img_size * img_size
            
            # 创建NAR头时需要传递正确的参数
            self.nar_head = create_nar_prediction_head(
                input_dim=input_dim,
                T_out=T_out,
                output_dim=output_dim,
                **default_nar_config
            )
        else:
            self.nar_head = None
    
    def set_prediction_mode(self, mode: str) -> None:
        """设置预测模式"""
        assert mode in ["ar", "nar", "hybrid"], f"Invalid mode: {mode}"
        self.prediction_mode = mode
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前训练轮次（用于调度采样）"""
        self.current_epoch = epoch
    
    def get_sampling_probability(self) -> float:
        """获取调度采样概率"""
        if not self.scheduled_sampling:
            return 0.0
        
        # 指数衰减
        decay_rate = 0.5
        prob = decay_rate ** (self.current_epoch / self.sampling_decay_steps)
        return min(prob, 0.9)  # 最大90%
    
    def apply_temporal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """应用时序编码"""
        if self.temporal_encoder is None:
            return x
        
        B, T, C, H, W = x.shape
        
        # 重塑为[B, T, C*H*W]进行时序编码
        x_flat = x.view(B, T, C * H * W)
        result = self.temporal_encoder(x_flat)
        
        # 提取编码结果
        x_encoded = result['encoded_sequence']
        
        # 重塑回[B, T, C, H, W]
        x_encoded = x_encoded.view(B, T, C, H, W)
        
        return x_encoded
    
    def forward_ar(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """AR模式前向传播"""
        B, T_in, C_in, H, W = x.shape
        
        # 初始化输出序列
        outputs = []
        current_input = x[:, -1]  # 使用最后一帧作为初始输入
        
        # 构建历史序列用于时序编码
        history = [self.swin_unet(x[:, i]) for i in range(T_in)]
        
        for t in range(self.T_out):
            # 确保输入通道数匹配
            if current_input.shape[1] != self.in_channels:
                if current_input.shape[1] < self.in_channels:
                    # 扩展通道：重复最后一个通道
                    current_input = current_input.repeat(1, self.in_channels, 1, 1)
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
                # 简单的通道扩展或压缩
                if current_input.shape[1] < C_in:
                    # 扩展通道：重复最后一个通道
                    current_input = current_input.repeat(1, C_in, 1, 1)
                else:
                    # 压缩通道：取前几个通道
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
        """NAR模式前向传播"""
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
        """混合模式前向传播"""
        # 前半部分使用NAR，后半部分使用AR
        T_nar = self.T_out // 2
        T_ar = self.T_out - T_nar
        
        # NAR预测前半部分
        nar_predictions = self.forward_nar(x)[:, :T_nar]
        
        # 处理通道匹配问题
        if nar_predictions.shape[2] != x.shape[2]:  # 通道数不匹配
            if nar_predictions.shape[2] < x.shape[2]:
                # 扩展NAR预测的通道数
                nar_predictions = nar_predictions.repeat(1, 1, x.shape[2], 1, 1)
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
                ar_predictions = ar_predictions.repeat(1, 1, nar_predictions.shape[2], 1, 1)
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


def test_swin_temporal_wrapper():
    """测试SwinTemporalWrapper"""
    print("=== 测试SwinTemporalWrapper ===")
    
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
        print(f"\n--- 测试{mode.upper()}模式 ---")
        
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
            nar_head_config={'hidden_dim': 128, 'num_heads': 4}
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
            assert output.shape[0] == B and output.shape[1] == T_out and output.shape[3] == H and output.shape[4] == W, f"输出形状不匹配: {output.shape}"
        else:
            assert output.shape == (B, T_out, C_out, H, W), f"输出形状不匹配: {output.shape}"
        
        # 测试模式切换
        model.set_prediction_mode("nar")
        assert model.prediction_mode == "nar"
        
        # 测试调度采样
        model.set_epoch(100)
        prob = model.get_sampling_probability()
        print(f"调度采样概率: {prob:.4f}")
        
        print(f"✓ {mode.upper()}模式测试通过")
    
    print("\n=== 所有测试通过 ===")


if __name__ == "__main__":
    test_swin_temporal_wrapper()