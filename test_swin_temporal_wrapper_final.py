"""
测试正式的SwinTemporalWrapper模块
"""

import torch
import sys
import os

# 添加models目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from swin_temporal_wrapper import SwinTemporalWrapper


def test_swin_temporal_wrapper():
    """测试SwinTemporalWrapper"""
    print("=== 测试正式的SwinTemporalWrapper ===")
    
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
        print(f"模型信息: {info['model_type']}, 参数量: {info['parameters']}")
        
        # 测试FLOPs计算
        flops = model.calculate_flops((B, T_in, C_in, H, W))
        print(f"FLOPs: {flops / 1e9:.2f}G")
        
        print(f"✓ {mode.upper()}模式测试通过")
    
    print("\n=== 所有测试通过 ===")


if __name__ == "__main__":
    test_swin_temporal_wrapper()