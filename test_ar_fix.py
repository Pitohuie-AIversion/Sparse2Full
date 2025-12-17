#!/usr/bin/env python3
"""
测试AR训练修复后的多GPU问题
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper

def test_ar_wrapper_forward():
    """测试ARWrapper的forward方法调用"""
    print("🧪 测试ARWrapper forward方法...")
    
    # 创建模型
    base_model = SwinUNet(
        in_channels=2,
        out_channels=2,
        img_size=128,
        patch_size=4,
        window_size=8,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    )
    
    ar_model = ARWrapper(
        single_frame_model=base_model,
        detach_rollout=True,
        scheduled_sampling=False
    )
    
    # 测试单GPU
    print("📱 测试单GPU...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ar_model = ar_model.to(device)
    
    # 创建测试数据
    batch_size = 2
    input_seq = torch.randn(batch_size, 1, 2, 128, 128).to(device)
    target_seq = torch.randn(batch_size, 5, 2, 128, 128).to(device)
    
    # 测试不同的调用方式
    print("  测试位置参数调用...")
    try:
        pred1 = ar_model(input_seq, T_out=5, teacher=target_seq)
        print(f"  ✅ 位置参数调用成功: {pred1.shape}")
    except Exception as e:
        print(f"  ❌ 位置参数调用失败: {e}")
    
    print("  测试关键字参数调用...")
    try:
        pred2 = ar_model(x_in=input_seq, T_out=5, teacher=target_seq)
        print(f"  ✅ 关键字参数调用成功: {pred2.shape}")
    except Exception as e:
        print(f"  ❌ 关键字参数调用失败: {e}")
    
    # 测试多GPU
    if torch.cuda.device_count() > 1:
        print("🔄 测试多GPU DataParallel...")
        ar_model_dp = nn.DataParallel(ar_model, device_ids=list(range(torch.cuda.device_count())))
        
        print("  测试位置参数调用...")
        try:
            pred3 = ar_model_dp(input_seq, T_out=5, teacher=target_seq)
            print(f"  ✅ DataParallel位置参数调用成功: {pred3.shape}")
        except Exception as e:
            print(f"  ❌ DataParallel位置参数调用失败: {e}")
        
        print("  测试关键字参数调用...")
        try:
            pred4 = ar_model_dp(x_in=input_seq, T_out=5, teacher=target_seq)
            print(f"  ✅ DataParallel关键字参数调用成功: {pred4.shape}")
        except Exception as e:
            print(f"  ❌ DataParallel关键字参数调用失败: {e}")
    else:
        print("⚠️ 只有一张GPU，跳过多GPU测试")

def test_memory_usage():
    """测试内存使用情况"""
    print("\n💾 测试内存使用情况...")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"  GPU {i}: {memory_allocated:.2f}GB / {memory_total:.2f}GB allocated")
            print(f"         {memory_reserved:.2f}GB / {memory_total:.2f}GB reserved")

if __name__ == "__main__":
    print("🚀 开始AR训练修复测试...")
    test_ar_wrapper_forward()
    test_memory_usage()
    print("✅ 测试完成！")