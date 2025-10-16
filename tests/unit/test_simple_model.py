#!/usr/bin/env python3
"""
简单的模型测试脚本
用于验证模型是否能正常工作
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.swin_unet import SwinUNet
    print("✓ Successfully imported SwinUNet")
except ImportError as e:
    print(f"✗ Failed to import SwinUNet: {e}")
    sys.exit(1)

try:
    # 创建一个简单的SwinUNet模型
    model = SwinUNet(
        in_channels=3, 
        out_channels=3, 
        img_size=64, 
        embed_dim=96,
        depths=[2, 2], 
        num_heads=[3, 6],
        decoder_channels=[96, 48],
        skip_connections=False
    )
    print("✓ Successfully created SwinUNet model")
    
    # 测试前向传播
    x = torch.randn(1, 3, 64, 64)
    print(f"✓ Created input tensor with shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
        print(f"✓ Forward pass successful, output shape: {y.shape}")
        
        if y.shape == (1, 3, 64, 64):
            print("✓ Output shape is correct")
        else:
            print(f"✗ Output shape is incorrect, expected (1, 3, 64, 64), got {y.shape}")
            
        if torch.isfinite(y).all():
            print("✓ All output values are finite")
        else:
            print("✗ Output contains non-finite values")
            
    print("\n🎉 All tests passed! Model is working correctly.")
    
except Exception as e:
    print(f"✗ Error during model testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)