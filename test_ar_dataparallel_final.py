#!/usr/bin/env python3
"""
最终测试AR DataParallel修复
验证多GPU参数传递是否正常工作
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper

class DataParallelWrapper(nn.DataParallel):
    """DataParallel包装器 - 使用位置参数"""
    def forward(self, x_in, T_out=1, teacher=None, train_mode=None):
        # 使用位置参数调用父类forward
        return super().forward(x_in, T_out, teacher, train_mode)

def test_ar_dataparallel():
    """测试AR DataParallel功能"""
    print("🧪 测试AR DataParallel修复...")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 检测到 {gpu_count} 张GPU")
    
    if gpu_count < 2:
        print("⚠️  需要至少2张GPU进行DataParallel测试")
        return False
    
    # 创建模型
    base_model = SwinUNet(
        in_channels=1,
        out_channels=1,
        img_size=128,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    # AR包装
    ar_model = ARWrapper(
        model=base_model,
        T_in=1,
        T_out=5,
        teacher_forcing_ratio=0.5
    )
    
    # DataParallel包装
    model = DataParallelWrapper(ar_model)
    model = model.cuda()
    
    print("✅ 模型创建成功")
    
    # 测试数据
    batch_size = 4
    T_in = 1
    T_out = 5
    H, W = 128, 128
    
    # 输入数据
    x_in = torch.randn(batch_size, T_in, 1, H, W).cuda()
    teacher = torch.randn(batch_size, T_out, 1, H, W).cuda()
    
    print(f"📊 输入形状: {x_in.shape}")
    print(f"📊 教师信号形状: {teacher.shape}")
    
    # 测试训练模式（使用位置参数）
    try:
        model.train()
        with torch.no_grad():
            output_train = model(x_in, T_out, teacher, True)
        print(f"✅ 训练模式输出形状: {output_train.shape}")
        
        # 测试推理模式（使用位置参数）
        model.eval()
        with torch.no_grad():
            output_eval = model(x_in, T_out)
        print(f"✅ 推理模式输出形状: {output_eval.shape}")
        
        # 验证输出形状
        expected_shape = (batch_size, T_out, 1, H, W)
        if output_train.shape == expected_shape and output_eval.shape == expected_shape:
            print("✅ DataParallel参数传递测试通过！")
            return True
        else:
            print(f"❌ 输出形状不匹配，期望: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ DataParallel测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ar_dataparallel()
    if success:
        print("\n🎉 所有测试通过！DataParallel修复成功！")
        sys.exit(0)
    else:
        print("\n💥 测试失败！")
        sys.exit(1)