#!/usr/bin/env python3
"""
测试AR训练的DataParallel参数传递修复
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper

def test_dataparallel_fix():
    """测试DataParallel参数传递修复"""
    print("🧪 测试DataParallel参数传递修复...")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    if torch.cuda.device_count() < 2:
        print("❌ 需要至少2张GPU")
        return False
    
    print(f"✅ 检测到 {torch.cuda.device_count()} 张GPU")
    
    # 创建模型
    base_model = SwinUNet(
        in_channels=2,
        out_channels=2,
        img_size=128,
        patch_size=4,
        window_size=8,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        embed_dim=96
    )
    
    model = ARWrapper(base_model)
    model = model.cuda()
    
    # 创建DataParallel包装器（使用修复后的版本）
    class DataParallelWrapper(nn.DataParallel):
        def forward(self, x_in, T_out=1, teacher=None, train_mode=None):
            # 使用位置参数传递，避免DataParallel的keyword参数问题
            return super().forward(x_in, T_out, teacher, train_mode)
    
    model = DataParallelWrapper(model, device_ids=[0, 1])
    
    # 创建测试数据
    batch_size = 4
    x_in = torch.randn(batch_size, 1, 2, 128, 128).cuda()
    T_out = 5
    teacher = torch.randn(batch_size, T_out, 2, 128, 128).cuda()
    
    print(f"📊 输入形状: {x_in.shape}")
    print(f"📊 输出时间步: {T_out}")
    print(f"📊 教师信号形状: {teacher.shape}")
    
    try:
        # 测试训练模式（使用位置参数）
        model.train()
        print("🔄 测试训练模式...")
        pred_train = model(x_in, T_out, teacher)
        print(f"✅ 训练模式输出形状: {pred_train.shape}")
        
        # 测试推理模式（使用位置参数）
        model.eval()
        print("🔄 测试推理模式...")
        with torch.no_grad():
            pred_eval = model(x_in, T_out)
        print(f"✅ 推理模式输出形状: {pred_eval.shape}")
        
        print("🎉 DataParallel参数传递修复成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataparallel_fix()
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)