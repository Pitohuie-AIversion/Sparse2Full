#!/usr/bin/env python3
"""
测试AR训练的多GPU修复
"""

import torch
import torch.nn as nn
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper

def test_ar_wrapper_multi_gpu():
    """测试ARWrapper在多GPU环境下的参数传递"""
    print("🔧 测试AR训练多GPU修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    if torch.cuda.device_count() < 2:
        print("❌ 需要至少2张GPU")
        return False
    
    try:
        # 创建基础模型
        base_model = SwinUNet(
            in_channels=2,
            out_channels=2,
            img_size=256,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
        
        # 包装为AR模型
        model = ARWrapper(
            single_frame_model=base_model,
            detach_rollout=True,
            scheduled_sampling=False
        )
        
        model = model.to('cuda:0')
        
        # 创建自定义DataParallel包装器
        class DataParallelWrapper(nn.DataParallel):
            def forward(self, x_in, T_out=1, teacher=None, train_mode=None):
                return super().forward(x_in, T_out=T_out, teacher=teacher, train_mode=train_mode)
        
        # 使用自定义DataParallel
        model = DataParallelWrapper(model, device_ids=[0, 1])
        
        print(f"✅ 模型已设置在GPU: {list(range(torch.cuda.device_count()))}")
        
        # 测试数据
        batch_size = 4
        input_seq = torch.randn(batch_size, 2, 256, 256).cuda()
        target_seq = torch.randn(batch_size, 5, 2, 256, 256).cuda()
        
        print(f"📊 输入形状: {input_seq.shape}")
        print(f"📊 目标形状: {target_seq.shape}")
        
        # 测试训练模式调用
        model.train()
        print("🔄 测试训练模式...")
        pred_seq = model(x_in=input_seq, T_out=5, teacher=target_seq)
        print(f"✅ 训练模式输出形状: {pred_seq.shape}")
        
        # 测试推理模式调用
        model.eval()
        print("🔄 测试推理模式...")
        with torch.no_grad():
            pred_seq = model(x_in=input_seq, T_out=5)
            print(f"✅ 推理模式输出形状: {pred_seq.shape}")
        
        # 内存使用情况
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"📊 GPU {i} 内存使用: {memory_allocated:.2f} GB")
        
        print("🎉 多GPU AR训练修复测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ar_wrapper_multi_gpu()
    exit(0 if success else 1)