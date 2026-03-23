#!/usr/bin/env python3
"""
调试形状问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_shapes():
    """调试形状处理"""
    print("🔍 调试形状处理...")
    
    # 测试案例1: 5D输入
    input_seq = torch.randn(1, 1, 2, 32, 32)  # [B, T_in, C, H, W]
    target_seq = torch.randn(1, 3, 2, 32, 32)  # [B, T_out, C, H, W]
    pred_seq = torch.randn(1, 3, 2, 32, 32)   # [B, T_out, C, H, W]
    
    print(f"原始形状:")
    print(f"  input_seq: {input_seq.shape}")
    print(f"  target_seq: {target_seq.shape}")
    print(f"  pred_seq: {pred_seq.shape}")
    
    # 转换为numpy
    input_seq = input_seq.detach().cpu().numpy()
    target_seq = target_seq.detach().cpu().numpy()
    pred_seq = pred_seq.detach().cpu().numpy()
    
    # 处理形状
    if len(input_seq.shape) == 5:  # [B, T_in, C, H, W]
        input_frame = input_seq[0, -1]  # 取最后一个输入帧 [C, H, W]
        target_frames = target_seq[0]  # [T_out, C, H, W]
        pred_frames = pred_seq[0]  # [T_out, C, H, W]
    
    print(f"\n处理后形状:")
    print(f"  input_frame: {input_frame.shape}")
    print(f"  target_frames: {target_frames.shape}")
    print(f"  pred_frames: {pred_frames.shape}")
    
    # 测试imshow
    print(f"\n测试imshow:")
    
    # 输入帧
    input_img = input_frame[0] if input_frame.ndim > 2 else input_frame
    print(f"  input_img初始: {input_img.shape}")
    while input_img.ndim > 2:
        input_img = input_img[0]
        print(f"  input_img降维后: {input_img.shape}")
    
    # 目标帧
    target_img = target_frames[0, 0] if target_frames.ndim > 3 else target_frames[0]
    print(f"  target_img初始: {target_img.shape}")
    while target_img.ndim > 2:
        target_img = target_img[0]
        print(f"  target_img降维后: {target_img.shape}")
    
    # 预测帧
    pred_img = pred_frames[0, 0] if pred_frames.ndim > 3 else pred_frames[0]
    print(f"  pred_img初始: {pred_img.shape}")
    while pred_img.ndim > 2:
        pred_img = pred_img[0]
        print(f"  pred_img降维后: {pred_img.shape}")
    
    # 尝试显示
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_img, cmap='RdBu_r')
        axes[0].set_title('Input')
        axes[1].imshow(target_img, cmap='RdBu_r')
        axes[1].set_title('Target')
        axes[2].imshow(pred_img, cmap='RdBu_r')
        axes[2].set_title('Prediction')
        plt.savefig('debug_shapes.png', dpi=150)
        plt.close()
        print("✅ 图像显示成功")
    except Exception as e:
        print(f"❌ 图像显示失败: {e}")
    
    # 测试案例2: 4D输入
    print(f"\n" + "="*50)
    print("测试案例2: 4D输入")
    
    input_seq2 = torch.randn(1, 2, 32, 32)     # [T_in, C, H, W]
    target_seq2 = torch.randn(3, 2, 32, 32)    # [T_out, C, H, W]
    pred_seq2 = torch.randn(3, 2, 32, 32)      # [T_out, C, H, W]
    
    print(f"原始形状:")
    print(f"  input_seq2: {input_seq2.shape}")
    print(f"  target_seq2: {target_seq2.shape}")
    print(f"  pred_seq2: {pred_seq2.shape}")
    
    # 转换为numpy
    input_seq2 = input_seq2.detach().cpu().numpy()
    target_seq2 = target_seq2.detach().cpu().numpy()
    pred_seq2 = pred_seq2.detach().cpu().numpy()
    
    # 处理形状
    if len(input_seq2.shape) == 4:  # [B, C, H, W] 或 [T, C, H, W]
        if input_seq2.shape[0] <= 8:  # 假设是batch维度
            input_frame2 = input_seq2[0]  # [C, H, W]
            target_frames2 = target_seq2  # [T, C, H, W]
            pred_frames2 = pred_seq2  # [T, C, H, W]
        else:  # 假设是时间维度
            input_frame2 = input_seq2[-1]  # 取最后一个输入帧 [C, H, W]
            target_frames2 = target_seq2  # [T, C, H, W]
            pred_frames2 = pred_seq2  # [T, C, H, W]
    
    print(f"\n处理后形状:")
    print(f"  input_frame2: {input_frame2.shape}")
    print(f"  target_frames2: {target_frames2.shape}")
    print(f"  pred_frames2: {pred_frames2.shape}")
    
    # 测试imshow
    print(f"\n测试imshow:")
    
    # 输入帧
    input_img2 = input_frame2[0] if input_frame2.ndim > 2 else input_frame2
    print(f"  input_img2初始: {input_img2.shape}")
    while input_img2.ndim > 2:
        input_img2 = input_img2[0]
        print(f"  input_img2降维后: {input_img2.shape}")
    
    # 目标帧
    target_img2 = target_frames2[0, 0] if target_frames2.ndim > 3 else target_frames2[0]
    print(f"  target_img2初始: {target_img2.shape}")
    while target_img2.ndim > 2:
        target_img2 = target_img2[0]
        print(f"  target_img2降维后: {target_img2.shape}")
    
    # 预测帧
    pred_img2 = pred_frames2[0, 0] if pred_frames2.ndim > 3 else pred_frames2[0]
    print(f"  pred_img2初始: {pred_img2.shape}")
    while pred_img2.ndim > 2:
        pred_img2 = pred_img2[0]
        print(f"  pred_img2降维后: {pred_img2.shape}")
    
    # 尝试显示
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_img2, cmap='RdBu_r')
        axes[0].set_title('Input')
        axes[1].imshow(target_img2, cmap='RdBu_r')
        axes[1].set_title('Target')
        axes[2].imshow(pred_img2, cmap='RdBu_r')
        axes[2].set_title('Prediction')
        plt.savefig('debug_shapes2.png', dpi=150)
        plt.close()
        print("✅ 图像显示成功")
    except Exception as e:
        print(f"❌ 图像显示失败: {e}")

if __name__ == "__main__":
    debug_shapes()