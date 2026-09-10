#!/usr/bin/env python3
"""
调试可视化问题
"""

import torch
import numpy as np
from pathlib import Path
from tools.visualization.ar_visualizer import ARTrainingVisualizer

def test_ar_visualizer():
    """测试AR可视化器"""
    print("🧪 调试AR可视化器...")
    
    # 创建测试数据
    batch_size = 1
    T_in = 1
    T_out = 5
    channels = 1
    height = 64
    width = 64
    
    input_seq = torch.randn(batch_size, T_in, channels, height, width)
    target_seq = torch.randn(batch_size, T_out, channels, height, width)
    pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)
    
    print(f"数据形状:")
    print(f"  input_seq: {input_seq.shape}")
    print(f"  target_seq: {target_seq.shape}")
    print(f"  pred_seq: {pred_seq.shape}")
    
    # 创建可视化器
    viz_dir = Path("debug_viz")
    viz_dir.mkdir(exist_ok=True)
    
    visualizer = ARTrainingVisualizer(str(viz_dir))
    
    # 测试可视化
    try:
        success = visualizer.visualize_ar_predictions(
            input_seq, target_seq, pred_seq, epoch=10, timestep=0
        )
        print(f"✅ AR预测可视化: {'成功' if success else '失败'}")
    except Exception as e:
        print(f"❌ AR预测可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ar_visualizer()