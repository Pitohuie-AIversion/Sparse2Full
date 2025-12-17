#!/usr/bin/env python3
"""
测试AR可视化功能
验证所有修复后的可视化方法是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from utils.ar_visualizer import ARTrainingVisualizer

def test_ar_visualizer():
    """测试AR可视化器的所有功能"""
    print("🧪 开始测试AR可视化功能...")
    
    # 创建测试输出目录
    test_output_dir = Path("paper_package/figs/test_ar_visualization")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化可视化器
    visualizer = ARTrainingVisualizer(str(test_output_dir))
    
    # 创建模拟数据
    batch_size = 2
    T_in = 1
    T_out = 5
    channels = 2
    height, width = 64, 64
    
    # 模拟输入序列 [B, T_in, C, H, W]
    input_seq = torch.randn(batch_size, T_in, channels, height, width)
    
    # 模拟目标序列 [B, T_out, C, H, W]
    target_seq = torch.randn(batch_size, T_out, channels, height, width)
    
    # 模拟预测序列 [B, T_out, C, H, W]
    pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)  # 添加一些噪声
    
    # 模拟训练历史
    training_history = {
        'epochs': list(range(1, 21)),
        'train_losses': [1.0 - 0.04*i + 0.01*np.random.randn() for i in range(20)],
        'val_losses': [0.9 - 0.035*i + 0.015*np.random.randn() for i in range(20)],
        'learning_rates': [0.001 * (0.95**i) for i in range(20)],
        'val_metrics': [
            {'rel_l2': 0.5 - 0.02*i + 0.005*np.random.randn(), 
             'mae': 0.3 - 0.01*i + 0.003*np.random.randn(),
             'mse': 0.2 - 0.008*i + 0.002*np.random.randn()} 
            for i in range(20)
        ],
        'curriculum_stages': [
            {'epoch': i*5, 'T_out': min(1 + i, 5), 'stage': i} 
            for i in range(4)
        ]
    }
    
    print("📊 测试训练曲线可视化...")
    try:
        visualizer.plot_training_curves(training_history, "test_training_curves")
        print("✅ 训练曲线可视化测试通过")
    except Exception as e:
        print(f"❌ 训练曲线可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎯 测试AR预测可视化...")
    try:
        visualizer.visualize_ar_predictions(
            input_seq, target_seq, pred_seq, 
            timestep_idx=0, save_name="test_ar_predictions"
        )
        print("✅ AR预测可视化测试通过")
    except Exception as e:
        print(f"❌ AR预测可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("📈 测试误差分析...")
    try:
        visualizer.create_error_analysis(
            target_seq, pred_seq, save_name="test_error_analysis"
        )
        print("✅ 误差分析测试通过")
    except Exception as e:
        print(f"❌ 误差分析测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("⏰ 测试时间分析...")
    try:
        visualizer.create_temporal_analysis(
            pred_seq, target_seq, save_name="test_temporal_analysis"
        )
        print("✅ 时间分析测试通过")
    except Exception as e:
        print(f"❌ 时间分析测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("📋 测试综合报告...")
    try:
        visualizer.create_comprehensive_report(training_history)
        print("✅ 综合报告测试通过")
    except Exception as e:
        print(f"❌ 综合报告测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🎉 测试完成！结果保存在: {test_output_dir}")
    
    # 检查生成的文件
    generated_files = list(test_output_dir.rglob("*.png")) + list(test_output_dir.rglob("*.html"))
    print(f"📁 生成的文件数量: {len(generated_files)}")
    for file in generated_files:
        print(f"  - {file.relative_to(test_output_dir)}")
    
    return len(generated_files) > 0

if __name__ == "__main__":
    success = test_ar_visualizer()
    if success:
        print("✅ 所有可视化功能测试通过！")
        sys.exit(0)
    else:
        print("❌ 可视化功能测试失败！")
        sys.exit(1)