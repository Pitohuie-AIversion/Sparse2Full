#!/usr/bin/env python3
"""
专门测试形状处理的修复
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_shape_handling():
    """测试形状处理"""
    print("🧪 测试形状处理...")
    
    from utils.ar_visualizer import ARTrainingVisualizer
    
    # 创建测试输出目录
    test_dir = Path("test_shape_fix")
    test_dir.mkdir(exist_ok=True)
    
    # 初始化可视化器
    visualizer = ARTrainingVisualizer(str(test_dir))
    
    # 测试不同的形状组合
    test_cases = [
        {
            "name": "5D输入 [B, T_in, C, H, W] -> [B, T_out, C, H, W]",
            "input_seq": torch.randn(1, 1, 2, 32, 32),  # [B, T_in, C, H, W]
            "target_seq": torch.randn(1, 3, 2, 32, 32),  # [B, T_out, C, H, W]
            "pred_seq": torch.randn(1, 3, 2, 32, 32),   # [B, T_out, C, H, W]
        },
        {
            "name": "4D输入 [T, C, H, W]",
            "input_seq": torch.randn(1, 2, 32, 32),     # [T_in, C, H, W]
            "target_seq": torch.randn(3, 2, 32, 32),    # [T_out, C, H, W]
            "pred_seq": torch.randn(3, 2, 32, 32),      # [T_out, C, H, W]
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📊 测试案例 {i+1}: {case['name']}")
        print(f"  输入形状: {case['input_seq'].shape}")
        print(f"  目标形状: {case['target_seq'].shape}")
        print(f"  预测形状: {case['pred_seq'].shape}")
        
        try:
            # 测试AR预测可视化
            visualizer.visualize_ar_predictions(
                case['input_seq'], case['target_seq'], case['pred_seq'],
                timestep_idx=i, save_name=f"test_case_{i+1}_predictions"
            )
            print(f"  ✅ AR预测可视化成功")
        except Exception as e:
            print(f"  ❌ AR预测可视化失败: {e}")
        
        try:
            # 测试误差分析
            visualizer.create_error_analysis(
                case['target_seq'], case['pred_seq'],
                save_name=f"test_case_{i+1}_error"
            )
            print(f"  ✅ 误差分析成功")
        except Exception as e:
            print(f"  ❌ 误差分析失败: {e}")
        
        try:
            # 测试时间分析
            visualizer.create_temporal_analysis(
                case['pred_seq'], case['target_seq'],
                save_name=f"test_case_{i+1}_temporal"
            )
            print(f"  ✅ 时间分析成功")
        except Exception as e:
            print(f"  ❌ 时间分析失败: {e}")
    
    # 检查生成的文件
    print(f"\n📁 生成的文件:")
    viz_dir = test_dir / "visualizations"
    if viz_dir.exists():
        for subdir in viz_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.png"))
                print(f"  {subdir.name}: {len(files)} 个文件")
    
    print(f"\n✅ 形状处理测试完成，结果保存在: {test_dir}")

if __name__ == "__main__":
    test_shape_handling()