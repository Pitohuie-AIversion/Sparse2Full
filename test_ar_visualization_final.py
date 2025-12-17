#!/usr/bin/env python3
"""
测试修复后的AR可视化功能
验证所有可视化组件是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ar_visualizer():
    """测试AR可视化器的所有功能"""
    print("🧪 开始测试AR可视化器...")
    
    try:
        from utils.ar_visualizer import ARTrainingVisualizer
        
        # 创建测试输出目录
        test_dir = project_root / "paper_package" / "figs" / "ar_test_final"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化可视化器
        visualizer = ARTrainingVisualizer(str(test_dir))
        print("✅ AR可视化器初始化成功")
        
        # 1. 测试训练曲线可视化
        print("\n📈 测试训练曲线可视化...")
        history = {
            'epochs': list(range(1, 21)),
            'train_losses': [0.1 * np.exp(-0.1 * i) + 0.01 * np.random.random() for i in range(20)],
            'val_losses': [0.12 * np.exp(-0.08 * i) + 0.015 * np.random.random() for i in range(20)],
            'learning_rates': [0.001 * (0.9 ** (i // 5)) for i in range(20)],
            'val_metrics': [
                {'rel_l2': 0.1 * np.exp(-0.08 * i) + 0.01 * np.random.random(),
                 'mae': 0.05 * np.exp(-0.08 * i) + 0.005 * np.random.random(),
                 'mse': 0.01 * np.exp(-0.08 * i) + 0.001 * np.random.random()}
                for i in range(20)
            ],
            'curriculum_stages': [
                {'epoch': i * 5, 'T_out': min(1 + i, 5), 'stage': i}
                for i in range(4)
            ]
        }
        
        visualizer.plot_training_curves(history, "test_training_curves")
        print("✅ 训练曲线可视化成功")
        
        # 2. 测试AR预测可视化
        print("\n🎯 测试AR预测可视化...")
        
        # 创建测试数据 - 模拟真实的AR预测场景
        batch_size = 2
        T_in = 1
        T_out = 5
        channels = 2
        height, width = 64, 64
        
        # 输入序列 [B, T_in, C, H, W]
        input_seq = torch.randn(batch_size, T_in, channels, height, width)
        
        # 目标序列 [B, T_out, C, H, W]
        target_seq = torch.randn(batch_size, T_out, channels, height, width)
        
        # 预测序列 [B, T_out, C, H, W] - 添加一些噪声使其与目标不同
        pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)
        
        visualizer.visualize_ar_predictions(
            input_seq, target_seq, pred_seq, 
            timestep_idx=0, save_name="test_ar_predictions"
        )
        print("✅ AR预测可视化成功")
        
        # 3. 测试误差分析
        print("\n📊 测试误差分析...")
        visualizer.create_error_analysis(
            target_seq, pred_seq, save_name="test_error_analysis"
        )
        print("✅ 误差分析可视化成功")
        
        # 4. 测试时间分析
        print("\n⏰ 测试时间分析...")
        visualizer.create_temporal_analysis(
            pred_seq, target_seq, save_name="test_temporal_analysis"
        )
        print("✅ 时间分析可视化成功")
        
        # 5. 测试综合报告
        print("\n📋 测试综合报告...")
        visualizer.create_comprehensive_report(history)
        print("✅ 综合报告生成成功")
        
        # 检查生成的文件
        print("\n📁 检查生成的文件...")
        generated_files = []
        for subdir in ["training_curves", "predictions", "error_analysis", "temporal_analysis"]:
            subdir_path = test_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.glob("*.png"):
                    generated_files.append(str(file.relative_to(test_dir)))
        
        # 检查HTML报告
        html_report = test_dir / "comprehensive_report.html"
        if html_report.exists():
            generated_files.append("comprehensive_report.html")
        
        print(f"✅ 生成了 {len(generated_files)} 个文件:")
        for file in generated_files:
            print(f"   - {file}")
        
        print(f"\n🎉 所有测试通过！可视化文件保存在: {test_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shape_handling():
    """专门测试形状处理功能"""
    print("\n🔧 测试形状处理功能...")
    
    try:
        from utils.ar_visualizer import ARTrainingVisualizer
        
        test_dir = project_root / "paper_package" / "figs" / "shape_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = ARTrainingVisualizer(str(test_dir))
        
        # 测试各种形状的数据
        test_cases = [
            # (形状描述, target_seq形状, pred_seq形状)
            ("标准4D", (1, 5, 2, 32, 32), (1, 5, 2, 32, 32)),
            ("3D数据", (5, 1, 64, 64), (5, 1, 64, 64)),
            ("单通道", (1, 3, 1, 48, 48), (1, 3, 1, 48, 48)),
            ("大尺寸", (1, 2, 2, 128, 128), (1, 2, 2, 128, 128)),
        ]
        
        for desc, target_shape, pred_shape in test_cases:
            print(f"   测试 {desc}: {target_shape} -> {pred_shape}")
            
            target_seq = torch.randn(*target_shape)
            pred_seq = torch.randn(*pred_shape)
            
            # 测试误差分析
            visualizer.create_error_analysis(
                target_seq, pred_seq, save_name=f"shape_test_{desc.replace(' ', '_')}"
            )
            
            # 测试时间分析
            visualizer.create_temporal_analysis(
                pred_seq, target_seq, save_name=f"temporal_test_{desc.replace(' ', '_')}"
            )
            
            print(f"   ✅ {desc} 测试通过")
        
        print("✅ 所有形状处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 形状处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始AR可视化功能完整测试")
    print("=" * 50)
    
    # 设置matplotlib后端
    plt.switch_backend('Agg')  # 非交互式后端
    
    success = True
    
    # 测试基本功能
    if not test_ar_visualizer():
        success = False
    
    # 测试形状处理
    if not test_shape_handling():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！AR可视化功能完全正常")
        print("📁 可视化文件已保存到 paper_package/figs/ 目录")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    return success

if __name__ == "__main__":
    main()