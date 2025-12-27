#!/usr/bin/env python3
"""
最终验证AR可视化功能
确认所有修复都正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ar_visualizer():
    """测试AR可视化器的所有功能"""
    print("🧪 开始测试AR可视化器...")
    
    try:
        from utils.ar_visualizer import ARTrainingVisualizer
        
        # 创建测试输出目录
        test_dir = project_root / "paper_package" / "figs" / "final_verification"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化可视化器
        visualizer = ARTrainingVisualizer(str(test_dir))
        print("✅ AR可视化器初始化成功")
        
        # 1. 测试训练曲线可视化
        print("📊 测试训练曲线可视化...")
        history = {
            'epochs': list(range(50)),
            'train_losses': [1.0 - 0.01*i + 0.1*np.sin(i*0.1) for i in range(50)],
            'val_losses': [0.9 - 0.008*i + 0.05*np.sin(i*0.1) for i in range(50)],
            'learning_rates': [0.001 * (0.95**i) for i in range(50)],
            'val_metrics': [{'rel_l2': 0.5 - 0.005*i, 'mae': 0.3 - 0.003*i} for i in range(50)],
            'curriculum_stages': [
                {'epoch': 10, 'T_out': 5, 'stage': 1},
                {'epoch': 25, 'T_out': 10, 'stage': 2},
                {'epoch': 40, 'T_out': 20, 'stage': 3}
            ]
        }
        
        visualizer.plot_training_curves(history, "final_test_training_curves")
        print("✅ 训练曲线可视化成功")
        
        # 2. 测试AR预测可视化
        print("🎯 测试AR预测可视化...")
        # 创建测试数据 - 模拟真实的AR预测场景
        batch_size, T_in, T_out, channels, height, width = 2, 5, 10, 2, 64, 64
        
        # 输入序列 [B, T_in, C, H, W]
        input_seq = torch.randn(batch_size, T_in, channels, height, width)
        
        # 目标序列 [B, T_out, C, H, W] 
        target_seq = torch.randn(batch_size, T_out, channels, height, width)
        
        # 预测序列 [B, T_out, C, H, W]
        pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)  # 添加一些噪声
        
        visualizer.visualize_ar_predictions(
            input_seq, target_seq, pred_seq, 
            timestep_idx=0, save_name="final_test_ar_predictions"
        )
        print("✅ AR预测可视化成功")
        
        # 3. 测试误差分析
        print("📈 测试误差分析...")
        visualizer.create_error_analysis(
            target_seq, pred_seq, save_name="final_test_error_analysis"
        )
        print("✅ 误差分析可视化成功")
        
        # 4. 测试时间分析
        print("⏰ 测试时间分析...")
        visualizer.create_temporal_analysis(
            pred_seq, target_seq, save_name="final_test_temporal_analysis"
        )
        print("✅ 时间分析可视化成功")
        
        # 5. 测试综合报告
        print("📋 测试综合报告...")
        visualizer.create_comprehensive_report(history)
        print("✅ 综合报告生成成功")
        
        # 检查生成的文件
        print("\n📁 检查生成的文件...")
        generated_files = []
        for subdir in ["training_curves", "predictions", "error_analysis", "temporal_analysis"]:
            subdir_path = test_dir / "visualizations" / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.png"))
                generated_files.extend(files)
                print(f"  {subdir}: {len(files)} 个PNG文件")
        
        # 检查HTML报告
        html_files = list((test_dir / "visualizations").glob("*.html"))
        print(f"  HTML报告: {len(html_files)} 个文件")
        
        total_files = len(generated_files) + len(html_files)
        print(f"\n🎉 总共生成了 {total_files} 个可视化文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_training():
    """测试与训练脚本的集成"""
    print("\n🔗 测试与训练脚本的集成...")
    
    try:
        # 模拟训练脚本中的调用
        from utils.ar_visualizer import ARTrainingVisualizer
        
        test_dir = project_root / "paper_package" / "figs" / "integration_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = ARTrainingVisualizer(str(test_dir))
        
        # 模拟训练过程中的数据
        sample_batch = {
            'input_sequence': torch.randn(1, 5, 2, 64, 64),
            'target_sequence': torch.randn(1, 10, 2, 64, 64),
            'predictions': torch.randn(1, 10, 2, 64, 64)
        }
        
        # 模拟训练历史
        training_history = {
            'epochs': [0, 1, 2, 3, 4],
            'train_losses': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_losses': [0.9, 0.7, 0.5, 0.35, 0.25],
            'learning_rates': [0.001, 0.0008, 0.0006, 0.0004, 0.0002],
            'val_metrics': [{'rel_l2': 0.5, 'mae': 0.3} for _ in range(5)],
            'curriculum_stages': [{'epoch': 2, 'T_out': 10, 'stage': 1}]
        }
        
        # 测试训练脚本中的调用方式
        epoch = 4
        
        # 训练曲线
        visualizer.plot_training_curves(training_history, f"training_curves_epoch_{epoch}")
        
        # AR预测
        visualizer.visualize_ar_predictions(
            sample_batch['input_sequence'], 
            sample_batch['target_sequence'], 
            sample_batch['predictions'],
            timestep_idx=epoch, 
            save_name=f"ar_predictions_epoch_{epoch}"
        )
        
        # 误差分析
        visualizer.create_error_analysis(
            sample_batch['target_sequence'], 
            sample_batch['predictions'],
            save_name=f"error_analysis_epoch_{epoch}"
        )
        
        # 时间分析
        visualizer.create_temporal_analysis(
            sample_batch['predictions'], 
            sample_batch['target_sequence'],
            save_name=f"temporal_analysis_epoch_{epoch}"
        )
        
        print("✅ 训练脚本集成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始AR可视化功能最终验证")
    print("=" * 50)
    
    # 测试1: AR可视化器功能
    test1_passed = test_ar_visualizer()
    
    # 测试2: 训练脚本集成
    test2_passed = test_integration_with_training()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"  AR可视化器功能: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"  训练脚本集成: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！AR可视化功能完全正常工作！")
        print("📁 可视化文件已保存到 paper_package/figs/ 目录")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)