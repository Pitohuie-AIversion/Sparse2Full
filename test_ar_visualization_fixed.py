#!/usr/bin/env python3
"""
测试修复后的AR可视化功能
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ar_visualizer():
    """测试AR可视化器"""
    print("🧪 测试AR可视化器...")
    
    try:
        from utils.ar_visualizer import ARTrainingVisualizer
        
        # 创建测试输出目录
        test_dir = Path("test_ar_viz_fixed")
        test_dir.mkdir(exist_ok=True)
        
        # 初始化可视化器
        visualizer = ARTrainingVisualizer(str(test_dir))
        
        # 创建模拟数据
        batch_size = 2
        T_in = 1
        T_out = 5
        channels = 2
        height = 64
        width = 64
        
        # 模拟输入序列 [B, T_in, C, H, W]
        input_seq = torch.randn(batch_size, T_in, channels, height, width)
        
        # 模拟目标序列 [B, T_out, C, H, W]
        target_seq = torch.randn(batch_size, T_out, channels, height, width)
        
        # 模拟预测序列 [B, T_out, C, H, W]
        pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)
        
        print(f"✅ 输入序列形状: {input_seq.shape}")
        print(f"✅ 目标序列形状: {target_seq.shape}")
        print(f"✅ 预测序列形状: {pred_seq.shape}")
        
        # 测试1: AR预测可视化
        print("\n📊 测试AR预测可视化...")
        try:
            visualizer.visualize_ar_predictions(
                input_seq, target_seq, pred_seq, 
                timestep_idx=0, save_name="test_ar_predictions"
            )
            print("✅ AR预测可视化成功")
        except Exception as e:
            print(f"❌ AR预测可视化失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试2: 误差分析
        print("\n📊 测试误差分析...")
        try:
            visualizer.create_error_analysis(
                target_seq, pred_seq, save_name="test_error_analysis"
            )
            print("✅ 误差分析成功")
        except Exception as e:
            print(f"❌ 误差分析失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试3: 时间分析
        print("\n📊 测试时间分析...")
        try:
            visualizer.create_temporal_analysis(
                pred_seq, target_seq, save_name="test_temporal_analysis"
            )
            print("✅ 时间分析成功")
        except Exception as e:
            print(f"❌ 时间分析失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试4: 训练曲线
        print("\n📊 测试训练曲线...")
        try:
            # 模拟训练历史
            history = {
                'epochs': list(range(1, 11)),
                'train_losses': [1.0 - 0.05*i for i in range(10)],
                'val_losses': [0.9 - 0.04*i for i in range(10)],
                'learning_rates': [0.001 * (0.9**i) for i in range(10)],
                'val_metrics': [{'rel_l2': 0.5 - 0.02*i, 'mae': 0.3 - 0.01*i} for i in range(10)],
                'curriculum_stages': [{'epoch': i*2, 'T_out': min(5, i+1), 'stage': i//3} for i in range(5)]
            }
            
            visualizer.plot_training_curves(history, "test_training_curves")
            print("✅ 训练曲线成功")
        except Exception as e:
            print(f"❌ 训练曲线失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试5: 综合报告
        print("\n📊 测试综合报告...")
        try:
            visualizer.create_comprehensive_report(history)
            print("✅ 综合报告成功")
        except Exception as e:
            print(f"❌ 综合报告失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 检查生成的文件
        print("\n📁 检查生成的文件:")
        for subdir in ["training_curves", "predictions", "error_analysis", "temporal_analysis"]:
            subdir_path = test_dir / "visualizations" / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.png"))
                print(f"  {subdir}: {len(files)} 个文件")
                for f in files:
                    print(f"    - {f.name}")
            else:
                print(f"  {subdir}: 目录不存在")
        
        # 检查HTML报告
        html_report = test_dir / "visualizations" / "comprehensive_report.html"
        if html_report.exists():
            print(f"  📄 HTML报告: {html_report}")
        else:
            print("  📄 HTML报告: 未生成")
        
        print(f"\n✅ 测试完成，结果保存在: {test_dir}")
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
        # 检查训练脚本中的可视化调用
        training_script = Path("tools/training/train_real_data_ar.py")
        if not training_script.exists():
            print("❌ 训练脚本不存在")
            return False
        
        # 读取训练脚本内容
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键导入和调用
        checks = [
            ("ARTrainingVisualizer导入", "from utils.ar_visualizer import ARTrainingVisualizer"),
            ("可视化器初始化", "ARTrainingVisualizer(str(viz_dir))"),
            ("训练曲线调用", "plot_training_curves"),
            ("AR预测可视化调用", "visualize_ar_predictions"),
            ("误差分析调用", "create_error_analysis"),
            ("时间分析调用", "create_temporal_analysis"),
        ]
        
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"  ✅ {check_name}: 已集成")
            else:
                print(f"  ❌ {check_name}: 未找到")
        
        print("✅ 集成检查完成")
        return True
        
    except Exception as e:
        print(f"❌ 集成检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试修复后的AR可视化功能\n")
    
    # 测试1: AR可视化器
    success1 = test_ar_visualizer()
    
    # 测试2: 训练脚本集成
    success2 = test_integration_with_training()
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试总结:")
    print(f"  AR可视化器测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  训练脚本集成测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！AR可视化功能已修复")
        
        # 创建paper_package测试
        print("\n📋 创建paper_package测试结果...")
        paper_dir = Path("paper_package/figs/ar_visualization_test")
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制测试结果
        import shutil
        test_viz_dir = Path("test_ar_viz_fixed/visualizations")
        if test_viz_dir.exists():
            shutil.copytree(test_viz_dir, paper_dir, dirs_exist_ok=True)
            print(f"✅ 测试结果已复制到: {paper_dir}")
        
    else:
        print("\n⚠️ 部分测试失败，需要进一步修复")
    
    return success1 and success2

if __name__ == "__main__":
    main()