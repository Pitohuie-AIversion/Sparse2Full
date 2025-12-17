#!/usr/bin/env python3
"""
测试可视化功能集成
创建模拟数据来测试可视化功能是否正常工作
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# 导入可视化模块
try:
    from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer
    from tools.visualization.ar_visualizer import ARTrainingVisualizer
    VISUALIZATION_AVAILABLE = True
    print("✅ 可视化模块导入成功")
except ImportError as e:
    print(f"❌ 可视化模块导入失败: {e}")
    VISUALIZATION_AVAILABLE = False
    sys.exit(1)


def create_mock_training_history():
    """创建模拟训练历史数据"""
    epochs = 20
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_metrics': [],
        'learning_rates': [],
        'epochs': [],
        'curriculum_stages': []
    }
    
    # 生成模拟数据
    for epoch in range(epochs):
        # 模拟损失下降
        train_loss = 2.0 * np.exp(-epoch/10) + 0.1 * np.random.random()
        val_loss = 2.2 * np.exp(-epoch/10) + 0.15 * np.random.random()
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['epochs'].append(epoch)
        history['learning_rates'].append(1e-3 * (0.9 ** (epoch // 5)))
        
        # 模拟验证指标
        val_metrics = {
            'rel_l2': val_loss * 0.5,
            'mae': val_loss * 0.3,
            'psnr': 20 - val_loss * 2,
            'ssim': 0.9 - val_loss * 0.1,
            'val_loss': val_loss
        }
        history['val_metrics'].append(val_metrics)
        
        # 模拟课程学习阶段
        if epoch < 5:
            T_out = 5
            stage = 0
        elif epoch < 10:
            T_out = 10
            stage = 1
        elif epoch < 15:
            T_out = 15
            stage = 2
        else:
            T_out = 20
            stage = 3
            
        history['curriculum_stages'].append({
            'epoch': epoch,
            'T_out': T_out,
            'stage': stage
        })
    
    return history


def create_mock_ar_data():
    """创建模拟AR数据"""
    batch_size = 2
    T_in = 1
    T_out = 5
    channels = 1  # 改为单通道以避免可视化问题
    height = 64
    width = 64
    
    # 创建模拟输入序列
    input_seq = torch.randn(batch_size, T_in, channels, height, width)
    
    # 创建模拟目标序列
    target_seq = torch.randn(batch_size, T_out, channels, height, width)
    
    # 创建模拟预测序列（添加一些噪声）
    pred_seq = target_seq + 0.1 * torch.randn_like(target_seq)
    
    return {
        'input_sequence': input_seq,
        'target_sequence': target_seq,
        'predictions': pred_seq
    }


def test_training_visualization():
    """测试训练过程可视化"""
    print("\n🧪 测试训练过程可视化...")
    
    # 创建测试目录
    test_dir = Path("test_viz_integration")
    test_dir.mkdir(exist_ok=True)
    
    # 创建模拟运行目录
    run_dir = test_dir / "mock_run"
    run_dir.mkdir(exist_ok=True)
    
    # 创建模拟训练历史
    history = create_mock_training_history()
    
    # 保存训练历史
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 创建可视化器
    viz_dir = test_dir / "visualizations"
    visualizer = PDEBenchVisualizer(str(viz_dir))
    
    try:
        # 测试训练结果可视化
        success = visualizer.visualize_training_results(str(run_dir))
        if success:
            print("✅ 训练结果可视化成功")
        else:
            print("❌ 训练结果可视化失败")
            return False
        
        # 检查生成的文件
        expected_files = [
            viz_dir / "training_curves" / "training_curves.png",
            viz_dir / "training_curves" / "training_stats.json"
        ]
        
        for file_path in expected_files:
            if file_path.exists():
                print(f"✅ 文件已生成: {file_path}")
            else:
                print(f"❌ 文件未生成: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 训练可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ar_visualization():
    """测试AR预测可视化"""
    print("\n🧪 测试AR预测可视化...")
    
    # 创建测试目录
    test_dir = Path("test_viz_integration")
    viz_dir = test_dir / "ar_visualizations"
    
    # 创建可视化器
    visualizer = PDEBenchVisualizer(str(viz_dir))
    
    try:
        # 创建模拟AR数据
        sample_data = create_mock_ar_data()
        
        # 测试AR预测可视化
        success = visualizer.visualize_ar_predictions(
            sample_data['input_sequence'],
            sample_data['target_sequence'],
            sample_data['predictions'],
            epoch=10,
            timestep=0
        )
        
        if success:
            print("✅ AR预测可视化成功")
        else:
            print("❌ AR预测可视化失败")
            return False
        
        # 测试误差分析
        success = visualizer.create_error_analysis(
            sample_data['input_sequence'],
            sample_data['target_sequence'],
            sample_data['predictions']
        )
        
        if success:
            print("✅ 误差分析可视化成功")
        else:
            print("❌ 误差分析可视化失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AR可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_report():
    """测试综合报告生成"""
    print("\n🧪 测试综合报告生成...")
    
    # 创建测试目录
    test_dir = Path("test_viz_integration")
    run_dir = test_dir / "mock_run"
    viz_dir = test_dir / "comprehensive_report"
    
    # 创建可视化器
    visualizer = PDEBenchVisualizer(str(viz_dir))
    
    try:
        # 创建模拟样本数据
        sample_data = create_mock_ar_data()
        
        # 生成综合报告
        success = visualizer.create_comprehensive_report(
            str(run_dir),
            sample_data={
                'input_seq': sample_data['input_sequence'],
                'target_seq': sample_data['target_sequence'],
                'pred_seq': sample_data['predictions']
            }
        )
        
        if success:
            print("✅ 综合报告生成成功")
            
            # 检查HTML报告
            html_report = viz_dir / "report.html"
            if html_report.exists():
                print(f"✅ HTML报告已生成: {html_report}")
                return True
            else:
                print("❌ HTML报告未生成")
                return False
        else:
            print("❌ 综合报告生成失败")
            return False
        
    except Exception as e:
        print(f"❌ 综合报告测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始可视化功能集成测试...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not VISUALIZATION_AVAILABLE:
        print("❌ 可视化模块不可用，测试终止")
        return False
    
    # 运行测试
    tests = [
        ("训练过程可视化", test_training_visualization),
        ("AR预测可视化", test_ar_visualization),
        ("综合报告生成", test_comprehensive_report)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
                
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有可视化功能测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)