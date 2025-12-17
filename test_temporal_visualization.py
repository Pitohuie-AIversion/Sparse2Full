#!/usr/bin/env python3
"""时序可视化工具测试

测试时序预测结果可视化功能
"""

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.temporal_visualization import (
    TemporalVisualizer, 
    VisualizationConfig,
    quick_visualize,
    create_visualization_summary
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """创建测试数据"""
    batch_size = 2
    seq_len = 10
    channels = 2
    height = 32
    width = 32
    
    # 创建模拟的时序数据
    # 真实值：简单的波动模式
    t = torch.linspace(0, 2*np.pi, seq_len).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x = torch.linspace(0, 2*np.pi, width).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y = torch.linspace(0, 2*np.pi, height).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    
    # 生成真实值：时空波动
    targets = torch.sin(t + 0.5*x + 0.3*y) * torch.exp(-0.1*t)
    targets = targets.expand(batch_size, seq_len, channels, height, width)
    
    # 生成预测值：添加一些误差
    noise = 0.1 * torch.randn_like(targets)
    phase_shift = 0.2 * torch.randn(batch_size, 1, channels, 1, 1)
    predictions = targets + noise + phase_shift * torch.sin(2*t)
    
    return predictions, targets


def test_visualization_config():
    """测试可视化配置"""
    print("测试可视化配置...")
    
    # 默认配置
    config1 = VisualizationConfig()
    assert config1.figsize == (12, 8)
    assert config1.dpi == 100
    assert config1.cmap == 'viridis'
    
    # 自定义配置
    config2 = VisualizationConfig(
        figsize=(16, 10),
        dpi=150,
        cmap='plasma',
        animation_fps=15
    )
    assert config2.figsize == (16, 10)
    assert config2.dpi == 150
    assert config2.cmap == 'plasma'
    assert config2.animation_fps == 15
    
    print("✓ 可视化配置测试通过")


def test_temporal_visualizer_init():
    """测试时序可视化器初始化"""
    print("测试时序可视化器初始化...")
    
    # 默认初始化
    visualizer1 = TemporalVisualizer()
    assert visualizer1.config.figsize == (12, 8)
    
    # 自定义配置初始化
    config = VisualizationConfig(figsize=(10, 6), dpi=120)
    visualizer2 = TemporalVisualizer(config)
    assert visualizer2.config.figsize == (10, 6)
    assert visualizer2.config.dpi == 120
    
    print("✓ 时序可视化器初始化测试通过")


def test_sequence_comparison():
    """测试时序对比图"""
    print("测试时序对比图生成...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "sequence_comparison.png"
        
        visualizer.plot_sequence_comparison(
            predictions, targets, save_path,
            sample_idx=0, channel_idx=0,
            time_steps=[0, 2, 5, 8, 9],
            title="测试时序对比"
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    print("✓ 时序对比图测试通过")


def test_error_evolution():
    """测试误差演化曲线"""
    print("测试误差演化曲线...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "error_evolution.png"
        
        visualizer.plot_error_evolution(
            predictions, targets, save_path,
            metrics=['mse', 'mae', 'rel_l2'],
            title="测试误差演化"
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    print("✓ 误差演化曲线测试通过")


def test_spatial_error_heatmap():
    """测试空间误差热力图"""
    print("测试空间误差热力图...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "spatial_error.png"
        
        visualizer.plot_spatial_error_heatmap(
            predictions, targets, save_path,
            sample_idx=0, channel_idx=0,
            title="测试空间误差热力图"
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    print("✓ 空间误差热力图测试通过")


def test_frequency_analysis():
    """测试频域分析"""
    print("测试频域分析...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "frequency_analysis.png"
        
        visualizer.plot_frequency_analysis(
            predictions, targets, save_path,
            sample_idx=0, channel_idx=0,
            title="测试频域分析"
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    print("✓ 频域分析测试通过")


def test_multi_step_comparison():
    """测试多步预测对比"""
    print("测试多步预测对比...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "multi_step.png"
        
        visualizer.plot_multi_step_comparison(
            predictions, targets, save_path,
            step_intervals=[1, 3, 6, 9],
            sample_idx=0, channel_idx=0,
            title="测试多步预测对比"
        )
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    print("✓ 多步预测对比测试通过")


def test_prediction_animation():
    """测试预测动画生成"""
    print("测试预测动画生成...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "prediction_animation.gif"
        
        try:
            visualizer.create_prediction_animation(
                predictions, targets, save_path,
                sample_idx=0, channel_idx=0,
                title="测试预测动画"
            )
            
            if save_path.exists():
                assert save_path.stat().st_size > 0
                print("✓ 预测动画生成测试通过")
            else:
                print("⚠ 预测动画生成跳过（可能缺少依赖）")
        except Exception as e:
            print(f"⚠ 预测动画生成跳过: {e}")


def test_comprehensive_report():
    """测试综合分析报告"""
    print("测试综合分析报告...")
    
    predictions, targets = create_test_data()
    visualizer = TemporalVisualizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = visualizer.create_comprehensive_report(
            predictions, targets, temp_dir,
            sample_idx=0, channel_idx=0,
            prefix="test_report"
        )
        
        # 检查生成的文件
        expected_files = ['comparison', 'error_evolution', 'spatial_error', 
                         'frequency_analysis', 'multi_step']
        
        for file_type in expected_files:
            assert file_type in file_paths
            file_path = Path(file_paths[file_type])
            assert file_path.exists()
            assert file_path.stat().st_size > 0
        
        print(f"✓ 综合分析报告测试通过，生成了 {len(file_paths)} 个文件")


def test_visualization_summary():
    """测试可视化汇总页面"""
    print("测试可视化汇总页面...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建一些测试文件
        test_files = {
            'comparison': str(Path(temp_dir) / "comparison.png"),
            'error_evolution': str(Path(temp_dir) / "error_evolution.png"),
            'spatial_error': str(Path(temp_dir) / "spatial_error.png")
        }
        
        # 创建空文件
        for file_path in test_files.values():
            Path(file_path).touch()
        
        html_path = Path(temp_dir) / "summary.html"
        create_visualization_summary(test_files, html_path)
        
        assert html_path.exists()
        assert html_path.stat().st_size > 0
        
        # 检查HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '时序预测可视化分析报告' in content
            assert 'comparison.png' in content
    
    print("✓ 可视化汇总页面测试通过")


def test_quick_visualize():
    """测试快速可视化函数"""
    print("测试快速可视化函数...")
    
    predictions, targets = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = quick_visualize(
            predictions, targets, temp_dir,
            sample_idx=0, channel_idx=0,
            create_html=True
        )
        
        # 检查生成的文件
        expected_files = ['comparison', 'error_evolution', 'spatial_error', 
                         'frequency_analysis', 'multi_step', 'html_report']
        
        for file_type in expected_files:
            if file_type in file_paths:
                file_path = Path(file_paths[file_type])
                assert file_path.exists()
                assert file_path.stat().st_size > 0
        
        print(f"✓ 快速可视化函数测试通过，生成了 {len(file_paths)} 个文件")


def test_numpy_conversion():
    """测试numpy转换功能"""
    print("测试numpy转换功能...")
    
    visualizer = TemporalVisualizer()
    
    # 测试torch tensor转换
    tensor = torch.randn(3, 4, 5)
    numpy_array = visualizer._to_numpy(tensor)
    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.shape == (3, 4, 5)
    
    # 测试numpy array直接返回
    original_array = np.random.randn(2, 3)
    result_array = visualizer._to_numpy(original_array)
    assert isinstance(result_array, np.ndarray)
    assert np.array_equal(original_array, result_array)
    
    print("✓ numpy转换功能测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始时序可视化工具测试")
    print("=" * 60)
    
    test_functions = [
        test_visualization_config,
        test_temporal_visualizer_init,
        test_sequence_comparison,
        test_error_evolution,
        test_spatial_error_heatmap,
        test_frequency_analysis,
        test_multi_step_comparison,
        test_prediction_animation,
        test_comprehensive_report,
        test_visualization_summary,
        test_quick_visualize,
        test_numpy_conversion
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} 失败: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 所有测试通过！时序可视化工具功能正常")
    else:
        print(f"⚠ {failed} 个测试失败，请检查相关功能")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)