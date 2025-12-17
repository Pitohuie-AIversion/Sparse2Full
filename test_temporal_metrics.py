#!/usr/bin/env python3
"""测试时序指标计算功能

验证多步指标计算的正确性和性能
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

from utils.temporal_metrics import TemporalMetricsCalculator, TemporalMetrics

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_temporal_data(batch_size: int = 4, 
                                  time_steps: int = 10, 
                                  channels: int = 1,
                                  height: int = 64, 
                                  width: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建合成时序数据用于测试
    
    Returns:
        predictions: [B, T, C, H, W]
        targets: [B, T, C, H, W]
    """
    # 创建基础模式
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    predictions = torch.zeros(batch_size, time_steps, channels, height, width)
    targets = torch.zeros(batch_size, time_steps, channels, height, width)
    
    for b in range(batch_size):
        for t in range(time_steps):
            for c in range(channels):
                # 目标：随时间演化的波动模式
                t_tensor = torch.tensor(t, dtype=torch.float32)
                target_pattern = torch.sin(2 * np.pi * (X + Y) + t * 0.5) * torch.exp(-0.1 * t_tensor)
                targets[b, t, c] = target_pattern
                
                # 预测：添加一些误差，误差随时间累积
                error_scale = 0.1 * (1 + 0.1 * t)  # 误差随时间增长
                noise = torch.randn_like(target_pattern) * error_scale
                predictions[b, t, c] = target_pattern + noise
    
    return predictions, targets


def test_basic_temporal_metrics():
    """测试基础时序指标计算"""
    logger.info("🧪 测试基础时序指标计算...")
    
    try:
        calculator = TemporalMetricsCalculator(image_size=(64, 64))
        
        # 创建测试数据
        predictions, targets = create_synthetic_temporal_data(
            batch_size=2, time_steps=5, channels=1, height=64, width=64
        )
        
        # 计算指标
        metrics = calculator.compute_temporal_metrics(predictions, targets, inference_time=0.1)
        
        # 验证指标
        assert isinstance(metrics, TemporalMetrics)
        assert metrics.rel2_mean > 0
        assert metrics.rel2_last > 0
        assert metrics.rel2_worst >= metrics.rel2_mean
        assert len(metrics.step_wise_rel2) == 5
        assert len(metrics.step_wise_mae) == 5
        assert metrics.inference_latency_ms == 100.0  # 0.1s = 100ms
        assert metrics.per_step_latency_ms == 20.0    # 100ms / 5 steps
        
        logger.info("✅ 基础时序指标计算测试通过")
        logger.info(f"  Rel-L2 平均: {metrics.rel2_mean:.6f}")
        logger.info(f"  Rel-L2 最后: {metrics.rel2_last:.6f}")
        logger.info(f"  Rel-L2 最差: {metrics.rel2_worst:.6f}")
        logger.info(f"  时序一致性: {metrics.temporal_consistency:.4f}")
        logger.info(f"  误差累积率: {metrics.error_accumulation_rate:.6f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 基础时序指标计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_statistics():
    """测试批量统计功能"""
    logger.info("🧪 测试批量统计功能...")
    
    try:
        calculator = TemporalMetricsCalculator(image_size=(64, 64))
        
        # 创建多个样本的指标
        metrics_list = []
        for i in range(5):
            predictions, targets = create_synthetic_temporal_data(
                batch_size=1, time_steps=8, channels=1, height=64, width=64
            )
            metrics = calculator.compute_temporal_metrics(predictions, targets, inference_time=0.05)
            metrics_list.append(metrics)
        
        # 计算批量统计
        statistics = calculator.compute_batch_statistics(metrics_list)
        
        # 验证统计信息
        assert 'rel2_mean' in statistics
        assert 'rel2_last' in statistics
        assert 'rel2_worst' in statistics
        assert 'temporal_consistency' in statistics
        
        for metric_name, stats in statistics.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert stats['min'] <= stats['mean'] <= stats['max']
        
        logger.info("✅ 批量统计功能测试通过")
        logger.info(f"  样本数量: {len(metrics_list)}")
        logger.info(f"  Rel-L2平均统计: {statistics['rel2_mean']['mean']:.6f}±{statistics['rel2_mean']['std']:.6f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 批量统计功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_report():
    """测试指标报告生成"""
    logger.info("🧪 测试指标报告生成...")
    
    try:
        calculator = TemporalMetricsCalculator(image_size=(64, 64))
        
        # 创建测试数据
        predictions, targets = create_synthetic_temporal_data(
            batch_size=2, time_steps=6, channels=1, height=64, width=64
        )
        
        # 计算指标
        metrics = calculator.compute_temporal_metrics(predictions, targets, inference_time=0.08)
        
        # 生成报告
        report = calculator.format_metrics_report(metrics)
        
        # 验证报告内容
        assert "时序AR模型评估报告" in report
        assert "多步Rel-L2指标" in report
        assert "多步MAE指标" in report
        assert "时序特性" in report
        assert "性能统计" in report
        assert "步骤详情" in report
        
        logger.info("✅ 指标报告生成测试通过")
        logger.info("📄 生成的报告:")
        print(report)
        
        return True
    except Exception as e:
        logger.error(f"❌ 指标报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_accumulation():
    """测试误差累积率计算"""
    logger.info("🧪 测试误差累积率计算...")
    
    try:
        calculator = TemporalMetricsCalculator(image_size=(64, 64))
        
        # 创建具有明显误差累积的数据
        batch_size, time_steps, channels, height, width = 2, 10, 1, 64, 64
        predictions = torch.zeros(batch_size, time_steps, channels, height, width)
        targets = torch.zeros(batch_size, time_steps, channels, height, width)
        
        for b in range(batch_size):
            for t in range(time_steps):
                for c in range(channels):
                    # 目标保持相对稳定
                    targets[b, t, c] = torch.ones(height, width)
                    
                    # 预测误差随时间线性增长
                    error_magnitude = 0.1 * (t + 1)
                    predictions[b, t, c] = torch.ones(height, width) + error_magnitude
        
        # 计算指标
        metrics = calculator.compute_temporal_metrics(predictions, targets)
        
        # 验证误差累积率为正（误差确实在增长）
        assert metrics.error_accumulation_rate > 0, f"误差累积率应为正值，实际为: {metrics.error_accumulation_rate}"
        
        # 验证最差步骤确实比平均步骤差
        assert metrics.rel2_worst > metrics.rel2_mean, "最差步骤应比平均步骤差"
        
        logger.info("✅ 误差累积率计算测试通过")
        logger.info(f"  误差累积率: {metrics.error_accumulation_rate:.6f}")
        logger.info(f"  步骤误差趋势: {metrics.step_wise_rel2}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 误差累积率计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_consistency():
    """测试时序一致性计算"""
    logger.info("🧪 测试时序一致性计算...")
    
    try:
        calculator = TemporalMetricsCalculator(image_size=(64, 64))
        
        # 测试1: 完全一致的序列
        batch_size, time_steps, channels, height, width = 1, 5, 1, 64, 64
        consistent_pred = torch.ones(batch_size, time_steps, channels, height, width)
        consistent_target = torch.ones(batch_size, time_steps, channels, height, width)
        
        metrics1 = calculator.compute_temporal_metrics(consistent_pred, consistent_target)
        
        # 测试2: 不一致的序列
        inconsistent_pred = torch.zeros(batch_size, time_steps, channels, height, width)
        for t in range(time_steps):
            inconsistent_pred[0, t, 0] = torch.randn(height, width) * (t + 1)  # 随时间变化更大
        
        metrics2 = calculator.compute_temporal_metrics(inconsistent_pred, consistent_target)
        
        # 验证一致序列的时序一致性更高
        assert metrics1.temporal_consistency > metrics2.temporal_consistency, \
            f"一致序列应有更高的时序一致性: {metrics1.temporal_consistency} vs {metrics2.temporal_consistency}"
        
        logger.info("✅ 时序一致性计算测试通过")
        logger.info(f"  一致序列时序一致性: {metrics1.temporal_consistency:.4f}")
        logger.info(f"  不一致序列时序一致性: {metrics2.temporal_consistency:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 时序一致性计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("🚀 开始时序指标计算功能测试...")
    
    tests = [
        ("基础时序指标计算", test_basic_temporal_metrics),
        ("批量统计功能", test_batch_statistics),
        ("指标报告生成", test_metrics_report),
        ("误差累积率计算", test_error_accumulation),
        ("时序一致性计算", test_temporal_consistency),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        success = test_func()
        results[test_name] = success
        
        if success:
            logger.info(f"✅ {test_name} 测试通过")
        else:
            logger.error(f"❌ {test_name} 测试失败")
    
    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed < total:
        logger.warning(f"⚠️ {total - passed} 个测试失败")
    else:
        logger.info("🎉 所有测试通过！")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)