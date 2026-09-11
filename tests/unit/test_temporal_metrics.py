import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from utils.temporal_metrics import TemporalMetricsCalculator, TemporalMetrics
from utils.temporal_visualization import TemporalVisualizer, VisualizationConfig


class TestTemporalMetricsCalculator:
    """测试时序指标计算器"""

    @pytest.fixture
    def calculator(self):
        return TemporalMetricsCalculator(image_size=(32, 32))

    def test_compute_temporal_metrics_shape_validation(self, calculator):
        """测试 5D 输入形状校验"""
        with pytest.raises(ValueError, match="5D"):
            calculator.compute_temporal_metrics(
                torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
            )

    def test_compute_temporal_metrics_perfect_prediction(self, calculator):
        """测试完全相同预测时的理想指标"""
        target = torch.randn(2, 5, 1, 32, 32)
        pred = target.clone()

        metrics = calculator.compute_temporal_metrics(pred, target, inference_time=0.05)

        assert isinstance(metrics, TemporalMetrics)
        assert metrics.rel2_mean == pytest.approx(0.0, abs=1e-5)
        assert metrics.rel2_last == pytest.approx(0.0, abs=1e-5)
        assert metrics.mae_mean == pytest.approx(0.0, abs=1e-5)
        assert metrics.inference_latency_ms == pytest.approx(50.0, abs=1e-2)
        assert metrics.per_step_latency_ms == pytest.approx(10.0, abs=1e-2)
        assert len(metrics.step_wise_rel2) == 5
        assert len(metrics.step_wise_mae) == 5

    def test_temporal_consistency_and_accumulation(self, calculator):
        """测试时序一致性与累积误差计算"""
        B, T, C, H, W = 2, 6, 1, 32, 32
        # 构造线性增长误差
        target = torch.ones(B, T, C, H, W)
        pred = torch.zeros(B, T, C, H, W)
        for t in range(T):
            pred[:, t] = target[:, t] + (t + 1) * 0.1

        metrics = calculator.compute_temporal_metrics(pred, target)
        assert 0.0 <= metrics.temporal_consistency <= 1.0
        assert np.isfinite(metrics.error_accumulation_rate)

    def test_batch_statistics(self, calculator):
        """测试批量统计聚合"""
        target = torch.randn(2, 4, 1, 32, 32)
        m1 = calculator.compute_temporal_metrics(target, target)
        m2 = calculator.compute_temporal_metrics(target + 0.1, target)

        stats = calculator.compute_batch_statistics([m1, m2])
        assert "rel2_mean" in stats
        assert "mean" in stats["rel2_mean"]
        assert "std" in stats["rel2_mean"]
        assert stats["rel2_mean"]["min"] <= stats["rel2_mean"]["max"]

    def test_format_metrics_report(self, calculator):
        """测试报告格式化生成"""
        target = torch.randn(2, 4, 1, 32, 32)
        metrics = calculator.compute_temporal_metrics(target, target)
        report = calculator.format_metrics_report(metrics)
        assert "时序AR模型评估报告" in report
        assert "平均Rel-L2" in report


class TestTemporalVisualizer:
    """测试时序可视化器"""

    @pytest.fixture
    def visualizer(self):
        return TemporalVisualizer(VisualizationConfig(dpi=50))

    def test_plot_sequence_comparison(self, visualizer):
        """测试时序对比图保存"""
        B, T, C, H, W = 1, 4, 1, 16, 16
        pred = torch.randn(B, T, C, H, W)
        target = torch.randn(B, T, C, H, W)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "comparison.png"
            visualizer.plot_sequence_comparison(pred, target, save_path=out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_plot_error_evolution(self, visualizer):
        """测试误差演化曲线绘制"""
        B, T, C, H, W = 1, 5, 1, 16, 16
        pred = torch.randn(B, T, C, H, W)
        target = torch.randn(B, T, C, H, W)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "error_evolution.png"
            visualizer.plot_error_evolution(pred, target, save_path=out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0
