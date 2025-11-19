"""
测试 PDEBenchVisualizer 占位实现是否可用：
- 能被成功导入与实例化
- 在无外部依赖文件的情况下可以创建HTML报告占位
"""

import tempfile
from pathlib import Path

def test_pdebench_visualizer_basic_creation():
    from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = PDEBenchVisualizer(output_dir=tmpdir)
        assert Path(tmpdir).exists()

def test_pdebench_visualizer_create_report_without_dependencies():
    from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        out_dir = Path(tmpdir) / "viz"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 不提供训练日志/历史，报告生成应当稳健降级并成功创建HTML占位
        viz = PDEBenchVisualizer(output_dir=str(out_dir))
        ok = viz.create_comprehensive_report(str(run_dir))
        # 即使没有数据，函数也应返回True并生成一个占位HTML
        assert ok
        report = Path(out_dir) / "report.html"
        assert report.exists()