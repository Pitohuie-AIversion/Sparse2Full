#!/usr/bin/env python3
"""
从 runs 目录中自动查找最新的 AR 训练运行，读取 training_history.json，
使用修复后的 ARTrainingVisualizer 生成训练曲线与综合报告。
"""

import json
from pathlib import Path
from typing import Optional
import argparse
import sys

project_root = Path(__file__).resolve().parents[2]


def find_latest_run_with_history(runs_dir: Path, prefix: Optional[str] = None) -> Optional[Path]:
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue
        hist_file = d / "training_history.json"
        if hist_file.exists():
            candidates.append((d, hist_file.stat().st_mtime))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default=str(project_root / "runs"))
    ap.add_argument("--prefix", type=str, default="AR-")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs 目录不存在: {runs_dir}")
        return

    run_dir = find_latest_run_with_history(runs_dir, prefix=args.prefix)
    if run_dir is None:
        print("未找到包含 training_history.json 的运行目录")
        return

    history_file = run_dir / "training_history.json"
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    # 修正模块搜索路径后再延迟导入
    project_path = project_root
    if str(project_path) not in sys.path:
        sys.path.append(str(project_path))
    from utils.ar_visualizer import ARTrainingVisualizer

    # output_dir 设为运行目录，以便在其中创建 visualizations 子目录
    visualizer = ARTrainingVisualizer(str(run_dir))

    # 生成训练曲线
    visualizer.plot_training_curves(history, save_name="training_curves_latest")

    # 生成综合报告（包含所有已生成图片）
    report_path = visualizer.create_comprehensive_report(history)

    print("生成完成：")
    print(f"  训练曲线: {run_dir / 'visualizations' / 'training_curves' / 'training_curves_latest.png'}")
    print(f"  综合报告: {report_path}")


if __name__ == "__main__":
    main()
