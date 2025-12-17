import json
from pathlib import Path
import numpy as np

def test_eval_files_exist_after_training_run():
    # 取最近一次SwinTiny运行目录
    runs_dir = Path('runs')
    candidates = sorted([p for p in runs_dir.glob('*model_SwinTransformerTiny_*') if p.is_dir()])
    assert len(candidates) > 0
    run_dir = candidates[-1]
    eval_dir = run_dir / 'eval'
    metrics_file = eval_dir / 'metrics.jsonl'
    summary_file = eval_dir / 'summary_stats.json'
    results_md = eval_dir / 'results_table.md'
    assert metrics_file.exists(), 'metrics.jsonl not found'
    assert summary_file.exists(), 'summary_stats.json not found'
    assert results_md.exists(), 'results_table.md not found'
    # 简单内容校验
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) >= 1
    first = json.loads(lines[0])
    for k in ['rel_l2','mae','mse','psnr']:
        assert k in first

