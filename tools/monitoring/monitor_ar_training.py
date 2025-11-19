#!/usr/bin/env python3
"""
AR训练进度监控脚本
- 显示当前训练进度（epoch/batch）、损失曲线、验证指标
- 检查训练日志中的错误或警告
- 生成简洁训练状态报告（JSON/Markdown）
- 监控GPU/内存使用情况

用法：
  python tools/monitoring/monitor_ar_training.py --runs_dir runs --filter "AR-DR2D-Debug-SwinUNet-s2025"

脚本会自动选择最新的匹配run目录（包含training.log/training_history.json）。
"""
import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import psutil

# 可选GPU监控：pynvml优先，失败则调用nvidia-smi
def get_gpu_stats() -> Dict[str, Any]:
    stats = {"available": False, "devices": []}
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            stats["devices"].append({
                "index": i,
                "name": name,
                "memory_used_mb": mem.used / (1024*1024),
                "memory_total_mb": mem.total / (1024*1024),
                "gpu_util_percent": util.gpu,
                "mem_util_percent": util.memory
            })
        pynvml.nvmlShutdown()
        stats["available"] = device_count > 0
        return stats
    except Exception:
        # 尝试nvidia-smi
        try:
            import subprocess
            cmd = [
                'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ]
            out = subprocess.check_output(cmd).decode('utf-8').strip().splitlines()
            for line in out:
                parts = [p.strip() for p in line.split(',')]
                stats["devices"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "gpu_util_percent": float(parts[4]),
                    "mem_util_percent": None
                })
            stats["available"] = len(stats["devices"]) > 0
        except Exception:
            pass
        return stats


def find_latest_run(runs_dir: Path, filter_str: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if filter_str and filter_str not in p.name:
            continue
        log_path = p / 'training.log'
        hist_path = p / 'training_history.json'
        if log_path.exists() or hist_path.exists():
            candidates.append(p)
    if not candidates:
        return None
    # 选择最近修改的
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_log(log_path: Path) -> Dict[str, Any]:
    result = {
        "last_epoch_summary": None,
        "warnings": [],
        "errors": []
    }
    if not log_path.exists():
        return result
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 匹配Epoch行，例如：
        # Epoch   5/1000 | Train Loss: 41.034376 | Val Loss: 0.322266 | Best: 0.322266 | Time: 338.0s
        epoch_re = re.compile(r"Epoch\s+(\d+)\/(\d+)\s*\|\s*Train Loss:\s*([0-9\.]+)\s*\|\s*Val Loss:\s*([0-9\.]+)\s*\|\s*Best:\s*([0-9\.]+)")
        for line in reversed(lines):
            m = epoch_re.search(line)
            if m:
                result["last_epoch_summary"] = {
                    "epoch": int(m.group(1)),
                    "max_epochs": int(m.group(2)),
                    "train_loss": float(m.group(3)),
                    "val_loss": float(m.group(4)),
                    "best_val_loss": float(m.group(5))
                }
                break
        # 收集最近的WARN/ERROR
        for line in lines[-500:]:
            if 'WARNING' in line or '⚠️' in line:
                result["warnings"].append(line.strip())
            if 'ERROR' in line or '❌' in line:
                result["errors"].append(line.strip())
    except Exception as e:
        result["errors"].append(f"log parse failed: {e}")
    return result


def parse_history(history_path: Path) -> Dict[str, Any]:
    result = {
        "available": False,
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "epochs": [],
        "last_metrics": {},
    }
    if not history_path.exists():
        return result
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            hist = json.load(f)
        result["available"] = True
        result["train_losses"] = hist.get('train_losses', [])
        result["val_losses"] = hist.get('val_losses', [])
        result["learning_rates"] = hist.get('learning_rates', [])
        result["epochs"] = hist.get('epochs', [])
        val_metrics = hist.get('val_metrics', [])
        if isinstance(val_metrics, list) and val_metrics:
            # 取最后一组指标
            last = val_metrics[-1]
            # 转换可能的numpy数值
            result["last_metrics"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in last.items()}
    except Exception as e:
        result["available"] = False
        result.setdefault("errors", []).append(f"history parse failed: {e}")
    return result


def create_report(run_dir: Path, log_info: Dict[str, Any], hist_info: Dict[str, Any], gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.5)
    vm = psutil.virtual_memory()
    mem_percent = vm.percent
    mem_used_gb = (vm.used / (1024**3))

    report = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "run_dir": str(run_dir),
        "progress": log_info.get("last_epoch_summary"),
        "history": {
            "train_last": hist_info.get("train_losses", [])[-1] if hist_info.get("train_losses") else None,
            "val_last": hist_info.get("val_losses", [])[-1] if hist_info.get("val_losses") else None,
            "lr_last": hist_info.get("learning_rates", [])[-1] if hist_info.get("learning_rates") else None,
            "epochs_count": len(hist_info.get("epochs", [])),
            "last_metrics": hist_info.get("last_metrics", {})
        },
        "resources": {
            "cpu_percent": cpu,
            "mem_used_gb": mem_used_gb,
            "mem_percent": mem_percent,
            "gpu": gpu_info
        },
        "alerts": {
            "warnings": log_info.get("warnings", []),
            "errors": log_info.get("errors", [])
        }
    }
    return report


def save_report(run_dir: Path, report: Dict[str, Any]) -> Path:
    out_json = run_dir / 'monitoring_report.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {out_json}")
    # 同步输出简洁Markdown
    md = [
        f"# 训练状态报告 ({report['timestamp']})",
        f"- 目录: `{run_dir}`",
    ]
    prog = report.get('progress') or {}
    if prog:
        md += [
            f"- 进度: Epoch {prog.get('epoch')}/{prog.get('max_epochs')}",
            f"- Train Loss: {prog.get('train_loss')}",
            f"- Val Loss: {prog.get('val_loss')} (Best: {prog.get('best_val_loss')})",
        ]
    hist = report.get('history') or {}
    md += [
        f"- 最近学习率: {hist.get('lr_last')}",
        f"- 最近验证指标: {hist.get('last_metrics')}",
    ]
    alerts = report.get('alerts') or {}
    if alerts.get('warnings'):
        md.append(f"- 警告数: {len(alerts['warnings'])}")
    if alerts.get('errors'):
        md.append(f"- 错误数: {len(alerts['errors'])}")
    gpu = report['resources'].get('gpu', {})
    if gpu.get('available'):
        for dev in gpu.get('devices', []):
            md.append(f"- GPU{dev['index']} {dev['name']}: {dev['memory_used_mb']:.0f}/{dev['memory_total_mb']:.0f}MB, util {dev['gpu_util_percent']}%")
    md_path = run_dir / 'monitoring_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    return out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', type=str, default='runs')
    ap.add_argument('--filter', type=str, default=None, help='过滤实验名前缀，例如 AR-DR2D-Debug-SwinUNet-s2025')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs目录不存在: {runs_dir}")
        return

    run_dir = find_latest_run(runs_dir, args.filter)
    if run_dir is None:
        print(f"未找到匹配的run目录（filter={args.filter}）")
        return

    log_info = parse_log(run_dir / 'training.log')
    hist_info = parse_history(run_dir / 'training_history.json')
    gpu_info = get_gpu_stats()
    report = create_report(run_dir, log_info, hist_info, gpu_info)
    out_json = save_report(run_dir, report)

    # 控制台摘要
    print(json.dumps(report, ensure_ascii=False, indent=2))