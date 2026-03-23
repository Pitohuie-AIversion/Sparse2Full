#!/usr/bin/env python3
"""
缓存预热器（Cache Warmer）

目的：
- 通过顺序/并行读取 HDF5 数据文件来预热操作系统页缓存，从而加速内存填充（提升 `used`/`buff/cache`）。
- 不修改正在运行的训练进程的 DataLoader 配置（prefetch_factor / batch_size），属于轻量级外部加速。

使用：
- python tools/monitor/cache_warmer.py --exp runs/AR-DR2D-Debug-SwinUNet-s42 --workers 8
- 或显式指定数据路径：--data-path /path/to/2D_diff-react_NA_NA.h5

日志：
- 输出到 runs/<exp>/cache_warmer.log，记录内存使用、读取进度与速率估计。

注意：
- 每个 worker 在独立进程内单独打开 HDF5 文件，避免句柄共享问题。
- 默认只依赖 OS 页缓存，不持有大量 Python 数组，防止占用训练所需可用内存。
"""

import argparse
import os
import sys
import time
import math
import json
from datetime import datetime
from multiprocessing import Process, Queue

try:
    import h5py  # type: ignore
except Exception as e:
    print(f"ERROR: h5py import failed: {e}")
    sys.exit(1)


def read_meminfo():
    """Parse /proc/meminfo and return used, buff/cache, available in GB."""
    mem = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for ln in f:
                parts = ln.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                try:
                    mem[key] = float(val)  # kB
                except Exception:
                    pass
        mem_total = mem.get("MemTotal", 0.0)
        mem_free = mem.get("MemFree", 0.0)
        mem_buff = mem.get("Buffers", 0.0)
        mem_cache = mem.get("Cached", 0.0) + mem.get("SReclaimable", 0.0)
        mem_available = mem.get("MemAvailable", 0.0)
        used_kb = mem_total - mem_free - mem_buff - mem_cache
        return round(used_kb / (1024**2), 2), round((mem_buff + mem_cache) / (1024**2), 2), round(mem_available / (1024**2), 2)
    except Exception:
        return None, None, None


def parse_config_data_path(exp_path: str) -> str:
    """Try to read runs/<exp>/config_merged.yaml and extract data path."""
    cfg_path = os.path.join(exp_path, "config_merged.yaml")
    if not os.path.isfile(cfg_path):
        return ""
    try:
        # Minimal YAML parsing without dependencies
        data_path = ""
        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln_stripped = ln.strip()
                if ln_stripped.startswith("data_path:"):
                    # data_path: /path/to/file
                    data_path = ln_stripped.split(":", 1)[1].strip()
                    # remove quotes if any
                    if data_path.startswith("'") and data_path.endswith("'"):
                        data_path = data_path[1:-1]
                    if data_path.startswith('"') and data_path.endswith('"'):
                        data_path = data_path[1:-1]
                    break
        return data_path
    except Exception:
        return ""


def list_time_keys(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        return sorted([k for k in f.keys()])


def worker_read(h5_path: str, keys: list[str], progress: Queue, sleep_ms: int = 0):
    """Worker: open file and sequentially read datasets to warm OS cache."""
    try:
        with h5py.File(h5_path, "r") as f:
            for k in keys:
                if k not in f:
                    continue
                grp = f[k]
                if "data" not in grp:
                    continue
                ds = grp["data"]  # shape (N, H, W, C)
                # Read per-sample slices to avoid large allocations
                n = ds.shape[0]
                for i in range(n):
                    _ = ds[i]
                    # Optionally throttle
                    if sleep_ms > 0:
                        time.sleep(sleep_ms / 1000.0)
                progress.put({"key": k, "samples": n})
    except Exception as e:
        progress.put({"error": str(e)})


def chunk_keys(keys: list[str], n_chunks: int) -> list[list[str]]:
    if n_chunks <= 1:
        return [keys]
    chunks = [[] for _ in range(n_chunks)]
    for idx, k in enumerate(keys):
        chunks[idx % n_chunks].append(k)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="runs/AR-DR2D-Debug-SwinUNet-s42", help="实验目录（包含config_merged.yaml）")
    ap.add_argument("--data-path", type=str, default="", help="HDF5 数据路径（优先，如果为空则从exp解析）")
    ap.add_argument("--workers", type=int, default=8, help="并行读取的进程数")
    ap.add_argument("--sleep-ms", type=int, default=0, help="每样本读取间的微暂停，避免过度竞争")
    ap.add_argument("--log-filename", type=str, default="cache_warmer.log", help="日志文件名")
    args = ap.parse_args()

    exp_path = args.exp
    os.makedirs(exp_path, exist_ok=True)
    log_path = os.path.join(exp_path, args.log_filename)

    data_path = args.data_path or parse_config_data_path(exp_path)
    if not data_path:
        print("ERROR: 无法解析数据路径，请通过 --data-path 指定或确保配置存在")
        sys.exit(1)
    if not os.path.isfile(data_path):
        print(f"ERROR: 数据文件不存在: {data_path}")
        sys.exit(1)

    # Discover keys
    keys = list_time_keys(data_path)
    if not keys:
        print("ERROR: 文件中未发现时间组键")
        sys.exit(1)

    # Prepare progress queue
    progress: Queue = Queue()
    chunks = chunk_keys(keys, max(1, args.workers))
    procs: list[Process] = []

    # Start workers
    for idx, c in enumerate(chunks):
        p = Process(target=worker_read, args=(data_path, c, progress, args.sleep_ms), daemon=True)
        p.start()
        procs.append(p)

    start_ts = time.time()
    read_count = 0
    last_log = None

    with open(log_path, "a", encoding="utf-8") as lf:
        while any(p.is_alive() for p in procs) or not progress.empty():
            try:
                msg = progress.get(timeout=1.0)
            except Exception:
                msg = None
            if msg:
                if "error" in msg:
                    lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {msg['error']}\n")
                    lf.flush()
                    continue
                read_count += int(msg.get("samples", 0))
            # Periodic status
            now = time.time()
            if last_log is None or now - last_log >= 5.0:
                used, buff, avail = read_meminfo()
                dt_min = max((now - start_ts) / 60.0, 1e-9)
                rate = None
                if used is not None:
                    # 无法直接计算增量，因为只读 OS 缓存；仍然记录当前 used 与 buff/cache
                    rate = None
                lf.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] warmed_samples={read_count} | used={used}GB | buff/cache={buff}GB | avail={avail}GB | workers={len(procs)}\n"
                )
                lf.flush()
                last_log = now

    # Final summary
    used, buff, avail = read_meminfo()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE warmed_samples={read_count} | used={used}GB | buff/cache={buff}GB | avail={avail}GB\n"
        )
        lf.flush()

    print(json.dumps({
        "status": "done",
        "warmed_samples": read_count,
        "used_gb": used,
        "buff_cache_gb": buff,
        "available_gb": avail,
        "log_path": log_path,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()