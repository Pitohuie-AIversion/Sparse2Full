#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from datetime import datetime
from typing import Optional, Tuple, List


def sh(cmd: str) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return p.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")


def find_pid(default_pid: Optional[int]) -> Optional[int]:
    if default_pid and os.path.exists(f"/proc/{default_pid}"):
        return default_pid
    code, out, _ = sh("pgrep -n -f 'tools/training/train_real_data_ar.py'")
    if code == 0:
        out = out.strip()
        return int(out) if out else None
    return None


def get_gpu_info() -> List[Tuple[int, float, float, float]]:
    code, out, _ = sh(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    )
    infos: List[Tuple[int, float, float, float]] = []
    if code == 0:
        for ln in out.strip().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    used_gb = float(parts[1]) / 1024.0
                    total_gb = float(parts[2]) / 1024.0
                    util = float(parts[3])
                    infos.append((idx, used_gb, total_gb, util))
                except Exception:
                    continue
    return infos


def get_proc_gpu_mem(pid: int) -> Optional[float]:
    code, out, _ = sh(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
    )
    if code == 0:
        total_mb = 0.0
        for ln in out.strip().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2:
                try:
                    proc = int(parts[0])
                    if proc == pid:
                        mb = float(parts[1])
                        total_mb += mb
                except Exception:
                    continue
        return total_mb / 1024.0 if total_mb > 0 else 0.0
    return None


def get_system_mem_gb() -> Tuple[float, float, float]:
    code, out, _ = sh("free -g")
    if code == 0:
        lines = out.strip().splitlines()
        for ln in lines:
            if ln.lower().startswith("mem"):
                parts = ln.split()
                if len(parts) >= 7:
                    used = float(parts[2])
                    buffcache = float(parts[5])
                    avail = float(parts[6])
                    return used, buffcache, avail
    code, out, _ = sh("free -m")
    if code == 0:
        lines = out.strip().splitlines()
        for ln in lines:
            if ln.lower().startswith("mem"):
                parts = ln.split()
                if len(parts) >= 7:
                    used = float(parts[2]) / 1024.0
                    buffcache = float(parts[5]) / 1024.0
                    avail = float(parts[6]) / 1024.0
                    return used, buffcache, avail
    return 0.0, 0.0, 0.0


def get_vmstat() -> Tuple[Optional[float], Optional[float]]:
    code, out, _ = sh("vmstat 1 2")
    if code == 0:
        lines = out.strip().splitlines()
        if lines:
            parts = lines[-1].split()
            try:
                cpu_idle = float(parts[-2])
                io_wait = float(parts[-1])
                return io_wait, cpu_idle
            except Exception:
                pass
    return None, None


def get_proc_rss_gb(pid: int) -> Optional[float]:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    kb = float(ln.split()[1])
                    return kb / 1024.0 / 1024.0
    except Exception:
        return None
    return None


def get_proc_cpu(pid: int) -> Optional[float]:
    code, out, _ = sh(f"ps -p {pid} -o %cpu=")
    if code == 0:
        try:
            return float(out.strip())
        except Exception:
            return None
    return None


def get_children_count(pid: int) -> int:
    code, out, _ = sh(f"pgrep -P {pid}")
    if code == 0 and out.strip():
        return len(out.strip().splitlines())
    return 0


def count_h5_fds(pid: int) -> int:
    base = f"/proc/{pid}/fd"
    try:
        names = os.listdir(base)
    except Exception:
        return 0
    cnt = 0
    for n in names:
        p = os.path.join(base, n)
        try:
            target = os.readlink(p)
            if ".h5" in target.lower():
                cnt += 1
        except Exception:
            continue
    return cnt


def parse_log(log_path: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not log_path or not os.path.isfile(log_path):
        return None, None, None, None
    code, out, _ = sh(f"tail -n 200 '{log_path}'")
    if code != 0:
        return None, None, None, None
    epoch_iter = None
    throughput = None
    fetch_t = None
    compute_t = None
    for ln in out.splitlines()[::-1]:
        s = ln.strip()
        if epoch_iter is None and ("Epoch" in s or "Iter" in s):
            epoch_iter = s
        if throughput is None and ("Throughput" in s or "samples/s" in s):
            throughput = s
        if fetch_t is None and ("fetch" in s.lower()):
            fetch_t = s
        if compute_t is None and ("compute" in s.lower()):
            compute_t = s
        if epoch_iter and throughput and fetch_t and compute_t:
            break
    return epoch_iter, throughput, fetch_t, compute_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--log", type=str, default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid = find_pid(args.pid)
    used, buffcache, avail = get_system_mem_gb()
    io_wait, cpu_idle = get_vmstat()
    epoch_iter, throughput, fetch_t, compute_t = parse_log(args.log)

    rss = get_proc_rss_gb(pid) if pid else None
    pcpu = get_proc_cpu(pid) if pid else None
    children = get_children_count(pid) if pid else 0
    h5fds = count_h5_fds(pid) if pid else 0
    gpu_infos = get_gpu_info()
    proc_gpu = get_proc_gpu_mem(pid) if pid else None

    print(f"[{ts}] PID={pid if pid else 'NA'}")
    print(
        f"MEM used={used:.2f}GB buff/cache={buffcache:.2f}GB avail={avail:.2f}GB iowait={io_wait if io_wait is not None else 'NA'}% idle={cpu_idle if cpu_idle is not None else 'NA'}%"
    )
    if pid:
        print(
            f"PROC rss={rss if rss is not None else 'NA'}GB pcpu={pcpu if pcpu is not None else 'NA'}% children={children} h5_fds={h5fds}"
        )
    if gpu_infos:
        for idx, used_gb, total_gb, util in gpu_infos:
            print(f"GPU{idx} util={util:.0f}% mem={used_gb:.2f}/{total_gb:.0f}GB")
    print(f"PROC_GPU_MEM {proc_gpu if proc_gpu is not None else 'NA'}GB")
    if any([epoch_iter, throughput, fetch_t, compute_t]):
        print(
            f"LOG epoch_iter={epoch_iter or '-'} | throughput={throughput or '-'} | fetch={fetch_t or '-'} | compute={compute_t or '-'}"
        )


if __name__ == "__main__":
    main()

