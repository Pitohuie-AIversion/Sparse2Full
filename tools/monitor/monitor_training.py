#!/usr/bin/env python3
import argparse
import os
import time
import subprocess
from datetime import datetime
from typing import Optional, Tuple


def sh(cmd: str) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return p.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")


def find_pid(default_pid: Optional[int]) -> Optional[int]:
    if default_pid and os.path.exists(f"/proc/{default_pid}"):
        return default_pid
    # fallback: newest matching training script
    code, out, _ = sh("pgrep -n -f 'tools/training/train_real_data_ar.py'")
    if code == 0:
        out = out.strip()
        return int(out) if out else None
    return None


def get_system_mem_gb() -> Tuple[float, float, float]:
    # used, buff/cache, available in GB
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
    # fallback to MB for better accuracy
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
                # per vmstat man: last two columns are id (cpu idle) and wa (iowait), order may vary by distro
                # On many systems: ... id wa
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
    for ln in out.splitlines()[::-1]:  # scan from end
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
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--samples", type=int, default=0, help="0=无限循环")
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--target_gb", type=float, default=800.0)
    args = ap.parse_args()

    pid = find_pid(args.pid)
    print(f"PID: {pid if pid else '未找到'}")
    used_hist = []  # list of (ts, used_gb)
    milestone_400 = False
    milestone_800 = False

    n = 0
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        used, buffcache, avail = get_system_mem_gb()
        io_wait, cpu_idle = get_vmstat()

        epoch_iter, throughput, fetch_t, compute_t = parse_log(args.log)

        rss = get_proc_rss_gb(pid) if pid else None
        pcpu = get_proc_cpu(pid) if pid else None
        children = get_children_count(pid) if pid else 0
        h5fds = count_h5_fds(pid) if pid else 0

        used_hist.append((time.time(), used))
        # keep last 10 samples for rate
        if len(used_hist) > 10:
            used_hist = used_hist[-10:]
        rate_gb_per_min = None
        eta_min = None
        if len(used_hist) >= 2:
            t0, u0 = used_hist[0]
            t1, u1 = used_hist[-1]
            dt_min = max((t1 - t0) / 60.0, 1e-6)
            rate_gb_per_min = (u1 - u0) / dt_min
            if rate_gb_per_min and rate_gb_per_min > 0:
                eta_min = (args.target_gb - u1) / rate_gb_per_min
            else:
                eta_min = float("inf")

        rate_str = f"{rate_gb_per_min:.2f}" if rate_gb_per_min is not None else "NA"
        eta_str = f"{eta_min:.1f}" if (eta_min is not None and eta_min != float("inf")) else ("inf" if eta_min == float("inf") else "NA")
        rss_str = f"{rss:.3f}" if rss is not None else "NA"
        pcpu_str = f"{pcpu:.2f}" if pcpu is not None else "NA"
        iowait_str = f"{io_wait:.2f}" if io_wait is not None else "NA"
        idle_str = f"{cpu_idle:.2f}" if cpu_idle is not None else "NA"
        line = (
            f"[{ts}] used={used:.2f}GB buff/cache={buffcache:.2f}GB avail={avail:.2f}GB "
            f"iowait={iowait_str}% idle={idle_str}% "
            f"rss={rss_str}GB pcpu={pcpu_str}% children={children} h5_fds={h5fds} "
            f"rate={rate_str}GB/min eta800={eta_str}min"
        )
        print(line)
        # 里程碑标注
        if not milestone_400 and used >= 400.0:
            print(f"[MILESTONE] 400GB 达成: {ts}")
            milestone_400 = True
        if not milestone_800 and used >= 800.0:
            print(f"[MILESTONE] 800GB 达成: {ts}")
            milestone_800 = True
        if epoch_iter or throughput or fetch_t or compute_t:
            print(
                f"log: epoch_iter={epoch_iter or '-'} | throughput={throughput or '-'} | fetch={fetch_t or '-'} | compute={compute_t or '-'}"
            )
        else:
            print("log: 无Epoch/Iter/吞吐量/fetch/compute信息")

        n += 1
        if args.samples and n >= args.samples:
            break
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()