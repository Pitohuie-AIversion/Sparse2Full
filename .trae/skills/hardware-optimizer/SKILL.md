---
name: hardware-optimizer
description: Tools for monitoring, managing, and optimizing computer hardware resources (GPU VRAM, CPU, System Info) for Deep Learning tasks. Use when user asks to check GPU status, clear VRAM, debug OOM errors, or optimize hardware performance.
---

# Hardware Optimizer

This skill provides tools and guides for managing and optimizing hardware resources, specifically focusing on NVIDIA GPUs in a Deep Learning environment.

## Capabilities

- **Monitor GPU**: Track VRAM usage and utilization over time.
- **Clear VRAM**: Safe utilities to release GPU memory.
- **System Info**: Detailed report of CPU, RAM, and GPU topology.
- **Troubleshooting**: Guide for common hardware-related training issues.

## Tools & Scripts

### 1. Monitor GPU Usage
Use `scripts/gpu_monitor.py` to watch GPU status continuously. This is better than `nvidia-smi` for tracking usage spikes during training.

```bash
# Monitor for 60 seconds (default)
python scripts/gpu_monitor.py

# Monitor for 5 minutes
python scripts/gpu_monitor.py 300
```

### 2. Clear GPU Cache
Use `scripts/clear_cache.py` to aggressively release reserved memory.

```bash
python scripts/clear_cache.py
```

### 3. Get System Report
Use `scripts/system_info.py` to get a full snapshot of the hardware environment.

```bash
python scripts/system_info.py
```

## Reference Guides

- **Troubleshooting**: See [troubleshooting.md](references/troubleshooting.md) for OOM solutions and performance tuning.
