# Hardware Optimization & Troubleshooting Guide

## Common GPU Issues

### CUDA Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory.`

**Solutions:**
1. **Reduce Batch Size:** The most direct fix. Halve it and retry.
2. **Gradient Accumulation:** If batch size is too small for convergence, accumulate gradients over multiple steps.
3. **Mixed Precision (AMP):** Use `torch.cuda.amp.autocast()` to reduce memory usage (fp16 instead of fp32).
4. **Gradient Checkpointing:** `torch.utils.checkpoint` trades compute for memory by recomputing activations during backward pass.
5. **Clear Cache:** Run `scripts/clear_cache.py` or `torch.cuda.empty_cache()` manually (though PyTorch does this automatically usually).
6. **Check for Memory Leaks:** Ensure tensors are not being accumulated in a list (e.g., `losses.append(loss)` keeps the graph; use `losses.append(loss.item())`).

### GPU Low Utilization

**Symptoms:**
- `nvidia-smi` shows low volatile GPU-Util (e.g., < 50%) while training.

**Causes & Solutions:**
1. **Data Loading Bottleneck:** CPU can't feed GPU fast enough.
   - Increase `num_workers` in `DataLoader`.
   - Use `pin_memory=True` in `DataLoader`.
   - Optimize dataset `__getitem__` (avoid heavy processing on the fly).
2. **Small Batch Size:** GPU kernels launch overhead dominates. Increase batch size.
3. **IO Bottleneck:** Reading from slow disk (HDD vs NVMe). Cache data if possible.

### Slow Training

**Checks:**
- Check CPU usage (`top` / `htop`). If one core is 100%, likely Python GIL or single-threaded data processing.
- Check Throttle: Is GPU thermal throttling? (Check temperature in `nvidia-smi`).

## Hardware Optimization Tips

1. **TF32 on Ampere+ (A100, 3090, 4090, L40):**
   - Enabled by default in PyTorch, but ensure `torch.backends.cuda.matmul.allow_tf32 = True`.
   - Provides near-fp32 accuracy with significant speedup.

2. **Benchmarking:**
   - Use `torch.profiler` to identify which operators are slow.

3. **Multi-GPU:**
   - Use `DistributedDataParallel` (DDP) instead of `DataParallel` (DP) for better performance and multi-node support.
