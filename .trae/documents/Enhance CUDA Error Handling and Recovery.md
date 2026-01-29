# 解决方案：增强型 CUDA 错误处理与恢复机制

我已完成代码修改，实现了针对 CUDA OOM 和其他错误的健壮处理逻辑。

## 主要变更

1.  **统一清理函数 (`cleanup_cuda`)**:
    *   实现了彻底的显存清理，包括梯度清零、对象引用删除、垃圾回收 (GC) 和 `empty_cache`。
    *   支持配置 `training.debug_cuda` 进行同步调试。

2.  **增强的异常处理 (`handle_cuda_error`)**:
    *   **Fail-fast**: 对 "illegal memory access" 等致命错误，立即终止并打印堆栈，防止无限循环。
    *   **OOM 恢复**: 对 "out of memory" 错误，自动尝试降级。

3.  **OOM 降级策略 (Micro-batching)**:
    *   在 `train_epoch` 循环中，捕获 OOM 异常。
    *   自动将当前 Batch 拆分为多个 Micro-batches (默认 4 份)。
    *   使用梯度累积 (Gradient Accumulation) 完成该 Batch 的训练。
    *   若降级后仍失败，则清理并跳过该 Batch，记录跳过比例。

4.  **Smoke Test (冒烟测试)**:
    *   在正式训练前，自动运行一个 Batch 的完整前向/反向传播。
    *   输出显存峰值，用于快速检测显存泄漏或配置错误。

## 验证与使用

您可以使用以下命令验证修复效果。该命令会开启 `debug_cuda` 并进行一次冒烟测试。

**最小复现/验证命令：**

```bash
# 运行验证命令（覆盖配置以启用调试和Smoke Test）
python tools/training/train_real_data_ar.py \
    training.debug_cuda=true \
    training.smoke_test=true \
    training.oom_recovery.enabled=true \
    training.oom_recovery.microbatch_splits=4 \
    data.dataloader.batch_size=1 \
    training.epochs=1
```

**预期输出：**
*   **正常情况**: 日志显示 "🚬 Running SMOKE TEST...", 输出显存占用，然后 "✅ Smoke test passed."。
*   **OOM 情况**: 日志显示 "⚠️ CUDA OOM...", "🔄 尝试降级：Micro-batching...", 然后 "✅ Micro-batch recovery successful."。
*   **致命错误**: 日志显示 "❌ 致命CUDA错误...", 并立即退出。

现在您可以放心地运行您的训练任务了。如果再次遇到 OOM，系统将自动尝试恢复；如果遇到非法访问，它将立即停止并告诉您原因，而不是无限挂起。