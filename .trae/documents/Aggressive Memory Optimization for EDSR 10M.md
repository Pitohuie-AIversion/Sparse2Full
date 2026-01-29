The OOM error persists because even with a smaller batch size (8), the OOM recovery mechanism in `RealDataARTrainer` is incorrectly trying to recover by splitting batches, but failing on input dimension validation (`Expected 3D or 4D...`). This suggests the input tensor shape is somehow unexpected during the recovery attempt, or the recovery logic itself is flawed for this model.

However, the primary goal is to **prevent OOM** in the first place, not just recover from it. The EDSR model with 10M parameters (128 feats, 32 blocks) is very deep and memory-intensive due to large feature maps being maintained throughout the network (no downsampling in the body).

I will implement a more aggressive memory optimization strategy:

1. **Disable OOM Recovery**: Since it's failing and causing loops, we should disable it to fail fast or rely on standard gradient checkpointing.
2. **Drastically Reduce Batch Size**: Drop `batch_size` to **4** or **2**.
3. **Increase Gradient Accumulation**: Compensate to maintain effective batch size \~96 (e.g., BS=4 -> Acc=24).
4. **Strictly Enable Gradient Checkpointing**: Ensure `model.set_gradient_checkpointing(True)` is actually called if the model supports it. (EDSR might not implement this method natively, which is a key issue).
5. **Verify Model Implementation**: Check `models/spatial/edsr.py` (or similar) to see if it supports gradient checkpointing. If not, I will monkey-patch it or wrap the forward pass.

**Plan:**

1. **Check EDSR Implementation**: Verify if `gradient_checkpointing` is supported.
2. **Modify** **`run_edsr_strict_10m.py`**:

   * Set `batch_size = 4`.

   * Set `gradient_accumulation_steps = 24`.

   * Disable `oom_recovery`.
3. **Relaunch**: Kill previous processes and start again.

