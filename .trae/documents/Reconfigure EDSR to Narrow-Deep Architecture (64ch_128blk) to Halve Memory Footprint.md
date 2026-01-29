It's incredibly persistent. Even with `batch_size=1`, chunked checkpointing, and dual GPUs, it's still OOMing. This implies that the memory cost of a *single* forward/backward pass for one sample (128x128x128) is exceeding 44GB. This is theoretically possible if the intermediate activations are massive, but for EDSR (\~9.7M params), it's surprising.

Let's do a quick calculation:

* Input: 128x128

* Features: 128 channels

* Layers: 32 blocks \* 2 convs + extras ≈ 70 layers.

* Activation per layer: 128 \* 128 \* 128 \* 4 bytes (FP32) = 8 MB.

* Total Activations (without checkpointing): 70 \* 8 MB = 560 MB.

* This is tiny! Why is it taking 44GB?

**Wait, the input size might be wrong.** The log says: `Expected 3D ... but got input of size: [24, 1, 1, 128, 128]`. This looks like a 5D tensor, or a 4D tensor with weird dimensions.
Ah, `[24, 1, 1, 128, 128]`.
If the input is actually 128x128 spatial, but the channel dimension is somehow massive, or if the "128" is actually the *batch* size in some internal logic?
No, the log earlier showed `input_shape=(1, 1, 128, 128)` in the profiler.

**Hypothesis**: The "Chunk-based Checkpointing" implementation might be flawed. `torch.utils.checkpoint` requires the input to the checkpointed function to require gradients. If `x` (the input to the body) doesn't require gradients (e.g., first layer), checkpointing might be skipped or behave weirdly. But `x` comes from `self.head(x)`, so it should have grad.

**Another possibility**: The `checkpoint` function in PyTorch saves the *inputs* to the segment. If the segments are too small, we save too many inputs. If they are too large, the replay cost is high but memory should be low.
I used 4 chunks for 32 blocks. That's 8 blocks per chunk.
8 blocks \* 2 convs \* 8MB = 128 MB per chunk.
Re-running one chunk during backward takes \~128MB.
This shouldn't OOM.

**The "Expandable Segments" Warning**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The memory fragmentation might be the real killer here. We are allocating and freeing constantly.

**Plan to Fix (Last Resort before giving up on this architecture/size):**

1. **Enable** **`expandable_segments:True`**: Set this env var.
2. **Aggressive Garbage Collection**: Call `torch.cuda.empty_cache()` explicitly between epochs or even batches (slow, but might save us).
3. **CPU Offloading**: Move the optimizer state to CPU? No, simple DDP doesn't support that easily without ZeroRedundancyOptimizer.
4. **Reduce Model Size (Backtrack)**: If 10M params is truly impossible with this architecture on this hardware (which I doubt, but evidence suggests), we might need to reduce `n_feats` and increase `n_resblocks` to keep params same but reduce activation size?

   * Current: 128 feats, 32 blocks.

   * Alternative: 64 feats, 128 blocks?

   * Params: 32 \* (2 \* 128^2 \* 9) = 9.4M.

   * Params Alt: 128 \* (2 \* 64^2 \* 9) = 9.4M.

   * Activation per layer: 64*128*128\*4 = 4MB (Half).

   * Total Activations: 128 blocks \* 4MB = 512MB (Same total, but maybe better distribution?).

   * **Wait**, width (channels) is quadratic in params but linear in activation memory.

   * **Depth** is linear in params and linear in activation memory.

   * **Reducing width and increasing depth** is usually memory-friendlier for activations if we checkpoint!

   * Let's try **Narrower but Deeper**.

**Proposal**:
Change architecture to **Narrower (64 feats) but Deeper (128 blocks)**.
Params: `128 * 2 * 64 * 64 * 9 = 9,437,184`. (Same params!)
Memory: Feature map size drops from 128ch to 64ch (50% reduction per layer).
This should cut activation memory in half.

**Step-by-step Plan:**

1. Stop current training.
2. Modify `run_edsr_strict_10m.py`:

   * `n_feats: 64`

   * `n_resblocks: 128`

   * `batch_size: 2` (Try 2 again, or stick to 1). Let's stick to **1** to be safe first, then scale up.
3. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the script.
4. Relaunch.

