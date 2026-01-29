The 7.6M model (96 feats / 48 blocks) is **still OOMing** on dual GPUs. This confirms that even intermediate widths are too heavy for this specific 128x128x128 (assuming 3D-like behavior or massive 2D) activation footprint.

I will now execute the **Drastic Reduction** plan to a configuration that is guaranteed to run, and then we can slowly creep up if needed (or just accept the smaller model as the "EDSR baseline" given hardware constraints).

**New Configuration (Conservative Baseline):**

* `n_feats: 64` (Minimum reasonable width)

* `n_resblocks: 32` (Standard depth)

* **Params**: \~2.4M

* **Batch Size**: 2 (Per GPU) -> Global 4

* **Gradient Accumulation**: 24 -> Effective 96

* **Checkpointing**: Enabled (Chunked)

This is a very small model compared to the others, but it's the only way to get a successful run right now. Once it starts running stably, we can confirm the pipeline is healthy.

**Plan:**

1. Stop current run.
2. Modify `run_edsr_strict_10m.py`:

   * `n_feats: 64`

   * `n_resblocks: 32`
3. Relaunch.

