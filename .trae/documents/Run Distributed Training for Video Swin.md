I will launch the Video Swin Transformer training experiment using `torchrun` on 2 GPUs.

**Plan:**

1. **Configure Environment**:

   * Set `CUDA_VISIBLE_DEVICES=0,1` (assuming 2 GPUs are available).

   * Set `nproc_per_node=2` for `torchrun`.
2. **Update Config**:

   * Modify `thesis_paper/configs/temporal/ar_training_config_video_swin.yaml` to enable distributed training (`distributed.enabled: true`, `distributed.backend: nccl`).

   * Set `device.devices: 2`.
3. **Execute Command**:

   * Run the training script via `torchrun`.
4. **Monitor**:

   * Check if the process starts correctly and utilizes both GPUs.

**Config Changes Required:**
I need to quickly update the YAML to support DDP (Distributed Data Parallel) before launching.

**Command:**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/training/train_real_data_ar.py --config thesis_paper/configs/temporal/ar_training_config_video_swin.yaml
```

