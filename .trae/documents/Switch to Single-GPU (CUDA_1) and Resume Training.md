I will perform the following steps to switch the training to single-GPU (CUDA:1) and resume from the breakpoint:

1. **Terminate Current Training**:

   * Find and kill the `run_all_spatial_10m.py` process.

   * Find and kill the `train_real_data_ar.py` processes (dual-card workers).
2. **Modify** **`tools/training/run_all_spatial_10m.py`**:

   * **Environment**: Set `CUDA_VISIBLE_DEVICES=1` to strictly use the second GPU.

   * **Execution Command**: Switch from `torchrun` (distributed) to `python` (single process) to avoid DDP overhead.

   * **Config Overrides**: Change `"devices": 2` to `"devices": 1`.
3. **Resume Training**:

   * Relaunch the script using `nohup` in the background.

   * The script's existing logic will automatically detect the interrupted `ConvUNetLite` experiment and resume from the latest checkpoint (`last.ckpt`).
4. **Monitor**:

   * Restart the log monitoring (`tail -f`) to ensure the resume is successful and training continues on the correct device.

