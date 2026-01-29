I will create and launch a dedicated training task for the **EDSR** model, manually configured to reach approximately 10M parameters, as the automatic tuning previously resulted in an under-sized model (1.2M).

1. **Create Script**: `tools/training/run_edsr_strict_10m.py`

   * **Configuration**: Load `thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml`.

   * **Manual Overrides**: Explicitly set `n_feats=128` and `n_resblocks=32` to target \~9.4M parameters (bypassing auto-tune).

   * **Output Directory**: `runs_drd_paper/AR-DR2D-EDSR-SRx4-10M-Strict`.

   * **Device**: Force execution on **CUDA:1**

2. **Execute**:

   * Run the script using `nohup` in the background.

3. **Monitor**:

   * Check the logs to confirm the parameter count is close to 10M and training has started successfully.

