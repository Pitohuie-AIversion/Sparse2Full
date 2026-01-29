I will restart the dual-GPU experiment by updating the batch runner script to dynamically scan all spatial models and use the specified output directory.

**Plan:**

1.  **Update `tools/training/run_all_spatial_10m.py`**:
    *   **Dynamic Model Scanning**: Replace the hardcoded `MODEL_CATEGORIES` list with a dynamic call to `tools.training.model_loader.list_models()`. This ensures all available spatial models in `models/spatial` are tested as requested.
    *   **Verify Output Directory**: Confirm the script reads the output directory from the config (`run_sw_4x`), ensuring alignment with `thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml` line 10.
    *   **Confirm Batch Size**: Retain the config's `batch_size: 96`. On a dual-GPU setup, this typically results in 96 samples per GPU (total 192) or 48 per GPU depending on the dataloader implementation. Given the goal of 95% VRAM utilization for small (10M) models, this is a reasonable starting point.

2.  **Execute the Experiment**:
    *   Run the updated `tools/training/run_all_spatial_10m.py`.
    *   This will launch `torchrun` with `--nproc_per_node=2` for each model, ensuring dual-GPU utilization.

**Note on Batch Size**: The configuration sets `batch_size: 96`. For 10M parameter models on modern GPUs (like A100/A6000), this should be efficient without OOM. If OOM occurs, the script's `oom_recovery` is enabled to handle it.