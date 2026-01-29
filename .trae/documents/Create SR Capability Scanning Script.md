I will create a new scanning script `tools/scan_sr_capability.py` to evaluate the model's Super-Resolution performance across a range of input resolutions.

**Implementation Plan:**

1.  **Create Script**: `tools/scan_sr_capability.py`
    *   **Base**: Adapted from `tools/scan_crop_capability.py`.
    *   **Scan Range**: Input sizes `[32, 16, 8, 4, 2, 1]` (corresponding to downsampling scales x4, x8, x16, x32, x64, x128).
    *   **Configuration**:
        *   Base Config: `thesis_paper/configs/ar_paper_crop_edsr_spatial_only_refined.yaml`.
        *   **Dynamic Overrides**:
            *   Set task mode to `SR`.
            *   Calculate `scale = 128 // input_size`.
            *   Update `data.observation.scale` and `training.degradation.scale` to `scale`.
            *   Update `model.upscale` to `scale` (required for EDSR).
            *   Update experiment name to `AR-DR2D-SR-Scan-Input{size}`.
2.  **Execution Logic**:
    *   The script will sequentially run training jobs for each scale using `torch.distributed.run`.
    *   It will parse `test_results.json` to collect metrics (Rel-L2, PSNR, SSIM).
    *   Finally, it will generate a Markdown summary table of the performance curve.

This script will allow you to obtain the "limit is 1" scanning curve for the DRD EDSR SR task, exactly parallel to the Crop experiment.