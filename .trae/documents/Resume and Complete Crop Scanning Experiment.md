I have analyzed the status of your crop scanning experiment.

**Current Status:**

* **Size 48, 32, 16**: ✅ Completed (found `test_results.json`).

* **Size 8**: ⚠️ **Interrupted** at Epoch 43/100 (found `last.ckpt` in `runs_drd_paper/AR-DR2D-Crop-Scan-Size8-model_EDSR-s2025-20260119`).

* **Size 4, 1**: ⏳ Pending (not started).

**Plan:**

1. **Create a Resume Script**: I will create a new script `tools/scan_crop_capability_resume.py` that improves upon the original scanner.

   * **Smart Detection**: It will identify existing timestamped directories (e.g., `...-s2025-20260119`).

   * **Resume Logic**: For Size 8, it will automatically detect the `last.ckpt` and resume training from where it left off (Epoch 43) using `--resume`.

   * **Continue Sequence**: After Size 8 finishes, it will automatically proceed to train Size 4 and Size 1.

   * **Configuration**: It will maintain the same DDP (2 GPUs) and parameters as your previous runs.

2. **Execute**: Run this script to complete the experiment series without repeating work.

