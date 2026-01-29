I will proceed with a deep scan focusing on extremely small observation windows, excluding the baseline 64.

### 1. Experiment Design (Deep Sparsity Scan)
*   **Objective**: Find the absolute minimal information required for reconstruction, pushing the limit down to a single pixel.
*   **Scan Levels (Descending Size)**:
    *   **48x48** (Medium-Hard)
    *   **32x32** (Hard)
    *   **16x16** (Very Hard)
    *   **8x8** (Ultra Hard)
    *   **4x4** (Extreme)
    *   **1x1** (Single Pixel Limit)
*   **Fixed Parameters**: Input size 128x128, EDSR model.

### 2. Implementation Plan
I will create `tools/scan_crop_capability.py` to:
1.  **Iterate** through sizes: `[48, 32, 16, 8, 4, 1]`.
2.  **Train** models for each size using the consistent configuration.
3.  **Collect** metrics to plot the "Performance Collapse Curve" (Accuracy vs. Size).

### 3. Execution
I will write the script and start the scan immediately upon approval. The runs will be executed sequentially to ensure stability.