I will resolve the CUDA Out-Of-Memory (OOM) error by reducing the batch size and enabling gradient accumulation to maintain training stability.

### Diagnosis
- **Error**: `CUDA out of memory` followed by `OOM Recovery failed`.
- **Cause**: The current batch size of **96** with full-resolution inputs ($128 \times 128$) for the EDSR model is exceeding the GPU memory limit (44GB). Unlike the previous SR task where inputs were small ($32 \times 32$), the "Crop & Pad" task uses full-size sparse inputs, significantly increasing memory usage per sample.
- **Secondary Issue**: The automatic OOM recovery mechanism failed because it didn't handle the 5D input tensor `[B, T, C, H, W]` correctly, but preventing the OOM is the root fix.

### Fix Plan
1.  **Reduce Batch Size**: Lower `batch_size` from **96** to **32** in `thesis_paper/configs/ar_paper_aligned_crop4_shallow_water.yaml`.
2.  **Compensate with Accumulation**: Increase `gradient_accumulation_steps` from **1** to **3**.
    *   Effective Batch Size: $32 \times 3 = 96$ (Unchanged).
    *   Memory Footprint: Reduced by ~3x.
3.  **Restart Training**: Launch the training again with the optimized configuration.

I will apply these changes immediately.
