I have successfully updated the configuration and training script to implement the **Random Crop & Pad (Sparse Observation)** task as requested.

### Changes Implemented:

1.  **Configuration (`ar_paper_aligned_crop4_shallow_water.yaml`)**:
    *   **Observation Mode**: Set to `Crop` with `boundary: zero`.
    *   **Model**: Configured `EDSR` with `upscale: 1` (since input is a full-size sparse image, not low-res).
    *   **Crop Size**: Set `crop_size: [128, 128]` in observation (output size) and `training.crop.size: 64` (window size).

2.  **Training Script (`train_real_data_ar.py`)**:
    *   **Renamed & Updated Logic**: Replaced `_apply_crop_training` with `_apply_random_masking`.
    *   **Sparse Input Generation**: Now generates **Full-Size Sparse Inputs** ($128 \times 128$ image with only the crop visible, rest zero) instead of small patches.
    *   **Mask Support**: Returns the generated masks.
    *   **DC Loss Consistency**: Updated `train_epoch` to use the exact random mask applied to the input for calculating the Data Consistency (DC) loss on the prediction (`Mask * Pred` vs `Mask * GT`), ensuring correct physics constraints.

3.  **Verification**:
    *   Performed a debug run (`debug_masking_test`) which confirmed:
        *   Input shape is `[B, 1, 1, 128, 128]` (Sparse Full Size).
        *   Target shape is `[B, 1, 1, 128, 128]` (Dense Full Size).
        *   Training loop runs correctly with valid loss convergence.

### Current Status:
*   The **formal training** command has been launched in the background (`experiment_name: AR-DR2D-EDSR-CropMask-10M-300ep`).
*   The system is now training an EDSR model to reconstruct full dense fields from sparse random crops (Inpainting task).
