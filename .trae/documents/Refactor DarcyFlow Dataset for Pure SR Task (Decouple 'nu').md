# Plan to Refine DarcyFlow Dataset for SR Task

The user correctly identified that for the Sensor/SR simulation task, the physical parameter `nu` is irrelevant. The task is to reconstruct the High-Res `tensor` from a Low-Res observation. We need to decouple `nu` and strictly define the Input/Output pair as `LR(tensor)` -> `HR(tensor)`.

## 1. Modify `DarcyFlowDataset` Logic
The current implementation hardcodes `nu` as the input. I will refactor `datasets/darcy_flow_dataset.py` to:
- **Remove `nu` dependency**: Change default keys to `["tensor"]` only.
- **Update `__getitem__`**:
    - Load only `tensor` (HR Ground Truth).
    - Apply **Degradation Operator** (SR/Crop) to `tensor` to generate `lr_tensor`.
    - **Set `input_sequence` = `lr_tensor`**: This ensures the standard "input" field carries the Low-Res observation.
    - **Set `observed_lr_sequence` = `lr_tensor`**: For compatibility with the trainer's specific SR logic.
    - **Set `target_sequence` = `tensor`**: The HR ground truth.
- **Handle Dimensions**: Ensure `lr_tensor` shape matches model expectations (e.g., `[1, 1, 32, 32]` for SRx4).

## 2. Update Configuration
I will update `thesis_paper/configs/ar_paper_aligned_sr4_2D_darcy_flow.yaml` to:
- **Remove `nu` from keys**: Set `data.keys: ["tensor"]`.
- **Verify Channels**: Ensure `input_channels` and `target_channels` are set to 1 (correct).

## 3. Execution & Verification
- **Kill existing training**: The current process is training on `nu` -> `u`, which is wrong.
- **Restart Training**: Launch with the corrected dataset logic.
- **Verify**: Check logs to ensure data loading is successful and shapes are correct (Input: 32x32, Target: 128x128).
