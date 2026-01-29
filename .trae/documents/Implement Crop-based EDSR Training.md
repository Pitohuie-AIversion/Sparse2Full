I will implement crop-based EDSR training by modifying the configuration and training script.

### 1. Configuration Changes (`thesis_paper/configs/ar_paper_aligned_crop4_shallow_water.yaml`)
- **Model**: Change `model.name` to `EDSR` and update model-specific parameters (e.g., `n_resblocks`, `n_feats`).
- **Crop Training**: Add a new `training.crop` section to define:
  - `enabled`: `true`
  - `size`: `64` (Target HR crop size)
  - `stride`: `null` (Random cropping for training)
  - `patches_per_image`: `4` (Number of crops to extract per source image)
- **Data Loading**: Adjust `batch_size` to accommodate the increased number of samples due to cropping (effective batch size = `batch_size` * `patches_per_image`).
- **Observation**: Ensure `observation.mode` is set to `SR` with `scale_factor: 4` to define the High-Res to Low-Res degradation.

### 2. Training Script Changes (`tools/training/train_real_data_ar.py`)
- **Preprocessing Helper**: Add a `_random_crop(tensor, size)` method to the `RealDataARTrainer` class. This method will extract random patches from the High-Resolution (Ground Truth) images.
- **Training Loop (`train_epoch`)**:
  - Insert the cropping logic **before** the observation operator application.
  - **Flow**:
    1. **Load Batch**: Get full-resolution Ground Truth images $X_{GT}$ (128x128).
    2. **Crop**: Extract random HR patches $P_{GT}$ (e.g., 64x64).
    3. **Degrade**: Apply the SR observation operator to $P_{GT}$ to generate Low-Resolution inputs $P_{LR}$ (e.g., 16x16 for x4 scale).
    4. **Forward**: Pass $P_{LR}$ to EDSR to predict $\hat{P}_{HR}$.
    5. **Loss**: Compute loss between $\hat{P}_{HR}$ and $P_{GT}$.
- **Validation**:
  - Keep validation on **full images** (128x128) to ensure global consistency and check for boundary artifacts as requested. This validates that the patch-trained model generalizes to full fields.
- **Memory Optimization**: Ensure unused tensors are cleared and operations are efficient.

### 3. Verification
- **Reproducibility**: Ensure the random cropping uses the global random state (seeded).
- **Testing**: I will run a short training cycle to verify the pipeline runs without errors and the loss decreases.
