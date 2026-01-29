I will finalize the configuration for the full training run and then execute it.

### 1. Final Configuration Check
- I have reviewed `thesis_paper/configs/ar_paper_aligned_crop4_shallow_water.yaml`.
- The `description` still says "快速Debug配置 - EDSR Crop训练". I will update it to reflect the full training run.
- `data.sample_limit` is `null` (correct for full training).
- `training.epochs` is 300 (correct).
- `training.crop` is enabled with `size: 64` and `patches_per_image: 4`.
- `model` is `EDSR` with `upscale: 4`.
- `logging.experiment_name` is "AR-DR2D-10M-300ep". I will update this to "AR-DR2D-EDSR-SRx4-10M-300ep" to match the experiment name.

### 2. Execution
- I will run the training command using the verified configuration file.
- I will monitor the initial output to ensure it starts correctly and is not in "Debug" mode (e.g., correct sample count).

### Plan Steps:
1.  **Update Config**: Edit the yaml file to set the description and experiment name for the production run.
2.  **Run Training**: Execute `python tools/training/train_real_data_ar.py --config thesis_paper/configs/ar_paper_aligned_crop4_shallow_water.yaml`.
