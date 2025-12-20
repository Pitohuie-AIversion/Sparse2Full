I will perform the following steps to fulfill your request:

1.  **Create a Batch Training Script**: I will create a Python script `tools/training/run_all_spatial_10m.py` that:
    *   Identifies all spatial models located in `models/spatial` using the project's `ModelLoader`.
    *   Iterates through each model.
    *   Constructs a `torchrun` command to execute `tools/training/train_real_data_ar.py` on 2 GPUs (`--nproc_per_node=2`).
    *   Passes the specific configuration arguments:
        *   Config file: `thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml`
        *   Model name: `model.name={model_name}`
        *   Target parameters: `model_budget.target_params_m=10.0`
        *   Auto-tuning: `model_budget.auto_tune=True` (This triggers the built-in parameter adjustment logic in the training script to reach ~10M parameters).
        *   Experiment name: `experiment.name=AR-ShallowWater-10M-{model_name}`

2.  **Execute the Script**: I will run the newly created script. This will start the training process for each model sequentially, fully utilizing both GPUs for each run.

This approach ensures all spatial models are benchmarked with consistent parameter counts (~10M) on the Shallow Water dataset, utilizing your hardware resources efficiently.