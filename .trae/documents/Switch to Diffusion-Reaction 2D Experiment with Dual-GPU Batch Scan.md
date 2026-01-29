I will perform the following steps to switch the experiment to the Diffusion-Reaction 2D dataset with the specified 8 models and dual-GPU configuration:

1. **Terminate Previous Run**: Locate and stop the background process running the Shallow Water experiments.
2. **Update Batch Script (`tools/training/run_all_spatial_10m.py`)**:

   * **Config Path**: Update to `thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml`.

   * **Model List**: Restrict `MODEL_CATEGORIES` to the 8 specific models: `edsr`, `nafnet`, `UformerLite`, `uno`, `ConvUNetLite`, `UNet`, `stablefno2d`, `RestormerLite`.

   * **Experiment Naming**: Update the experiment naming convention to `AR-DR2D-{model_name}-SRx4-10M-300ep`.

   * **Output Directory**: Explicitly set the base output directory to `runs_drd_paper` and ensure each model writes to `runs_drd_paper/AR-DR2D-{model_name}-SRx4-10M-300ep`, strictly following the YAML structure.
3. **Launch Background Training**: Execute the updated script using `nohup` to ensure it continues running after SSH disconnects, utilizing 2 GPUs via `torchrun`.
4. **Start Monitoring**: Open a new terminal to monitor the output log (`nohup_drd_10m.out`) in real-time.

