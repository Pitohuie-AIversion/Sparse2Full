I will update the batch training script `run_all_spatial_10m.py` to match the user's requirements and execute the experiment.

**Plan:**

1. **Update** **`tools/training/run_all_spatial_10m.py`**:

   * **Set Output Directory**: Modify the script to explicitly use `run_sw_4x` as the output directory, ensuring alignment with line 10 of the configuration file (`thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml`).

   * **Disable Smoke Test**: The current script has `"smoke_test": True`. I will set this to `False` to run the actual full experiment.

   * **Disable Strict Mode**: Set `"strict_mode": False` to allow models that don't perfectly hit 10M parameters to run (as per previous agreement).

   * **Update Model List**: Ensure the `MODEL_CATEGORIES` list is up-to-date and includes the models we want to test (filtering out the ones we know are broken components if necessary, though the script likely handles them).

   * **Correct Experiment Naming**: Ensure the experiment naming logic (`exp_prefix`) aligns with how `find_existing_run` looks for folders, to properly support resuming.

2. **Verify Configuration**:

   * Confirm that `thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml` indeed specifies `output_dir: run_sw_4x` at line 10. (Already verified via `Read` tool).

3. **Execute the Experiment**:

   * Run `python tools/training/run_all_spatial_10m.py`.

   * This will start the batch training process, iterating through models and saving results to `run_sw_4x/`.

4. **Monitor and Confirm**:

   * I will check the creation of the `run_sw_4x` directory and the subdirectories for the first few models to confirm the path is correct.

