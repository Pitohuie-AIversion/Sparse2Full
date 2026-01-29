I will verify if all spatial models can successfully reach the target parameter size of 10M.

**Plan:**

1. **Execute Model Smoke Test**:

   * Run the `tools/training/train_real_data_ar.py` script with the `--smoke-all` flag.

   * Set the target parameter size to 10M using `--target-params-m 10.0`.

   * Use a large tolerance of 2.0M (`--tolerance-m 2.0`) to allow for some variation, as achieving exactly 10M can be difficult for some architectures.

   * Override config settings to speed up the process (batch size 1, sample limit 5, no validation/logging).

   **Command:**

   ```bash
   python tools/training/train_real_data_ar.py \
       --config thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml \
       --smoke-all \
       --target-params-m 10.0 \
       --tolerance-m 2.0 \
       training.epochs=1 \
       training.batch_size=1 \
       data.dataloader.batch_size=1 \
       data.sample_limit=5 \
       training.validation.enabled=False \
       logging.log_model=False
   ```

2. **Analyze Results**:

   * The script will output a JSON file `runs/smoke_all_results.json` containing the parameter counts for each model.

   * I will read this file and summarize which models successfully configured to \~10M parameters and which failed or deviated significantly.

3. **Report to User**:

   * List the models that successfully achieved \~10M parameters.

   * Identify any models that failed to initialize or could not be tuned to the target size.

   * Provide the actual parameter counts for verification.

