I will verify the 10M parameter configuration for all models using a memory-safe approach. The previous failure was caused by running all models sequentially within a single process, leading to memory fragmentation and accumulation.

**Plan:**

1. **Create a Safe Verification Script (`tools/check_10m_params_safe.py`)**:

   * This script will identify all available models.

   * It will iterate through each model and verify it in a **separate isolated process** (using `subprocess`). This ensures full GPU memory release after each model check, preventing Out-Of-Memory (OOM) errors.

   * For each model, it will run the training script in "dry run" mode with:

     * Target parameters: 10M (`--target-params-m 10.0`)

     * Tolerance: ±2M (`--tolerance-m 2.0`)

     * Minimal batch size and samples to ensure speed.

   * It will collect results (Success/Failure, Actual Parameter Count) into a summary file.

2. **Execute the Verification**:

   * Run `python tools/check_10m_params_safe.py`.

   * This will take longer than the previous method but is safe for the server.

3. **Report Results**:

   * Present a summary table of which models successfully scaled to \~10M parameters.

   * Highlight any models that failed to configure or run.

