I will perform a fast and safe parameter verification by bypassing the time-consuming data loading step.

**Plan:**

1. **Create a Single Model Checker (`tools/check_single_model_params.py`)**:

   * This helper script will verify one model at a time.

   * It will "mock" (disable) the data loading process to save time.

   * It will initialize the trainer, configure the model with `auto_tune=True` (to attempt reaching 10M parameters), and report the final parameter count.

2. **Create a Driver Script (`tools/check_params_fast_isolated.py`)**:

   * This script will list all available models.

   * It will run the single model checker for each model in a **separate process**.

   * This ensures both speed (no data loading) and safety (full memory cleanup between models).

3. **Execute and Report**:

   * Run the driver script.

   * Provide a concise report of which models can successfully be configured to \~10M parameters.

