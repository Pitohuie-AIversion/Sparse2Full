I will ensure all valid spatial models pass a comprehensive functional test (Smoke Test) by filtering out invalid components and fixing any runtime issues.

**Plan:**

1. **Filter Invalid Components**:

   * Update `tools/training/model_loader.py` to exclude `branchencoder`, `overlappatchembed`, and any other non-model components identified in previous checks. These are internal building blocks, not standalone models.

2. **Execute Comprehensive Smoke Test**:

   * Create and run a new verification script `tools/verify_all_models_runtime.py`.

   * This script will run a **real training step** (not just parameter check) for every valid model using the `ar_paper_aligned_sr4_shallow_water.yaml` config.

   * It will use minimal settings (1 epoch, 2 batch size, 4 samples) to be fast but thorough.

   * This ensures data loading, forward pass, loss calculation, and backward pass all work.

3. **Fix Runtime Errors**:

   * Analyze the smoke test results.

   * If any model fails (e.g., dimension mismatch, missing arguments), I will modify the model code or the adapter logic in `train_real_data_ar.py` to fix it.

4. **Final Report**:

   * Provide a summary showing 100% pass rate for all valid spatial models.

