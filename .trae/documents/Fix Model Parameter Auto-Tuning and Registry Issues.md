I will fix the model configuration issues to allow more models to approach the 10M parameter target, or explain why some cannot.

**Analysis of Failures:**

1. **Registry Errors (`branchencoder`,** **`convbnact`, etc.)**: These models are missing from the standard registry or have inconsistent naming between the registry and the `tools/training/model_loader.py` logic. I need to ensure they are correctly registered or mapped.
2. **Parameter Budget Failures (`UNO`,** **`NAFNet`,** **`Restormer`,** **`RCAN`** **- Too Large)**: These models default to very large configurations (e.g., high channel counts, many layers). The current `_auto_tune_model_params` logic likely fails to reduce them aggressively enough or doesn't know *which* hyperparameters to tune for these specific architectures.
3. **Parameter Budget Failures (`EDSR`,** **`RDN`,** **`PerceiverIO`** **- Too Small)**: These models default to small configurations. The auto-tuner needs to know how to *scale them up* (e.g., increasing `n_resblocks`, `n_feats`).
4. **Component "Models" (`Down`,** **`Up`,** **`SpectralConv2d`)**: These are building blocks, not full models. They should be excluded from the "runnable models" list.

**Plan:**

1. **Filter Invalid Models**:

   * Modify `tools/training/model_loader.py` or the `list_models()` function to exclude utility classes like `Down`, `Up`, `SpectralConv2d`, etc., which are not standalone models.

2. **Enhance Auto-Tuning Logic**:

   * Update the `_auto_tune_model_params` method in `tools/training/train_real_data_ar.py`.

   * **Add specific tuning rules** for the failing models:

     * **UNO**: Tune `hidden_channels` (or similar width param) and `n_layers`.

     * **NAFNet/Restormer**: Tune `width` (channels) and `enc_blk_nums`/`dec_blk_nums`.

     * **RCAN/RDN/EDSR**: Tune `n_resgroups`, `n_resblocks`, and `n_feats`.

     * **PerceiverIO**: Tune `num_latents` or `latent_dim`.

3. **Fix Registry Mappings**:

   * Check `models/__init__.py` and `tools/training/model_loader.py` to ensure `branchencoder`, `convbnact`, etc., are either correctly aliased or removed if they are just components.

4. **Verification**:

   * Re-run the **safe verification script** (`tools/check_params_fast_isolated.py`) for the fixed models to confirm they now pass.

**Implementation Details:**

* I will focus on fixing the **Auto-Tuning Logic** first, as this is the primary reason for "parameter mismatch" failures.

* I will add a dictionary of `tuning_strategies` to `_auto_tune_model_params` to handle different architectures intelligently.

