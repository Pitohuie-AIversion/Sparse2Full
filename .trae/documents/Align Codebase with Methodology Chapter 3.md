I have identified the causes for the parameter count misalignment. It is a combination of configuration settings and code logic limitations.

### **Root Causes**

1. **Configuration (`auto_tune: false`)**:
   The primary reason is that `auto_tune` is set to `false` in your run configurations (as seen in `runs_drd/.../config_merged.yaml`). This means the training script ignores the `target_params_m: 10.0` setting and uses the default model parameters, which vary widely (e.g., \~50K for Lite models, \~56M for SwinUNet).

2. **Code Logic Limitations in** **`_auto_tune_model_params`**:
   Even if `auto_tune` were enabled, the current logic in `tools/training/train_real_data_ar.py` would fail to align several models to 10M:

   * **MLP**: The tuner adjusts `embed_dim`, but the `MLPModel` expects `hidden_dims`. The tuner also has a hard width cap (1024) that prevents it from reaching 10M.

   * **LIIF**: The tuner doesn't handle `LIIF` specifically, and the fallback logic fails because LIIF uses `hidden_list`, not `embed_dim` or `width`.

   * **SwinUNet**: The tuning range is restricted to `[base_dim, base_dim*2]`. Since the default `base_dim` (96) yields \~56M params, the tuner never explores smaller values to reach 10M.

   * **Lite Models (Restormer/NAFNet)**: These fall into a generic fallback block that may not scale them aggressively enough or correctly if their specific parameter names differ.

### **Proposed Solution**

I will fix the code logic in `tools/training/train_real_data_ar.py` to properly support tuning for these models.

**Implementation Steps:**

1. **Improve** **`_auto_tune_model_params`**:

   * **Swin/Transformer**: Expand the search range to allow scaling *down* (e.g., `0.5 * base` to `2.0 * base`).

   * **MLP**: Fix parameter mapping (update `hidden_dims` based on tuned width) and remove the restrictive 1024 width cap.

   * **LIIF**: Add a specific handler to tune `hidden_list`.

   * **Lite Models**: Add specific handlers or improve the generic fallback to ensure `embed_dim` is correctly scaled for `RestormerLite`, `NAFNetLite`, etc.
2. **Enable Auto-Tuning**:

   * I will update the default configuration or the specific YAML you are using to set `auto_tune: true` so the fixes take effect. (Or I can modify the code to default `auto_tune` to True if `model_budget` is present).

### **Verification**

I will run the reproduction script again after the fixes to verify that:

* `MLP` scales to \~10M.

* `LIIF` scales to \~10M.

* `SwinUNet` scales down to \~10M.

* `RestormerLite` scales up to \~10M.

