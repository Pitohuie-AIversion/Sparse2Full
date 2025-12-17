## Scope
- Investigate temporal model and training pipeline sources of persistently high validation loss, focusing on temporal context, scheduled sampling, input shaping, and validation composition.

## What I Will Inspect (Read-Only)
- Temporal pipeline structure and inputs:
  - `models/temporal/components/sequential_spatiotemporal.py`
    - Spatial→Temporal interface and padding logic: 444–463, 458–461
    - TemporalFeatureExtractor behavior and dynamic projection: 267–301
    - Teacher Forcing mixing and decay policy: 603–626, 639–646
    - Rollout inference path for stepwise prediction: 655–694
  - Input flattening and expected input_dim calculation: 498–516
- Validation loop composition and metrics aggregation:
  - `tools/training/train_real_data_ar.py`
    - Validation cropping to `current_T_out`: 4011–4039
    - Validation loss composition and fallback to `rel_l2+mae`: 4134–4166, 4170–4241
    - Per-epoch logging and summaries: 5395–5400
- Loss function implementation and weights consumption:
  - `ops/losses.py`
    - `compute_ar_total_loss` inputs, weight extraction, reconstruction and optional spectral/DC components: 1000–1147, 1149–1173
- Spatial backbone behavior:
  - `models/spatial/fno2d.py`: PE-free FNO and 2D conv combining; coordinate injection via `fc0` and FFT path: 175–193, 220–236

## Likely Issues to Validate
- **Temporal context limitation**: Initial `T_in=1` (thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml:97) constrains sequence modeling; even with `T_in=3`, confirm how many frames reach transformer per batch.
- **Duplicate last-frame padding**: The temporal input is padded by repeating the last observed frame (sequential_spatiotemporal.py:458–461), feeding static sequences to the encoder.
- **Teacher Forcing distribution gap**: Early epochs mix GT heavily (603–626) while validation uses only model outputs, lowering train loss without val change.
- **Input flattening collapsing spatial structure**: Temporal input dimension is `(spatial_feature_dim + out_channels) * H * W` (498–516), then reduced via linear projection to `temporal_dim`, losing spatial locality.
- **Validation loss composition mismatch**: When reconstruction loss is missing, validation uses `rel_l2 + mae` (4134–4166); ensure training uses the same composition for comparison.
- **Missing temporal positional encoding**: Transformer without PE (267–301) underfits sequential dynamics.

## Proposed Fixes (to implement after inspection)
1. **Remove duplicate padding**: Do not repeat last frame; instead feed true `[B, T_in, ...]`, and generate `T_out` via rollout with scheduled sampling.
2. **Switch training to rollout**: Use `rollout_inference` during training with scheduled sampling (655–694) to align train/val distributions.
3. **Enable temporal positional encoding**: Inject sinusoidal PE at `[B, T, D]` before transformer; gate via config `temporal.use_positional_encoding: true`.
4. **Reduce temporal input spatial size before flattening**: Use `temporal.reduce_spatial: 'avg_pool'` with `reduce_size: [16,16]` to preserve structure and reduce dimensional collapse.
5. **Align validation composition**: Ensure `compute_ar_total_loss` is used uniformly; log “last-step” Rel-L2 to measure rollout error growth.
6. **Tune teacher forcing**: Lower early TF (cap `teacher_prob ≤ 0.3`) and decay faster (e.g., `teacher_forcing_decay: 0.8`).

## Deliverables
- A concise diagnosis report citing exact code locations and how they affect validation loss.
- A patch plan enumerating config/code changes (PE, rollout training, padding removal, reduce_spatial).
- A verification plan: per-epoch `Val/RelL2`, `Val/MAE`, last-step Rel-L2, with `metrics.jsonl` and `results.md` summarizing trends.

## Next Step
- I will perform the read-only inspection above, confirm each suspected issue with code references and current config values, then proceed to implement the fixes once you confirm. 