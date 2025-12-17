## Key Findings
- T_in=1 limits temporal context (thesis_paper/configs/temporal/ar_training_config_debug_temporal_cpu copy.yaml:97); transformer sees only a single frame.
- Validation crops to current T_out (tools/training/train_real_data_ar.py:4011–4039), but training may be more optimized for earlier stages, leading to mismatch sensitivity.
- Teacher Forcing uses `teacher_prob = decay^epoch` (models/temporal/components/sequential_spatiotemporal.py:639–646) and mixes GT into spatial outputs (603–626). High early TF reduces train loss but not validation.
- Validation `val_loss` aggregates reconstruction loss or fallback `rel_l2+mae` (tools/training/train_real_data_ar.py:4134–4166, 4170–4241); with `loss.mae_weight=0.0`, training mainly optimizes `rel_l2`, but validation summary may still behave differently.
- TemporalFeatureExtractor lacks positional encoding (models/temporal/components/sequential_spatiotemporal.py:246–301); transformer without PE often underfits sequential dynamics.

## Plan (Config + Code)
### 1) Align Train/Val Temporal Settings
- Increase temporal context: `data.T_in: 3` (or 5) to provide history.
- Fix curriculum for early convergence: stage `[T_out:1, epochs:40] → [T_out:3, 30] → [T_out:5, 30]` so train/val are strictly aligned during each stage.
- Reduce teacher forcing: set `teacher_forcing_decay` from 0.95 to ~0.8 so TF decays faster; optionally cap `teacher_prob` at ≤0.3 in early epochs.

### 2) Improve Validation Sensitivity
- Keep validation loss strictly in the same composition as training: ensure `compute_ar_total_loss` is used and weights are identical (already true when `training.loss_weights` are provided).
- Log an explicit “last-step” metric (RelL2/MAE) during validation (tools/training/train_real_data_ar.py:4170–4241) to detect rollout error growth.

### 3) Add Temporal Positional Encoding
- Implement sinusoidal temporal PE in `TemporalFeatureExtractor` (models/temporal/components/sequential_spatiotemporal.py:246–301):
  - Create PE `[T, d_model]`; add to projected inputs before transformer.
  - Gate via config flag: `temporal.use_positional_encoding: true`.

### 4) CPU Utilization Tuning (Non-blocking)
- Keep `num_workers` near CPU logical cores (192) and adjust `prefetch_factor` to 12–16.
- Monitor `iowait`; if high, reduce workers to ~128 and increase `prefetch_factor` to 16 to balance IO vs scheduling.

### 5) Outputs & Verification
- Export per-epoch `Val/RelL2`, `Val/MAE`, and “last-step” RelL2 to `metrics.jsonl` under `runs/<exp>/`.
- Provide a short `results.md` summarizing convergence trends across stages.

### 6) Execution Steps
- Update YAML: `T_in`, curriculum stages, `teacher_forcing_decay`, ensure `training.loss_weights` includes `derivative_consistency` & `energy_consistency`.
- Add PE in transformer and config flag; rebuild and resume training (CPU).
- Monitor CPU and adjust workers/prefetch/batch to reach ≥90% utilization, resuming from checkpoints automatically.

If approved, I will implement the YAML changes, add temporal positional encoding, restart CPU training, and deliver metrics/summary demonstrating validation convergence improvements.