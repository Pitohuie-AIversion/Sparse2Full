Okay, I will refine the plan to explicitly include "3loss\_ablation" in the output directory structure, ensuring the results are clearly organized for your thesis.

### Revised Plan: Batch Loss Ablation Script with Organized Output

1. **Create Script**: `experiment_scripts/batch_run_loss_ablation.sh`.
2. **Key Features**:

   * **Base Configuration**: Uses your spatial-only config (`thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml`).

   * **Output Organization**:

     * Sets a common `logging.experiment_name` prefix: `3loss_ablation`.

     * The full experiment name will be dynamic: `3loss_ablation/A0_RecOnly`, `3loss_ablation/A2_RecSpec`, `3loss_ablation/A3_Full`.

     * This ensures all logs and checkpoints are saved under a dedicated `runs_3loss_ablation/` folder (depending on how your training script handles paths, usually it uses `experiment_name`).

   * **Experiments**:

     1. **A0 (Rec Only)**: `loss.spectral.weight=0.0` `loss.degradation_consistency.weight=0.0`
     2. **A2 (Rec + Spec)**: `loss.spectral.weight=0.5` `loss.degradation_consistency.weight=0.0`
     3. **A3 (Full)**: `loss.spectral.weight=0.5` `loss.degradation_consistency.weight=1.0` (Your "Ours" setting).

   * **Execution**: Sequential execution to prevent GPU OOM (Out Of Memory).

### Script Structure

The script will iterate through these configurations and run `torchrun`. I'll add comments to make it easy for you to add more ablation studies later (e.g., A1: Rec + DC).
