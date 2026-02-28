# Add Training Curves to Chapter 4 Spec

## Why
Chapter 4 currently presents experimental results primarily through tables (e.g., Table 4-2, 4-4, 4-9). While these tables quantify the final performance, they lack the temporal dimension of the training process. Specifically, the claims about "Sequential Training Stability" (Section 4.3.2) and "Physical Loss Convergence" (Section 4.4.1) are best supported by visual curves showing metric evolution over epochs. Adding these curves will strengthen the evidence for the proposed methods' robustness and efficiency.

## What Changes
- **Add Python Script**: Create `tools/plot_training_curves.py` to extract metrics (Loss, Rel-L2, PSNR) from `training_history.json` files in `runs/` and generate publication-quality plots.
- **Generate Figures**:
    - **Figure 4-4 (Convergence Comparison)**: Comparative convergence curves of EDSR vs. UNet vs. FNO on SWE dataset (validating Table 4-2).
    - **Figure 4-5 (Sequential Evolution)**: Dual-axis plot showing Rel-L2 and fRMSE-High evolution during Stage 2 -> Stage 3 transition (validating Section 4.3.2).
    - **Figure 4-6 (Ablation Dynamics)**: Validation Loss/Metric curves for "MSE Only" vs. "Full Loss" (validating Section 4.4.1).
- **Update Manuscript**: Insert these figures into `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md` with appropriate academic captions.

## Impact
- **Affected Specs**: None.
- **Affected Code**: `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md` (content addition).
- **New Artifacts**: `tools/plot_training_curves.py`, `thesis_paper/manuscript_5_chapter/images/fig4-*.png`.

## ADDED Requirements
### Requirement: Training Convergence Plot
The system SHALL generate a plot comparing the validation Rel-L2 convergence of EDSR, UNet, and FNO over epochs.
#### Scenario: Visualization
- **WHEN** the plotting script is run
- **THEN** it generates `fig4-4_training_convergence.png` showing EDSR converging faster and to a lower error.

### Requirement: Sequential Training Evolution Plot
The system SHALL generate a plot showing the transition from Stage 2 (Spatial Pre-training) to Stage 3 (End-to-End Fine-tuning).
#### Scenario: Visualization
- **WHEN** the plotting script is run
- **THEN** it generates `fig4-5_sequential_evolution.png` clearly marking the "Unfreeze" point and the subsequent metric improvement (or trade-off).

### Requirement: Ablation Study Plot
The system SHALL generate a plot comparing the training dynamics of the model with and without physical consistency losses.
#### Scenario: Visualization
- **WHEN** the plotting script is run
- **THEN** it generates `fig4-6_ablation_curves.png` showing the "Full Loss" variant achieving better validation performance or stability.
