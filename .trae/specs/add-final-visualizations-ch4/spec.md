# Add Final Visualizations to Chapter 4 Spec

## Why
While Chapter 4 contains essential result metrics and training curves, it currently lacks two critical types of visualizations that are standard in high-quality SciML theses:
1.  **Temporal Error Accumulation (Rollout Error)**: For spatiotemporal tasks (Section 4.2.2), it is crucial to show how errors accumulate over time steps. This demonstrates the model's long-term stability versus autoregressive drift.
2.  **Failure Case Analysis**: To be academically rigorous, the thesis should explicitly visualize and discuss where the model fails (e.g., boundaries, high-frequency turbulence). This is mentioned in the text (Section 4.2.6) but lacks a dedicated figure.

## What Changes
- **Update Plotting Script**: Enhance `tools/plot_training_curves.py` (or create `tools/plot_additional_results.py`) to generate:
    - **Figure 4-7 (Rollout Error)**: A line plot of Rel-L2 error vs. Time Step (1 to T) for EDSR vs UNet vs FNO on the DRD dataset.
    - **Figure 4-8 (Failure Cases)**: A visual comparison of "Best Case" vs "Worst Case" reconstructions, highlighting boundary artifacts or smoothing in turbulent regions.
- **Update Manuscript**: Insert these figures into `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md` with appropriate captions and discussion.
- **Refine Table Numbering**: Fix the duplicate table numbering (e.g., multiple Table 4-5s) observed in the manuscript.

## Impact
- **Affected Specs**: None.
- **Affected Code**: `thesis_paper/manuscript_5_chapter/chapter4_results_verification.md`.
- **New Artifacts**: `fig4-7_rollout_error.png`, `fig4-8_failure_cases.png`.

## ADDED Requirements
### Requirement: Temporal Rollout Error Plot
The system SHALL generate a plot showing the accumulation of reconstruction error over autoregressive time steps.
#### Scenario: Visualization
- **WHEN** the plotting script is run
- **THEN** it generates `fig4-7_rollout_error.png` showing the error trend over time.

### Requirement: Failure Case Visualization
The system SHALL generate a figure highlighting specific failure modes (e.g., boundary errors).
#### Scenario: Visualization
- **WHEN** the plotting script is run
- **THEN** it generates `fig4-8_failure_cases.png` with zoomed-in views of artifacts.

### Requirement: Table Renumbering
The system SHALL renumber all tables in Chapter 4 sequentially to resolve duplicates.
#### Scenario: Formatting
- **WHEN** the manuscript is updated
- **THEN** tables are numbered 4-1 through 4-N without duplicates.
