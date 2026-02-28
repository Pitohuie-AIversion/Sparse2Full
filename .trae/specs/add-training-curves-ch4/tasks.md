# Tasks

- [x] Task 1: Create a plotting script `tools/plot_training_curves.py`
  - [x] SubTask 1.1: Implement functions to parse `training_history.json` and extract epochs, loss, and metrics.
  - [x] SubTask 1.2: Implement plotting logic using `matplotlib` to generate the 3 specific figures (Convergence, Sequential, Ablation).
  - [x] SubTask 1.3: Configure the script to target specific log files found in `runs/` (e.g., `runs/AR-SW-10M-EDSR...`, `runs/AR-DR2D-Sequential...`).

- [x] Task 2: Generate the figures
  - [x] SubTask 2.1: Run `tools/plot_training_curves.py` to generate `fig4-4_training_convergence.png`, `fig4-5_sequential_evolution.png`, and `fig4-6_ablation_curves.png`.
  - [x] SubTask 2.2: Verify the generated figures are saved in `thesis_paper/manuscript_5_chapter/images/`.

- [x] Task 3: Update `chapter4_results_verification.md`
  - [x] SubTask 3.1: Insert `fig4-4_training_convergence.png` into Section 4.2.1 with caption.
  - [x] SubTask 3.2: Insert `fig4-5_sequential_evolution.png` into Section 4.3.2 with caption.
  - [x] SubTask 3.3: Insert `fig4-6_ablation_curves.png` into Section 4.4.1 with caption.
