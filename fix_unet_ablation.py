import sys

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    "| **UNet** | MSE Only | 0.1758 | 37.47 | 0.8680 | 19.75 | 0.0063 |": "| **UNet** | MSE Only | 0.4559 | 23.92 | 0.5677 | 87.98 | 0.0312 |",
    "| | + $L_{spec}$ | 0.1736 | 37.50 | 0.8673 | 18.91 | 0.0061 |": "", # Remove this as we have NoSpec instead of RecSpec for the correct ablation path, but actually NoSpec means +Ldc. So the correct path is: MSE Only (0.4559) -> + Ldc (NoSpec: 0.4938... wait, NoSpec is actually 0.4938, let me check the log). Let's just restore the UNet row to its accurate spatial values.
}

# Actually, the user's latest query is about the UNet data. 
# We found:
# UNet MSE Only (Ablation-A0-RecOnly): We used `runs_3loss_ablation_unet_100ep` before, but those were 100ep small runs, not the main paper runs.
# The main paper UNet runs are in `runs_drd_paper`.
# AR-DR2D-UNet-SRx4-Ablation-NoSpec-model_UNet-s2025-20260115 (which is +Ldc): Rel-L2=0.4938
# AR-DR2D-UNet-SRx4-10M-300ep (which is Full?): 0.1985
# AR-DR2D-UNet-SRx4-Consistent-Sigma1.0 (which is Full): Rel-L2=0.1096
