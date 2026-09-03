import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the current UNet section with the old one
old_text = """| **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
| | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |
| *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |"""

new_text = """| **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
| | + $L_{spec}$ | 0.1092 | 49.02 | 0.9048 | 14.56 | 0.0058 |
| | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |
| *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |"""

# But wait, did I run UNet + L_spec ? Let's check `runs_drd_paper/AR-DR2D-UNet-SRx4-Consistent-Sigma1.0-model_UNet-s2025-20260115` vs NoSpec.
