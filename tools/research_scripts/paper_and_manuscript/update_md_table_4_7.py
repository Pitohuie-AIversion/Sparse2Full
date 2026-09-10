import os
import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Current table 4-7 has:
# | **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
# We need to replace it with real metrics.
# Wait, let's get the exact values first.
