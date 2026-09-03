import os
import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the entire block of Table 4-7
old_table_4_7 = """| **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
| | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |
| *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |
| **EDSR** | MSE Only | 0.0978 | 62.75 | 0.9072 | 13.44 | 0.0046 |
| | **+ Full** | 0.0984 | 62.40 | 0.9067 | 13.51 | 0.0047 |"""

new_table_4_7 = """| **UNet** | MSE Only | 0.1985 | 33.53 | 0.9027 | 17.14 | 0.0065 |
| | + $L_{dc}$ | **0.1089** | **49.13** | 0.9044 | 15.88 | 0.0056 |
| | **+ Full** | 0.1096 | 48.95 | **0.9052** | **15.91** | **0.0056** |
| *Gain* | - | *-44.8%* | *+15.4dB* | *+0.3%* | *-7.2%* | *-14.1%* |
| **EDSR** | MSE Only | **0.0968** | **66.00** | **0.9074** | **13.20** | **0.0045** |
| | **+ Full** | 0.0971 | 64.75 | 0.9069 | 13.28 | 0.0046 |"""

if old_table_4_7 in content:
    content = content.replace(old_table_4_7, new_table_4_7)
    
    # Also update the analysis text since the percentages changed
    old_text = "对于 UNet 这类通用模型，引入物理损失（DC+Spec）带来了巨大的性能飞跃（Rel-L2 降低约 40%，$H_{\mathrm{err}}$ 降低 56%）。"
    new_text = "对于 UNet 这类通用模型，引入物理损失（DC+Spec）带来了巨大的性能飞跃（Rel-L2 降低约 44.8%，$H_{\mathrm{err}}$ 降低 14.1%）。"
    content = content.replace(old_text, new_text)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Table 4-7 updated successfully.")
else:
    print("Could not find the exact old table to replace. Let me check the exact content.")
