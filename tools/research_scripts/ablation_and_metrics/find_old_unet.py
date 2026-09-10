import os
import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's revert the UNet part back to what it originally was before my modifications today.
# Original UNet data (from git):
# | **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
# | | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
# | | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |
# | *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |

# The problem was: 0.1780 is the rollout error for UNet (from Table 4-5).
# Wait, the user said "能找到之前的unet，这个最新的被我改动过代码".
# The user means the `runs_3loss_ablation_unet_100ep` was NOT the real UNet data, and the code was modified.
# The actual UNet ablation data used in the paper was the one previously there!
# Let's revert the UNet data and EDSR data back to the original values, but keep the EDSR RecSpec line if needed, or just revert everything.

# The user's exact words: "这个批次的unet数据对吗，有没有之前的", "能找到之前的unet，这个最新的被我改动过代码"
