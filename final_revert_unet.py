import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# We need to revert the analysis text back to original for point 1, because the EDSR analysis was updated to reflect the new findings, but the UNet part should reflect the old findings since the user said UNet was actually modified.
# Wait, the user said: "这个批次的unet数据对吗，有没有之前的", "能找到之前的unet，这个最新的被我改动过代码".
# The user wants to use the ORIGINAL UNet data, which is already in the file:
# | **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
# | | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
# | | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |

# However, the user also asked: "我是不是还有加入Lspec这样控制变量的实验".
# The user wants to know if there is an experiment for UNet with ONLY Lspec.
# Let's check `runs_drd_paper` for any UNet + Lspec.
# But wait, did I check if there is an ablation UNet with Lspec?
