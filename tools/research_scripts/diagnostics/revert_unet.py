import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# The user explicitly asked "这个批次的unet数据对吗，有没有之前的", "能找到之前的unet，这个最新的被我改动过代码"
# This means my substitution with 0.1758 and 0.4559 etc. was entirely incorrect because those UNet models were tampered with.
# The user wants me to revert back to the ORIGINAL UNet data!
# The original UNet data (from Table 4-7) before my edits:
# | **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
# | | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
# | | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |

# Wait, let's restore the entire UNet table block to the original.
# But keep EDSR intact as the user didn't mention it. Wait, EDSR was also modified to 0.3379. 
# Let's revert the ENTIRE Table 4-7 and analysis to its original state, and ONLY ADD the EDSR RecSpec if needed, or just revert it fully to what it was.

# Original Table 4-7 block:
original_table_and_text = """**表 4-7 损失函数消融 (SR $\\times 4$)**

| 模型 | 损失组合 | Rel-L2 $\\downarrow$ | PSNR $\\uparrow$ | SSIM $\\uparrow$ | fRMSE-Low $\\downarrow$ | $H_{\\mathrm{err}}$ $\\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |
| | + $L_{dc}$ | 0.1089 | 49.13 | 0.9044 | 15.88 | 0.0056 |
| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |
| *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |
| **EDSR** | MSE Only | 0.0978 | 62.75 | 0.9072 | 13.44 | 0.0046 |
| | **+ Full** | 0.0984 | 62.40 | 0.9067 | 13.51 | 0.0047 |

**结果与机理解析**：
1.  **物理感知损失对弱骨干的有效改善**：对于 UNet 这类通用模型，引入物理损失（DC+Spec）带来了巨大的性能飞跃（Rel-L2 降低约 40%，$H_{\mathrm{err}}$ 降低 56%）。
2.  **强骨干的“内隐”一致性**：对于 EDSR，引入额外 Loss 后 $H_{\mathrm{err}}$ 变化微乎其微。这揭示了优秀的残差网络架构本身就具备极强的拟合观测数据的能力，引入物理损失的价值在于规范未观测区域的物理行为。"""

# Find current block
match = re.search(r"\*\*表 4-7 损失函数消融.*?物理场重建。", content, flags=re.DOTALL)
if match:
    content = content.replace(match.group(0), original_table_and_text)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Reverted to the original Table 4-7.")
else:
    print("Could not find the current Table 4-7 block.")
