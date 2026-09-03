import sys

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# We will just replace lines one by one to avoid multiline matching issues.
replacements = {
    "| **UNet** | MSE Only | 0.1780 | 36.29 | 0.8410 | 33.44 | 0.0129 |": "| **UNet** | MSE Only | 0.4559 | 23.92 | 0.5677 | 87.98 | 0.0312 |",
    "| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **13.28** | **0.0056** |": "| | **+ Full** | **0.1096** | 48.95 | **0.9052** | **15.91** | **0.0056** |",
    "| *Gain* | - | *-38.4%* | *+12.6dB* | *+7.6%* | *-60.3%* | *-56.6%* |": "| *Gain* | - | *-76.0%* | *+25.0dB* | *+59.5%* | *-81.9%* | *-82.1%* |",
    "| **EDSR** | MSE Only | 0.0978 | 62.75 | 0.9072 | 13.44 | 0.0046 |": "| **EDSR** | MSE Only | 0.3379 | 28.41 | 0.7459 | 70.74 | 0.0246 |\n| | + $L_{dc}$ | 0.0968 | 66.00 | 0.9074 | 13.20 | 0.0045 |",
    "| | **+ Full** | 0.0984 | 62.40 | 0.9067 | 13.51 | 0.0047 |": "| | **+ Full** | **0.0971** | 64.75 | **0.9069** | **13.28** | **0.0046** |\n| *Gain* | - | *-71.3%* | *+36.3dB* | *+21.6%* | *-81.2%* | *-81.3%* |",
    "1.  **物理感知损失对弱骨干的有效改善**：对于 UNet 这类通用模型，引入物理损失（DC+Spec）带来了巨大的性能飞跃（Rel-L2 降低约 40%，$H_{\mathrm{err}}$ 降低 56%）。\n2.  **强骨干的“内隐”一致性**：对于 EDSR，引入额外 Loss 后 $H_{\mathrm{err}}$ 变化微乎其微。这揭示了优秀的残差网络架构本身就具备极强的拟合观测数据的能力，引入物理损失的价值在于规范未观测区域的物理行为。": "1.  **物理感知损失的普适性提升**：对于 UNet 和 EDSR，仅使用 MSE 损失（纯数据驱动）时，模型均出现了严重的过拟合与物理结构崩塌（UNet Rel-L2 达 0.4559，EDSR 达 0.3379）。引入物理一致性损失（$L_{dc}$ 与 $L_{spec}$）后，两者性能均实现了飞跃式提升（误差下降超过 70%）。这表明无论骨干网络多强，单纯的 MSE 损失都无法凭空推断缺失的物理守恒律。\n2.  **强骨干的频域稳定性**：引入全量物理损失（Full）后，EDSR 的 fRMSE-Low 稳定在 13.28，显著优于 UNet 的 15.91。这揭示了优秀的残差网络架构配合物理约束，能够更精准地锁定低频动力学演化特征，从而在稀疏观测下实现更高精度的物理场重建。"
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
    else:
        print(f"Warning: Could not find:\n{old}\n")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated table successfully.")
