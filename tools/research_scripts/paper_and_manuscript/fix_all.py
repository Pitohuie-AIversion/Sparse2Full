import os
import re

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Restore user's Fig 4-1 text
content = content.replace(
    "![图 4-1: SWE 数据集上不同架构的训练收敛曲线对比。EDSR (Ours) 展现出更快的收敛速度与更低的稳态误差，显著优于 UNet 与 FNO 基线。](images/fig4-4_training_convergence.png)",
    "![图 4-1: SWE 数据集上不同架构的训练收敛曲线对比。EDSR (Ours) 展现出更快的收敛速度与更低的稳态误差，显著优于 UNet、FNO 以及其他拓展基线模型。](images/fig4-4_training_convergence_extended.png)"
)

# Insert Fig 4-2 and text
old_text = "为进一步在有限计算预算下筛选最优基线，我们进行了**1M 参数量预算**下的横向对比（表 4-3）。\n\n**表 4-3"
new_text = """为进一步在有限计算预算下筛选最优基线，我们进行了**1M 参数量预算**下的横向对比（表 4-3）。通过图 4-2 的效率与精度权衡图（Trade-off）可以直观地看出，在相近的参数预算下，不同架构在计算速度与重构精度之间展现出不同的 Pareto 最优边界。

![图 4-2: 1M 参数预算下的模型效率与精度权衡图（Efficiency-Accuracy Trade-off）。横轴为模型推理时延（ms），纵轴为相对误差（Rel-L2）。散点大小反映模型的参数量（Params），点越小代表模型越轻量化。虚线连接了处于 Pareto 前沿的模型。可以看出，EDSR 精度最高但时延偏高；ConvUNetLite 速度最快且精度次优；NAFNet 虽精度较好但计算代价过高。](../../figures/edsr/efficiency_accuracy_tradeoff.png)

**表 4-3"""
content = content.replace(old_text, new_text)

# Shift remaining figures: 4-8 down to 4-2 becomes 4-9 down to 4-3
replacements = [
    ("图 4-8", "图 4-9"),
    ("图 4-7", "图 4-8"),
    ("图 4-6", "图 4-7"),
    ("图 4-5", "图 4-6"),
    ("图 4-4", "图 4-5"),
    ("图 4-3", "图 4-4"),
    ("图 4-2: 时空预测", "图 4-3: 时空预测"),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Fix applied successfully!")
