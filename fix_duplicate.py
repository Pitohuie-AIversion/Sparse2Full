import re

files = [
    'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md',
    'thesis_paper/manuscript_5_chapter/thesis_full.md'
]

pattern = re.compile(r'\*\*分析\*\*：仅在无噪数据上训练的模型对高频噪声较敏感。此外，在 \*\*2D Darcy Flow\*\* 数据集上的验证表明，“统一观测算子 \+ 残差骨干”的范式具备处理不同物理机制的通用潜力。.*?(?=### 4\.4\.2)', re.DOTALL)

correct_text = """**分析**：仅在无噪数据上训练的模型对高频噪声较敏感。此外，在 **2D Darcy Flow** 数据集上的验证表明，“统一观测算子 + 残差骨干”的范式具备处理不同物理机制的通用潜力。

为进一步量化该结论，我们在 Darcy Flow 数据集（SR $\\times 4$ 任务）上进行了多样本跨域验证实验。具体实验定量结果如下表所示：

**表 4-11 2D Darcy Flow 数据集跨域验证结果**

| 模型 | Rel-L2 | MAE | PSNR (dB) | SSIM | DC Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EDSR | 0.0331 | 0.0128 | 38.79 | 0.9945 | 0.0013 |

从上述数据可以观察到，基于 EDSR 的统一架构在该数据集上表现出了极低的相对 L2 误差（0.0331）和极高的峰值信噪比（38.79 dB）。对应的多个测试样本的可视化预测结果（从左至右依次为：样本 2、样本 1 和样本 9192）如下图所示：

![Darcy Flow 验证多样本可视化](../../paper_package/figs/DarcyFlow/darcy_flow_verification_horizontal.pdf)

结果表明，模型不仅能够精准恢复 Darcy Flow 复杂的全局物理结构与局部细节，而且重建误差（Error）被控制在极小的范围内。这有力地证明了本文提出的观测与重构范式在跨物理域任务中具有良好的适用性和结构层面的泛化能力。

"""

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace duplicate sections
    content = pattern.sub(correct_text, content)
    
    # Fix the table numbers if needed
    # We want Extreme Sparsity table to be 4-12
    content = re.sub(r'\*\*表 4-13 极度稀疏能力扫描\*\*', '**表 4-12 极度稀疏能力扫描**', content)
    content = re.sub(r'\*\*表 4-13 模型资源效率对比', '**表 4-13 模型资源效率对比', content) # ensure it is 13

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Duplicates fixed successfully.")
