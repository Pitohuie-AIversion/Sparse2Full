import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

old_text = """![图 4-2: EDSR 模型在 SWE 超分辨率任务中的综合优势。 (a) 重建图像细节对比，EDSR 能够清晰还原细小涡旋结构；(b) PSNR 与参数量的帕累托前沿，EDSR 在 1M 参数量级下取得最佳精度（*p<0.05）；(c) 误差热图放大区域；(d) 各基线模型的重建误差对比柱状图。](images/fig4_edsr_advantage.png)

![图 4-3: 1M 参数预算下的模型效率与精度权衡图。在相近参数量级下，EDSR 能够提供最高的重建精度（低 Rel-L2），而 ConvUNetLite 则展现出极快的推理速度，体现了不同架构在资源受限场景下的 Pareto 最优边界。圆点大小与模型参数量成正比。](images/fig4_latency_accuracy_tradeoff.png)

为进一步在有限计算预算下筛选最优基线，我们进行了**1M 参数量预算**下的横向对比（表 4-3）。"""

new_text = """为进一步在有限计算预算下筛选最优基线，我们进行了**1M 参数量预算**下的横向对比（表 4-3）。通过图 4-2 的效率与精度权衡图（Trade-off）可以直观地看出，在相近的参数预算下，不同架构在计算速度与重构精度之间展现出不同的 Pareto 最优边界。

![图 4-2: 1M 参数预算下的模型效率与精度权衡图（Efficiency-Accuracy Trade-off）。横轴为模型推理时延（ms），纵轴为相对误差（Rel-L2）。散点大小反映模型的参数量（Params），点越小代表模型越轻量化。虚线连接了处于 Pareto 前沿的模型。可以看出，EDSR 精度最高但时延偏高；ConvUNetLite 速度最快且精度次优；NAFNet 虽精度较好但计算代价过高。](../../figures/edsr/efficiency_accuracy_tradeoff.png)"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replace success!")
else:
    print("Old text not found!")
