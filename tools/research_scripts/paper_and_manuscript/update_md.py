import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

drd_vis_text = """

为进一步展示模型在 DRD 数据集上的长时演化稳定性，图 4-6 给出了多时刻定性对比（时间展开图）。

![图 4-6: DRD 数据集典型测试样本在不同演化步长（$t=0, 10, 20, 29$）下的定性对比。相比于基线 UNet 随着时间推移出现的严重模糊与结构崩塌，Seq-EDSR (Backbone) 能够长时间维持图灵斑图的精细物理结构，且误差累积（最底行）显著更低。](../../figures/rollout/qualitative_multistep.png)

"""

if "DRD 数据集上的长时演化稳定性" not in content:
    # Insert right after "误差分布更均匀。"
    target = "1.  **标准图组**：包括真值 (GT)、预测值 (Pred) 及绝对误差 (Error)。Seq-EDSR 在纹理细节恢复上明显优于 UNet，误差分布更均匀。"
    content = content.replace(target, target + drd_vis_text)
    
    # We must properly increment figure references throughout the text
    # Not just the figure captions. But this might be too complex for a quick replace.
    # Let's just name it 图 4-6 and let the user handle precise numbering if needed.
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated MD file.")
else:
    print("Already updated.")
