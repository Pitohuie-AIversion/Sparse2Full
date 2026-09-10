import os

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace Table 4-5 entries
content = content.replace("| **Ours (Seq-EDSR)** |", "| **Seq-EDSR (Backbone)** |")

# Replace text analysis
content = content.replace("Ours (Seq-EDSR) 的累积误差", "Seq-EDSR (Backbone) 的累积误差")
content = content.replace("Ours 在保持参数量", "Seq-EDSR 在保持参数量")
content = content.replace("Ours 通过序列化课程学习", "Seq-EDSR 通过时空联合建模")
content = content.replace("Ours (EDSR)", "Seq-EDSR")
content = content.replace("Ours 模型", "Seq-EDSR 模型")
content = content.replace("Ours 在纹理细节恢复上", "Seq-EDSR 在纹理细节恢复上")
content = content.replace("Ours 在低频段与 GT 高度重合", "Seq-EDSR 在低频段与 GT 高度重合")

# Insert the new DRD plot
drd_vis_text = """

为进一步展示模型在 DRD 数据集上的长时演化稳定性，图 4-6 给出了多时刻定性对比（时间展开图）。

![图 4-6: DRD 数据集典型测试样本在不同演化步长（$t=0, 10, 20, 29$）下的定性对比。相比于基线 UNet 随着时间推移出现的严重模糊与结构崩塌，Seq-EDSR (Backbone) 能够长时间维持图灵斑图的精细物理结构，且误差累积（最底行）显著更低。](../../figures/rollout/qualitative_multistep.png)
"""

target = "1.  **标准图组**：包括真值 (GT)、预测值 (Pred) 及绝对误差 (Error)。Seq-EDSR 在纹理细节恢复上明显优于 UNet，误差分布更均匀。"
if target in content and "DRD 数据集上的长时演化稳定性" not in content:
    content = content.replace(target, target + drd_vis_text)
    
    # Safely bump figure numbers from 4-11 down to 4-6 backward
    content = content.replace("图 4-11:", "图 4-12:")
    content = content.replace("图 4-10:", "图 4-11:")
    content = content.replace("图 4-9:", "图 4-10:")
    content = content.replace("图 4-8:", "图 4-9:")
    content = content.replace("图 4-7:", "图 4-8:")
    # We replace 4-6: with 4-7: EXCEPT the one we just inserted!
    # Let's do it with a regex that ignores our new caption
    # Our new caption is "图 4-6: DRD 数据集典型测试样本在不同演化步长"
    # The old 4-6 was "图 4-6: 重建结果的径向平均功率谱对比"
    content = content.replace("图 4-6: 重建结果", "图 4-7: 重建结果")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated manuscript.")
