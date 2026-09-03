import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Target text block
old_text = """| 1 | 0.01 | 1.0055 | 0.9952 | - |

**结果分析**："""

new_text = """| 1 | 0.01 | 1.0055 | 0.9952 | - |

![图 4-3: 不同观测面积（Area Pct）下各架构的重建误差退化曲线。横轴为对数刻度的观测面积占比，纵轴为相对误差（Rel-L2）。随着观测面积的极度萎缩，所有模型均向 Rel-L2 $\approx 1.0$ 的失效极限（红色虚线）逼近。其中，带跳跃连接的 UNet 展现出比纯残差结构的 EDSR 更平滑的降级特性，而 PartialConvUNet 则在所有测试点上均失效。](../../figures/crop_limit/crop_capability_curve.png)

**结果分析**："""

content = content.replace(old_text, new_text)

# We need to shift figure numbers again for figures that come after this new one.
# The newly inserted one is 图 4-3.
# Let's shift existing 4-3 -> 4-4, 4-4 -> 4-5, etc.

# We must be careful because we just inserted 4-3.
# The existing figures were:
# 4-1: 训练收敛曲线对比
# 4-2: 1M 参数预算
# 4-3: 时空预测 (needs to become 4-4)
# 4-4: 典型测试样本 (needs to become 4-5)
# 4-5: 重建结果功率谱 (needs to become 4-6)
# 4-6: 典型失败案例 (needs to become 4-7)
# 4-7: 损失函数消融 (needs to become 4-8)
# 4-8: 序列化课程 (needs to become 4-9)
# 4-9: 资源-精度权衡 (needs to become 4-10)

replacements = [
    ("图 4-9", "图 4-10"),
    ("图 4-8", "图 4-9"),
    ("图 4-7", "图 4-8"),
    ("图 4-6", "图 4-7"),
    ("图 4-5", "图 4-6"),
    ("图 4-4", "图 4-5"),
    ("图 4-3: 时空预测", "图 4-4: 时空预测"),
]

for old, new in replacements:
    content = content.replace(old, new)

# And fix text references:
# The old "图 4-4 展示了典型测试样本" now should be "图 4-5 展示了典型测试样本"
content = content.replace("图 4-4 展示了典型测试样本", "图 4-5 展示了典型测试样本")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Markdown updated!")
