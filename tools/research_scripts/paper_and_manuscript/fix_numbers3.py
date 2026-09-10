import os
filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We need to make sure the newly inserted plot is Fig 4-3, and the next one is 4-4, etc.
# Let's just find and replace the incorrect ones.
content = content.replace("![图 4-3: 不同观测面积", "![图 4-3_TEMP: 不同观测面积")
content = content.replace("![图 4-5: 时空预测", "![图 4-4: 时空预测")
content = content.replace("图 4-6 展示了典型", "图 4-5 展示了典型")
content = content.replace("![图 4-6: 典型测试样本", "![图 4-5: 典型测试样本")
content = content.replace("![图 4-7: 重建结果", "![图 4-6: 重建结果")
content = content.replace("![图 4-8: 典型失败案例", "![图 4-7: 典型失败案例")
content = content.replace("![图 4-9: 损失函数", "![图 4-8: 损失函数")
content = content.replace("![图 4-10: 序列化课程", "![图 4-9: 序列化课程")
content = content.replace("![图 4-11: 不同模型架构", "![图 4-10: 不同模型架构")
content = content.replace("![图 4-3_TEMP: 不同观测面积", "![图 4-3: 不同观测面积")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
