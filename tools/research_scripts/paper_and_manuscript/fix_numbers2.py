import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace references in text too!
content = content.replace("图 4-4 展示了典型", "图 4-4_TEMP 展示了典型")

# Replace Captions
content = content.replace("![图 4-4: 时空预测", "![图 4-3: 时空预测")
content = content.replace("![图 4-5: 典型测试", "![图 4-4: 典型测试")
content = content.replace("图 4-4_TEMP 展示了典型", "图 4-4 展示了典型")

content = content.replace("![图 4-6: 重建结果", "![图 4-5: 重建结果")
content = content.replace("![图 4-7: 典型失败", "![图 4-6: 典型失败")
content = content.replace("![图 4-8: 损失函数", "![图 4-7: 损失函数")
content = content.replace("![图 4-9: 序列化课程", "![图 4-8: 序列化课程")
content = content.replace("![图 4-9: 不同模型", "![图 4-9: 不同模型") # This one is already 4-9, leave it or just keep it 4-9

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("All figure numbers fixed!")
