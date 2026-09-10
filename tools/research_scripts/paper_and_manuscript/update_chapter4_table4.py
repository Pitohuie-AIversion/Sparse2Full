import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("| **112**   | **76.56**    | 0.4668          | -               | 1.0000$^{\dagger}$     |", "| **112**   | **76.56**    | 0.4668          | OOM             | 1.0000$^{\dagger}$     |")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Table updated completely!")
