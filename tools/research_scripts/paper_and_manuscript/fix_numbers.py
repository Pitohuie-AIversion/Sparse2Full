import os
import re

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace sequentially from back to front to avoid overlapping replacements
replacements = [
    ("图 4-10", "图 4-9"),
    ("图 4-9", "图 4-8"),
    ("图 4-8", "图 4-7"),
    ("图 4-7", "图 4-6"),
    ("图 4-6", "图 4-5"),
    ("图 4-5", "图 4-4"),
    ("图 4-4", "图 4-3"),
    ("图 4-3 展示了", "图 4-4 展示了") # The reference to old 4-5 (which was 4-3 in text before)
]

for old, new in replacements:
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Numbering fixed!")
