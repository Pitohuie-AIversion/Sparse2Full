import re

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fig_count = 1
new_lines = []

for line in lines:
    # Match ![图 4-X: ...] or ![图 4-X ...] or 图 4-X 展示了
    # Wait, it's easier to just find all occurrences of "图 4-\d+" and replace them.
    # But wait, there are references in text like "图 4-1", "图 4-2", etc.
    pass

