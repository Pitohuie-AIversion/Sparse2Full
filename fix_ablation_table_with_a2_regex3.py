import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_text = "EDSR Rel-L2 从 0.3379 降至 0.1885"
new_text = "EDSR Rel-L2 从 0.3379 降至 0.1703"

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated text.")
else:
    print("Could not find the exact text.")
