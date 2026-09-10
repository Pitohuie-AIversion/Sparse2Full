import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's see what is currently in the file.
idx1 = content.find("**表 4-7")
idx2 = content.find("### 4.3.2")
if idx1 != -1 and idx2 != -1:
    print(content[idx1:idx2])
