import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# I notice that earlier the replacement for "EDSR Rel-L2 从 0.3379 降至 0.1885" failed
# because the text is actually: "EDSR Rel-L2 从 0.3379 降至 0.1885" ... wait, let's search for 0.1885
if "0.1885" in content:
    content = content.replace("0.1885", "0.1703")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Replaced all 0.1885 with 0.1703.")
else:
    print("Could not find 0.1885.")
