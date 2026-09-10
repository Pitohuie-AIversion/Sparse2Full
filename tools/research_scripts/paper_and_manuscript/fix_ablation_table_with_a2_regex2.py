import re

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the EDSR RecSpec line with the actual data:
# | | + $L_{spec}$ | 0.1885 | 37.39 | 0.8680 | 23.55 | 0.0078 | -> | | + $L_{spec}$ | 0.1703 | 37.86 | 0.8704 | 16.41 | 0.0048 |

old_line = "| | + $L_{spec}$ | 0.1885 | 37.39 | 0.8680 | 23.55 | 0.0078 |"
new_line = "| | + $L_{spec}$ | 0.1703 | 37.86 | 0.8704 | 16.41 | 0.0048 |"

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # Also update the analysis text if it references the old value:
    # "EDSR Rel-L2 从 0.3379 降至 0.1885" -> "EDSR Rel-L2 从 0.3379 降至 0.1703"
    content = content.replace("EDSR Rel-L2 从 0.3379 降至 0.1885", "EDSR Rel-L2 从 0.3379 降至 0.1703")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated EDSR RecSpec metrics.")
else:
    print("Could not find the exact line to replace.")
