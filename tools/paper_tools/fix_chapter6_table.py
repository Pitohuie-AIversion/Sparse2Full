
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter6.md"

def fix_chapter6_table():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix the table header in Table 6-4
    # Old: | 噪声水平 $\sigma_n$ \| Rel-L2 (Mean) $\downarrow$ | Std | 性能衰减幅度 |
    # New: | 噪声水平 $\sigma_n$ | Rel-L2 (Mean) $\downarrow$ | Std | 性能衰减幅度 |
    
    # We use replace with string literals to be safe
    old_str = r"| 噪声水平 $\sigma_n$ \| Rel-L2 (Mean) $\downarrow$ | Std | 性能衰减幅度 |"
    new_str = r"| 噪声水平 $\sigma_n$ | Rel-L2 (Mean) $\downarrow$ | Std | 性能衰减幅度 |"
    
    if old_str in content:
        content = content.replace(old_str, new_str)
        print("Fixed Table 6-4 header")
    else:
        print("Table 6-4 header pattern not found. Checking for variations...")
        # Fallback regex if spacing is different
        pattern = r"\|\s*噪声水平\s*\$\\sigma_n\$\s*\\\|\s*Rel-L2"
        if re.search(pattern, content):
             content = re.sub(r"\|\s*噪声水平\s*\$\\sigma_n\$\s*\\\|\s*Rel-L2", r"| 噪声水平 $\\sigma_n$ | Rel-L2", content)
             print("Fixed Table 6-4 header using regex")
        else:
             print("Could not find Table 6-4 header to fix")

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print("No changes made to chapter6.md")

if __name__ == "__main__":
    fix_chapter6_table()
