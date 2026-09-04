
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter0_notation.md"

def fix_notation_table():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    changes_count = 0
    
    for line in lines:
        original_line = line
        
        # Target specific broken separators: " \| " followed by Type keywords
        # 1. " \| 连续场 \| " -> " | 连续场 | "
        if " \\| 连续场 \\| " in line:
            line = line.replace(" \\| 连续场 \\| ", " | 连续场 | ")
            
        # 2. " \| 算子 \| " -> " | 算子 | "
        if " \\| 算子 \\| " in line:
            line = line.replace(" \\| 算子 \\| ", " | 算子 | ")
            
        # 3. " \| 函数 \| " -> " | 函数 | "
        if " \\| 函数 \\| " in line:
            line = line.replace(" \\| 函数 \\| ", " | 函数 | ")
            
        # Also check for single occurrences if they are not paired
        # e.g. " \| 算子 " but not followed by " \| " immediately?
        # The pattern seems to be `| ... \| Type \| ... |`
        
        if line != original_line:
            changes_count += 1
            print(f"Fixed: {original_line.strip()} -> {line.strip()}")
            
        new_lines.append(line)

    if changes_count > 0:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed {changes_count} lines in chapter0_notation.md")
    else:
        print("No lines matched the fix patterns.")

if __name__ == "__main__":
    fix_notation_table()
