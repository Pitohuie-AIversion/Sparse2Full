
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter7.md"

def fix_chapter7_v2():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix RelL2 with $ inside (Baseline)
    # \mathrm{RelL2}^{\text{baseline}$}_j
    # Change to: \mathrm{RelL2}^{\text{baseline}}_j
    content = content.replace(r'\mathrm{RelL2}^{\text{baseline}$}_j', r'\mathrm{RelL2}^{\text{baseline}}_j')
    
    # 2. Fix RelL2 with $ inside (Ours)
    # \mathrm{RelL2}^{\text{ours}$}_j
    # Change to: \mathrm{RelL2}^{\text{ours}}_j
    content = content.replace(r'\mathrm{RelL2}^{\text{ours}$}_j', r'\mathrm{RelL2}^{\text{ours}}_j')
    
    # 3. Fix tag equation block
    # Context: e_{\max}=\max_i e^{(i)}. \tag{7-5}
    # User input error: `ParseError: KaTeX parse error: \tag works only in display equations$`
    # This implies the environment isn't recognized as display math or \tag is inside inline math.
    # My read output shows: $$ ... \tag{7-5} $$
    # This should be valid in standard LaTeX or MathJax/KaTeX display mode.
    # However, if Pandoc converts it to `\[ ... \]`, KaTeX might complain about `\tag` if it's not supported in that specific mode or if it's mixed.
    # But usually `$$` works.
    # Let's check if there are any other `$` interfering.
    
    # 4. Check for `img\_size` which might be escaped
    # Context: `img\_size = 128 / 256 / 512`
    # User input had: `img_size` (no backslash). My read had `img\_size`.
    # `\_` is valid in text.
    
    # 5. Check for `ParseError` strings that might have been pasted into the file?
    # No, user input shows error message *about* the file.
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter7.md v2")
    else:
        print("No changes needed for chapter7.md v2")

if __name__ == "__main__":
    fix_chapter7_v2()
