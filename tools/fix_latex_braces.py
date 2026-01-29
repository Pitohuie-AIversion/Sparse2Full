import os
import re

PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"

FILES_TO_FIX = [
    "thesis_paper/manuscript_gpt_review/chapter6.md",
    "thesis_paper/manuscript_gpt_review/chapter7.md",
    "thesis_paper/manuscript_gpt_review/chapter3.md",
    "thesis_paper/manuscript_gpt_review/chapter5.md",
    "thesis_paper/manuscript_gpt_review/symbol_checklist.md",
    "thesis_paper/manuscript_gpt_review/latex/manuscript/chapters/chap06_discussion.tex",
    "thesis_paper/manuscript_gpt_review/latex/manuscript/chapters/chap03_method.tex"
]

def fix_file(rel_path):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(abs_path):
        print(f"Warning: File not found {abs_path}")
        return

    with open(abs_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: u^{(z OR \hat u^{(z
    # We want to ensure it's u^{(z)}
    # Regex lookahead to see if } is missing
    # But simpler: replace u^{(z with u^{(z)} globally, then fix double }} if any?
    # No, u^{(z} might be followed by something else.
    
    # Let's use regex substitution.
    # Pattern: u\^\{\(z(?!\))
    # This matches u^{(z NOT followed by )
    # Wait, the error is missing }, so it is u^{(z} -> u^{(z)} ?? 
    # The grep output showed: u^{(z}=...
    # So it is missing the closing brace AND paren? Or just brace?
    # Usually z-score is denoted u^{(z)}. 
    # The error says: Expected '}', got 'EOF' ... u^{(z
    # So the source has `u^{(z`. It needs `u^{(z)}` or at least `u^{(z}` -> `u^{(z)}` if `)` is part of the superscript.
    
    # Let's assume the intent is u^{(z)}.
    
    # Case 1: u^{(z} -> u^{(z)}
    # We replace `u^{(z` with `u^{(z)}`
    # But be careful if it is already correct.
    
    new_content = content
    
    # Fix u^{(z
    # match u or \hat u or \tilde u followed by ^{ (z
    # We replace `^{(z` with `^{(z)}` IF not followed by `)}`
    
    # Actually, simpler: replace `^{(z` with `^{(z)}` and then clean up `^{(z)}}` -> `^{(z)}` just in case?
    # Or use regex:
    
    # Replace u^{(z} (where } is missing or present but ) is missing)
    # The error snippet: u^{(z}=\frac...
    # It clearly misses the closing parenthesis AND brace? Or just brace?
    # `^{(z` opens a brace `{`, then `(`, then `z`. It needs `)}` to close.
    
    # Strategy: Replace `u^{(z` with `u^{(z)}` globally.
    # Also `\hat u^{(z` etc. -> so replace `^{(z` with `^{(z)}`
    
    # But wait, what if it was `u^{(z)}` already?
    # `u^{(z)}` -> `u^{(z)}` (no change if we are careful)
    # `u^{(z` -> `u^{(z)}`
    
    # Let's try replacing `^{(z}` with `^{(z)}` first (if } exists but ) missing)
    # And `^{(z` (if } missing)
    
    # Actually, looking at the error: `u^{(z}=\frac`
    # It seems the text is literal `u^{(z}`.
    
    # Step 1: Replace `^{(z}` with `^{(z)}` (fix missing paren if brace exists)
    # Step 2: Replace `^{(z` with `^{(z)}` (fix missing brace and paren)
    # But Step 2 might match the prefix of Step 1's result.
    
    # Better: regex `\^\{\(z(?!\)\})` -> replace with `^{(z)}`
    # Matches `^{(z` NOT followed by `)}`
    
    new_content = re.sub(r'\^\{\(z(?!\)\})', r'^{(z)}', new_content)
    
    # Also check for variations like `u^{(z)` (missing brace?) -> `^{(z)}`
    # Regex `\^\{\(z\)(?!\})` -> replace with `^{(z)}`
    new_content = re.sub(r'\^\{\(z\)(?!\})', r'^{(z)}', new_content)

    if new_content != content:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {rel_path}")
    else:
        print(f"No changes in {rel_path}")

if __name__ == "__main__":
    for f in FILES_TO_FIX:
        fix_file(f)
