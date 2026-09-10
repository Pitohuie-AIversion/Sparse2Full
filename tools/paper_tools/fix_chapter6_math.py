import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter6.md"

def fix_math(content):
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Fix Display Math Delimiters: [ ] -> $$
        # Check if line is exactly [ or ] or > [ or > ]
        if stripped in ['[', ']', '> [', '> ]']:
            # Preserve indentation/quoting
            prefix = line[:line.find('[')] if '[' in line else line[:line.find(']')]
            new_lines.append(f"{prefix}$$")
            continue
            
        # 2. Fix Inline Math: (...) -> $...$
        # Heuristic: if (...) contains latex-like symbols (\, _, ^, =) and is not a markdown link ](...)
        # We need to be careful not to break text like (see Fig. 1).
        
        def process_inline(text_inside_parens):
            # Check for LaTeX indicators
            if any(x in text_inside_parens for x in ['\\', '_', '^', '=', '<', '>']) and not text_inside_parens.startswith('http'):
                 # Also avoid replacing (1) or (a) or (2022)
                 if re.match(r'^\d+$', text_inside_parens) or re.match(r'^[a-z]$', text_inside_parens) or re.match(r'^\d{4}$', text_inside_parens):
                     return f"({text_inside_parens})"
                 
                 # Fix * -> _ inside math
                 # Specifically *{...} -> _{...} and *word -> _word
                 fixed_text = text_inside_parens.replace('*{', '_{')
                 fixed_text = re.sub(r'\*([a-zA-Z])', r'_\1', fixed_text)
                 
                 return f"${fixed_text}$"
            return f"({text_inside_parens})"

        def callback(match):
            full = match.group(0)
            if full.startswith(']'):
                # This is part of a link ](...), return as is
                return full
            else:
                # This is (...), process it
                # Strip parens
                inner = full[1:-1]
                return process_inline(inner)

        # Pattern matches ](anything) OR (anything)
        # Simple regex for non-nested parens: \([^)]*\)
        new_line = re.sub(r'\]\([^)]*\)|\([^)]*\)', callback, line)
        
        # 4. Fix specific known bad patterns if replace_inline missed them
        # e.g. \mathcal{F}* -> \mathcal{F}_
        new_line = new_line.replace(r'\mathcal{F}*', r'\mathcal{F}_')
        new_line = new_line.replace(r'\mathrm{fRMSE}*', r'\mathrm{fRMSE}_')
        new_line = new_line.replace(r'\overline{\mathrm{Rel\text{-}L2}}*', r'\overline{\mathrm{Rel\text{-}L2}}_')
        
        new_lines.append(new_line)

    return '\n'.join(new_lines)

if __name__ == "__main__":
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed_content = fix_math(content)
    
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed math in {FILE_PATH}")
