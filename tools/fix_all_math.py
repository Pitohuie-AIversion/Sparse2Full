import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review")

def fix_math_in_content(content):
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Fix Inline Math: (...) -> $...$
        # Reuse logic from previous scripts
        def process_inline(text_inside_parens):
            # Check for LaTeX indicators
            if any(x in text_inside_parens for x in ['\\', '_', '^', '=', '<', '>']) and not text_inside_parens.startswith('http'):
                 # Avoid replacing (1), (a), (2022)
                 if re.match(r'^\d+$', text_inside_parens) or re.match(r'^[a-z]$', text_inside_parens) or re.match(r'^\d{4}$', text_inside_parens):
                     return f"({text_inside_parens})"
                 
                 # Fix * -> _ inside math if any
                 fixed_text = text_inside_parens.replace('*{', '_{')
                 fixed_text = re.sub(r'\*([a-zA-Z])', r'_\1', fixed_text)
                 
                 return f"${fixed_text}$"
            return f"({text_inside_parens})"

        def callback(match):
            full = match.group(0)
            if full.startswith(']'):
                return full
            else:
                inner = full[1:-1]
                # Handle double parens ((...)) -> ($...$)
                if inner.startswith('(') and inner.endswith(')'):
                    inner_stripped = inner[1:-1]
                    return f"({process_inline(inner_stripped)})"
                return process_inline(inner)

        # Pattern matches ](anything) OR (anything)
        new_line = re.sub(r'\]\([^)]*\)|\([^)]*\)', callback, line)
        
        # 2. Fix Norms: |...|_2 -> \|...\|_2
        def fix_norms_in_math(match):
            m = match.group(0)
            if '_2' in m or '\|' not in m:
                # Replace | with \| if not already escaped
                fixed = re.sub(r'(?<!\\)\|', r'\|', m)
                return fixed
            return m

        # Apply to $$...$$
        new_line = re.sub(r'\$\$[^$]+\$\$', fix_norms_in_math, new_line)
        # Apply to $...$
        new_line = re.sub(r'\$[^$]+\$', fix_norms_in_math, new_line)
        
        new_lines.append(new_line)

    return '\n'.join(new_lines)

def fix_file(filename):
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    fixed_content = fix_math_in_content(content)
    
    if fixed_content != content:
        print(f"Fixed math in {filename}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
    else:
        pass

if __name__ == "__main__":
    # Get all .md files
    md_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
    
    print(f"Scanning {len(md_files)} files in {DOCS_DIR}...")
    for f in md_files:
        fix_file(f)
    print("Done.")
