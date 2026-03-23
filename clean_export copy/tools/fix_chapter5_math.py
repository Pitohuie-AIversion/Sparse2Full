import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter5.md"

def fix_math(content):
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Fix Inline Math: (...) -> $...$
        # Logic: if (...) contains latex-like symbols (\, _, ^, =) and is not a markdown link ](...)
        # Reuse logic from fix_chapter6_math.py
        
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
                    # Recurse? Or just strip?
                    # ((y_t, u_t)) -> inner is (y_t, u_t)
                    # We return ( + process_inline(inner_stripped) + )
                    inner_stripped = inner[1:-1]
                    return f"({process_inline(inner_stripped)})"
                return process_inline(inner)

        # Pattern matches ](anything) OR (anything)
        new_line = re.sub(r'\]\([^)]*\)|\([^)]*\)', callback, line)
        
        # 2. Fix Norms: |...|_2 -> \|...\|_2
        # Use regex to find |...|_2 and replace | with \|
        # Be careful not to replace text | (like in table)
        # We only do this if it looks like math context (inside $...$ or $$...$$)
        # But here we are processing line by line.
        # Since standard markdown math uses $, we can find $...$ blocks and modify inside.
        
        def fix_norms_in_math(match):
            m = match.group(0) # the whole math string including $
            # Replace | with \| if followed by _2 or if it pairs with another |
            # Simplest: replace | with \| globally inside math if it's L2 norm context
            if '_2' in m or '\|' not in m: # if it uses raw |
                # Replace | with \|, but only if not already \|
                # Regex replace (?<!\\)\| -> \|
                fixed = re.sub(r'(?<!\\)\|', r'\|', m)
                return fixed
            return m

        # Apply to $$...$$
        new_line = re.sub(r'\$\$[^$]+\$\$', fix_norms_in_math, new_line)
        # Apply to $...$
        new_line = re.sub(r'\$[^$]+\$', fix_norms_in_math, new_line)
        
        new_lines.append(new_line)

    return '\n'.join(new_lines)

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        exit(1)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed_content = fix_math(content)
    
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed math in {FILE_PATH}")
