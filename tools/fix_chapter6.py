
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter6.md"

def fix_chapter6():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix |H$\tilde u$-y|_2 -> \|H(\tilde u)-y\|_2
    # Line 131: H_{\mathrm{err}} \triangleq |H$\tilde u$-y|_2.
    # Also handle variants if any
    content = content.replace(r'|H$\tilde u$-y|_2', r'\|H(\tilde u)-y\|_2')
    
    # 2. Fix \|H$\tilde u$-y\|_2 -> \|H(\tilde u)-y\|_2
    # Line 205: $H_{\mathrm{err}}=\|H$\tilde u$-y\|_2$)
    content = content.replace(r'\|H$\tilde u$-y\|_2', r'\|H(\tilde u)-y\|_2')

    # 3. Fix t_{\mathrm{df}}=____, p=____$$ -> $t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$
    # Line 277: * t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$$
    # It seems the `\_\_\_\_` is already escaped in my read output?
    # "277→* t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$$"
    # But it lacks opening $. And ends with $$.
    # Let's replace the whole line content or regex.
    # Be careful about `_` vs `\_`.
    # If the file actually has `____` (underscores), then it needs fixing.
    # If the file has `\_\_\_\_` it is fine.
    # The error "Can't use function '$' in math mode" suggests `$$` issue?
    # Or maybe `t_{...}` is text mode and then `$$` appears?
    # Let's wrap it in single `$`.
    
    # Pattern: * t_{\mathrm{df}}=...$$
    # We want: * $t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$
    
    # Let's try to match the exact string from line 277
    # Note: `____` might be `____` or `\_\_\_\_`.
    # The user input said: `t_{\mathrm{df}}=____, p=____$$`
    # My read output said: `t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$$`
    # Maybe my read tool escaped it? No, read tool usually outputs raw.
    # So the file has `\_\_\_\_`.
    # But it is missing opening `$`.
    # And has closing `$$`.
    # Replace `t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$$` with `$t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$`
    content = content.replace(r't_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$$', r'$t_{\mathrm{df}}=\_\_\_\_, p=\_\_\_\_$')
    
    # 4. Fix Cohen’s $d=\ ____$ -> Cohen’s $d=\_\_\_\_$
    # Line 278: * Cohen’s $d=\ ____$（并注明是配对差值版本）
    # User error: "Expected group after '_'"
    # This implies `____` (4 underscores) inside math.
    # If it is `d=\ ____`, then `\` escapes space?
    # Then `____` are 4 underscores. First `_` expects group.
    # Fix: `d=\_\_\_\_`
    content = content.replace(r'd=\ ____$', r'd=\_\_\_\_$')
    content = content.replace(r'd=____$', r'd=\_\_\_\_$') # Just in case

    # 5. Additional check for H$\tilde u$
    content = content.replace(r'H$\tilde u$', r'H(\tilde u)')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter6.md")
    else:
        print("No changes needed for chapter6.md (check patterns)")
        # Debug print if needed
        # print(content[270:300]) 

if __name__ == "__main__":
    fix_chapter6()
