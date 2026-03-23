
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter5.md"

def fix_chapter5_residual():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Error 1: u^{(i}) -> u^{(i)}
    # Context: "对随机抽样的真值 $u^{(i)}："
    # Wait, my previous read showed `u^{(i)}`. 
    # But user says `ParseError: KaTeX parse error: Expected '}', got 'EOF' at end of input: u^{(i})`
    # This suggests the content is `$u^{(i` or `$u^{(i}` without `)}`.
    # Let's be aggressive with regex.
    # Replace `u^{(i})` (if exists) with `u^{(i)}`
    content = content.replace(r'u^{(i})', r'u^{(i)}')
    
    # Replace `$u^{(i}$` with `$u^{(i)}$` (if braces are weirdly placed)
    content = content.replace(r'$u^{(i}$', r'$u^{(i)}$')
    
    # Error 2: y^{(i}) -> y^{(i)}
    content = content.replace(r'y^{(i})', r'y^{(i)}')
    content = content.replace(r'$y^{(i}$', r'$y^{(i)}$')

    # Error 3: \hat y^{i}=Hu^{(i)}
    # User error: `\hat y^{i...}=Hu^{(i...)}`
    # Previous read: `\hat y^{(i)}=H(u^{(i)})` (line 108)
    # Maybe the user's error message refers to `\hat y^{i` (missing brace for superscript?)
    # Or `\hat y^{i` where `i` is not wrapped? `\hat y^i` is valid but `\hat y^{i` without `}` is not.
    # Let's fix `\hat y^{i` -> `\hat y^{(i)}` if it looks like that.
    
    # Check specifically for `\hat y^{$i$}` which I fixed to `\hat y^{(i)}`.
    # Maybe there is a `\hat y^{i` somewhere else?
    
    # Error 4: \mathrm{MSE}\left$\hat y^{$i$}, y...
    # My previous fix: `\mathrm{MSE}\left(\hat y^{(i)}`
    # Let's ensure no `$` inside `\left`
    
    # Additional manual check for `$` inside math blocks or missing braces
    # Pattern: `^{(i}` without `)}`
    
    # Let's blindly fix specific error strings reported by user if they match
    
    # Case: `u^{(i})`
    # If the file has `$u^{(i}$)` -> `u^{(i})` inside `$`? No.
    # If the file has `u^{(i})` in text?
    
    # Let's look at the lines 105-115 again from previous tool output:
    # 105: ... $u^{(i)}：
    # 107: ... $y^{(i)}；
    # 108: ... $\hat y^{(i)}=H(u^{(i)})；
    # 112: \mathrm{MSE}\left(\hat y^{(i)}, y^{(i)}\right)<\varepsilon
    
    # These look correct in the tool output. 
    # Why does the user still see error?
    # Maybe `i` inside `^{(i)}` is causing issues if `i` is interpreted as text? No, `i` is fine.
    # Maybe the Chinese colon `：` or semicolon `；` immediately after `$`?
    # `$u^{(i)}：` -> The `$` ends at `)}`. `：` is outside. This is fine.
    
    # Wait, the user error says `Expected '}', got 'EOF' at end of input: u^{(i})`.
    # This means `u^{(i})` is being parsed. 
    # `u^{(i})` implies `}` is missing for `{`. 
    # `^{(i}` -> `{` opens, `(` char, `i` char. `}` is expected.
    # If I have `u^{(i)}`, it is fine.
    # If I have `u^{(i})`, then `{` opens, `(` char, `i` char, `}` closes `{`, `)` char. 
    # `u^{(i})` is valid LaTeX: superscript is `(i`. Then `)` follows.
    # But if it is `$u^{(i})`? -> superscript `(i`. Then `)`. 
    # Why "Expected '}', got 'EOF'"?
    # That usually happens if `{` is not closed. e.g. `$u^{(i$`
    
    # Let's check if there are any unclosed braces.
    # Or maybe the user is referring to a different location?
    
    # Let's use a very specific replace for what the user likely sees vs what it should be.
    # User sees: `u^{(i})`
    # I will replace `u^{(i})` with `u^{(i)}` globally just in case.
    # Also `y^{(i})` -> `y^{(i)}`.
    
    # Also `\hat y^{i` -> `\hat y^{(i)}` (if `i` is single char, brace optional, but if it is `(i)` it needs braces).
    # If it is `\hat y^{i}` it is fine.
    # If it is `\hat y^{i)}` -> superscript `i)`.
    
    # Let's search for specific broken patterns mentioned.
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter5.md residual errors")
    else:
        # Force rewrite if we suspect hidden chars or just to be safe?
        # No, let's try to find the exact string.
        pass

    # Let's scan for `u^{(i})`
    if 'u^{(i})' in content:
        content = content.replace('u^{(i})', 'u^{(i)}')
        print("Found and fixed u^{(i})")

    if 'y^{(i})' in content:
        content = content.replace('y^{(i})', 'y^{(i)}')
        print("Found and fixed y^{(i})")
        
    # Check for `\hat y^{i`
    # User said: `\hat y^{iParseError`...
    # Maybe `\hat y^{i` is missing `}`
    # Scan for `\hat y^{i` followed by something other than `}`
    # Or just `\hat y^{i}` vs `\hat y^{(i)}`
    
    # Actually, look at line 108 in file: `\hat y^{(i)}=H(u^{(i)})`
    # This looks correct.
    
    # What if the user is seeing `\hat y^{$i$}`?
    # I replaced `\hat y^{$i$}` with `\hat y^{(i)}` in previous script.
    
    # Maybe the error is in the line 112 `\mathrm{MSE}`?
    # Line 112: `\mathrm{MSE}\left(\hat y^{(i)}, y^{(i)}\right)<\varepsilon`
    # This also looks correct.
    
    # Is it possible there are hidden invisible characters?
    # Or the user is providing an error log from a *previous* run?
    # "没有修复好还是有报错" implies they ran it again and it failed.
    # Or they are looking at the file in IDE and it shows red?
    
    # Let's assume there might be some `$` inside `\left` that I missed?
    # I did `content.replace(r'\mathrm{MSE}\left$\hat y^{$i$}', ...)`
    # If the spaces were different, it wouldn't match.
    
    # Let's read the file again very carefully with `cat -A` equivalent or just print exact repr around line 105.
    
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_chapter5_residual()
