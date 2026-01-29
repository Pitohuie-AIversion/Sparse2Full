import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter6.md"

def fix_file():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: $H_{\mathrm{err}}=|H(\tilde u$-y|_2 
    # -> $H_{\mathrm{err}}=\|H(\tilde u)-y\|_2$
    # The grep output showed: $H_{\mathrm{err}}=|H(\tilde u$-y|_2
    # Note: The original markdown might have `|` instead of `\|`.
    # Let's fix it to `\| ... \|_2`.
    
    # We'll use replace() for robustness if the string is exact
    old_str_1 = r"$H_{\mathrm{err}}=|H(\tilde u$-y|_2"
    new_str_1 = r"$H_{\mathrm{err}}=\|H(\tilde u)-y\|_2$"
    
    # Also handle the variant if it was partially fixed or different spacing
    # Regex might be safer
    import re
    # Pattern: H_{err} ... |H(\tilde u ... -y|_2
    # Fix to H(\tilde u)
    
    # Since I saw the exact string in the warning:
    if old_str_1 in content:
        content = content.replace(old_str_1, new_str_1)
        print("Fixed H_err formula")
    else:
        # Try a regex in case of spacing
        # $H_{\mathrm{err}}=|H(\tilde u$-y|_2
        # Look for |H(\tilde u$-y|_2 (spanning the $ boundary?)
        # Wait, the warning said: "Potential unbalanced parentheses in inline math: $H_{\mathrm{err}}=|H(\tilde u$"
        # This means the inline math ended at `u$`.
        # Ah! The original text likely was `$H_{\mathrm{err}}=|H(\tilde u$-y|_2` ?? 
        # No, markdown parser sees `$H_{\mathrm{err}}=|H(\tilde u$` as one math block, then `-y|_2` as text?
        # If so, the fix is to merge them.
        
        # Let's search for the text context.
        # "205: $H_{\mathrm{err}}=|H(\tilde u$-y|_2"
        # The content read previously showed:
        # 205→205. **口径同步下降**：$H_{\mathrm{err}}=|H(\tilde u$-y|_2) 与 Rel-L2 同步下降；
        # So the string is `$H_{\mathrm{err}}=|H(\tilde u$-y|_2)`
        # The `$` inside is likely a typo for `)` or just closing `$` too early?
        # Actually, it looks like: `$H_{\mathrm{err}}=|H(\tilde u$-y|_2`
        # The `$` after `u` closes the math. Then `-y|_2` follows.
        # We want `$H_{\mathrm{err}}=\|H(\tilde u)-y\|_2$`
        pass

    # Let's simple replace the whole line segment found in the file read
    # Line 205: **口径同步下降**：$H_{\mathrm{err}}=|H(\tilde u$-y|_2) 与 Rel-L2 同步下降；
    target_205 = r"$H_{\mathrm{err}}=|H(\tilde u$-y|_2)"
    fixed_205 = r"$H_{\mathrm{err}}=\|H(\tilde u)-y\|_2$"
    
    if target_205 in content:
        content = content.replace(target_205, fixed_205)
        print("Fixed Line 205")
    else:
        # Try regex for that line
        content = re.sub(r'\$H_{\\mathrm\{err\}\}=\|H\(\\tilde u\$-y\|_2\)', r'$H_{\\mathrm{err}}=\\|H(\\tilde u)-y\\|_2$', content)

    # Fix 2: $t(\mathrm{df}$=\ ____\ ,\ p=\ ____)
    # Warning: Potential unbalanced parentheses in inline math: $t(\mathrm{df}$
    # Context Line 277: * $t(\mathrm{df}$=\ ____\ ,\ p=\ ____)
    # The `$` closes after `df`. Then `=\ ____...`
    # Should be `$t(\mathrm{df}=\dots, p=\dots)$`
    
    target_277 = r"$t(\mathrm{df}$=\ ____\ ,\ p=\ ____)"
    fixed_277 = r"$t(\mathrm{df}=\_\_\_\_, p=\_\_\_\_)$"
    
    if target_277 in content:
        content = content.replace(target_277, fixed_277)
        print("Fixed Line 277")
    else:
        # Try regex
        content = re.sub(r'\$t\(\\mathrm\{df\}\$=\\ ____\\ ,\\ p=\\ ____\)', r'$t(\\mathrm{df}=\\_\_, p=\\_\_)$', content)

    # Also check line 45: * 真值（z-score 域）：$u^{(z)}$}=\frac{u-\mu}{\sigma_z})
    # This looks like `u^{(z)}$` followed by `}=\frac...`
    # It has an extra `}` inside? `$u^{(z)}$}`?
    # Context: 45→    * 真值（z-score 域）：$u^{(z)}$}=\frac{u-\mu}{\sigma_z})
    # Wait, the previous fix (fix_latex_braces) might have changed `u^{(z` to `u^{(z)}`.
    # If the original was `u^{(z}`, it became `u^{(z)}`.
    # So now it is `$u^{(z)}` ... `}=\frac...`?
    # The original text was likely `$u^{(z}=\frac...$`
    # If fix_latex_braces changed it to `$u^{(z)}=\frac...$`
    # But look at line 45 in the read output:
    # 45→    * 真值（z-score 域）：$u^{(z)}$}=\frac{u-\mu}{\sigma_z})
    # It has `$u^{(z)}$}=\frac...`
    # The `$` closes after `(z)}`. Then `}=\frac...` is outside math?
    # We want: `$u^{(z)}=\frac{u-\mu}{\sigma_z}$`
    
    # Let's fix this specific line 45 pattern
    # Match: $u^{(z)}$}=\frac{u-\mu}{\sigma_z})
    # Replace with: $u^{(z)}=\frac{u-\mu}{\sigma_z}$
    
    target_45 = r"$u^{(z)}$}=\frac{u-\mu}{\sigma_z})"
    fixed_45 = r"$u^{(z)}=\frac{u-\mu}{\sigma_z}$"
    
    if target_45 in content:
        content = content.replace(target_45, fixed_45)
        print("Fixed Line 45")
    
    # Similarly line 46: * 预测（原值域）：$\tilde u=\sigma_z \hat u^{(z)}$}+\mu)
    # It has `\hat u^{(z)}$}+\mu)`
    # Should be `\tilde u=\sigma_z \hat u^{(z)}+\mu`
    target_46 = r"$\tilde u=\sigma_z \hat u^{(z)}$}+\mu)"
    fixed_46 = r"$\tilde u=\sigma_z \hat u^{(z)}+\mu$"
    
    if target_46 in content:
        content = content.replace(target_46, fixed_46)
        print("Fixed Line 46")

    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_file()
