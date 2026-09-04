import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter0_notation.md"

def fix_file():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    
    # 1. $u$\mathbf{x}, t$$ -> $u(\mathbf{x}, t)$
    # The read output: `8→| $u$\mathbf{x}, t$$ | ...`
    # Let's use simple string replace for robustness.
    # The string is: `$u$\mathbf{x}, t$$`
    target_1 = r'$u$\mathbf{x}, t$$'
    fix_1 = r'$u(\mathbf{x}, t)$'
    if target_1 in content:
        content = content.replace(target_1, fix_1)
        print("Fixed $u$... pattern")
    else:
        print("Target 1 not found (exact match)")

    # 2. $$d=2$ 或 $3$$ -> $d=2$ 或 $3$
    target_2 = r'$$d=2$ 或 $3$$'
    fix_2 = r'$d=2$ 或 $3$'
    if target_2 in content:
        content = content.replace(target_2, fix_2)
        print("Fixed d=2 or 3 pattern")
    
    # 3. $n \sim \mathcal{N}$0, \sigma_n^2$$
    target_3 = r'$n \sim \mathcal{N}$0, \sigma_n^2$$'
    fix_3 = r'$n \sim \mathcal{N}(0, \sigma_n^2)$'
    if target_3 in content:
        content = content.replace(target_3, fix_3)
        print("Fixed n ~ N pattern")
    
    # 4. Operator fixes
    # Ops: H, DC, D_s, C_{h_c, w_c}, f_\theta, \mathcal{F}
    ops = ['H', 'DC', 'D_s', 'C_{h_c, w_c}', r'f_\theta', r'\mathcal{F}']
    for op in ops:
        old_str = f"${op}$\\cdot$$"
        new_str = f"${op}(\\cdot)$"
        if old_str in content:
            content = content.replace(old_str, new_str)
            print(f"Fixed {op} operator")
        
    # 5. Fix $$h_c, w_c$$
    if '$$h_c, w_c$$' in content:
        content = content.replace('$$h_c, w_c$$', '$h_c, w_c$')
        print("Fixed h_c, w_c")
    
    # 6. Fix `INTER_AREA` interpolation text
    target_6 = '$通常使用 `INTER_AREA` 插值$'
    fix_6 = '(通常使用 `INTER_AREA` 插值)'
    if target_6 in content:
        content = content.replace(target_6, fix_6)
        print("Fixed INTER_AREA text")

    # 7. Fix $\mathcal{K}_{\mathrm{low}}$ set definition
    # $\{$k_x, k_y$ : k_x \le K, k_y \le K\}$
    target_7 = r'$\{$k_x, k_y$'
    fix_7 = r'$\{ (k_x, k_y)'
    if target_7 in content:
        content = content.replace(target_7, fix_7)
        print("Fixed set definition")

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated chapter0_notation.md")
    else:
        print("No changes needed for chapter0_notation.md")

if __name__ == "__main__":
    fix_file()
