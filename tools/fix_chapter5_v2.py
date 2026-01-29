
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter5.md"

def fix_chapter5_v2():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix missing closing $ for u^{(i)}
    # Pattern: $u^{(i)}： -> $u^{(i)}$：
    content = content.replace(r'$u^{(i)}：', r'$u^{(i)}$：')
    
    # 2. Fix missing closing $ for y^{(i)}
    # Pattern: $y^{(i)}； -> $y^{(i)}$；
    content = content.replace(r'$y^{(i)}；', r'$y^{(i)}$；')
    
    # 3. Fix missing closing $ for equation
    # Pattern: $\hat y^{(i)}=H(u^{(i)})； -> $\hat y^{(i)}=H(u^{(i)})$；
    content = content.replace(r'$\hat y^{(i)}=H(u^{(i)})；', r'$\hat y^{(i)}=H(u^{(i)})$；')
    
    # 4. Fix \left$ inside MSE
    # Pattern: \mathrm{MSE}\left$\hat y^{(i)} -> \mathrm{MSE}\left(\hat y^{(i)}
    content = content.replace(r'\mathrm{MSE}\left$\hat y^{(i)}', r'\mathrm{MSE}\left(\hat y^{(i)}')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter5.md missing closing $ and nested $")
    else:
        print("No changes needed for chapter5.md v2")

if __name__ == "__main__":
    fix_chapter5_v2()
