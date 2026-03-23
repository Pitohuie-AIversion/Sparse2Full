import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter2.md"

def fix_file():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. $u_\theta$\mathbf{x},t$$ -> $u_\theta(\mathbf{x},t)$
    # Note the spacing and possible variation.
    # From read output: `$u_\theta$\mathbf{x},t$$` (Line 73)
    content = content.replace('$u_\\theta$\\mathbf{x},t$$', '$u_\\theta(\\mathbf{x},t)$')
    
    # 2. \mathcal{N}[u](\mathbf{x},t)=0,\quad $(\mathbf{x},t)\in \Omega\times[0,T],
    # The previous fix put `$(\mathbf{x},t)\in`.
    # It should be `(\mathbf{x},t)\in` (no $) because it's inside $$ block.
    # The original was `$\mathbf{x},t$\in`.
    # My previous script replaced `$\mathbf{x},t$\in` with `$(\mathbf{x},t)\in`.
    # So now it is `$(\mathbf{x},t)\in`.
    # I need to replace `$(\mathbf{x},t)\in` with `(\mathbf{x},t)\in`.
    # Be careful not to replace valid inline math like `$(\mathbf{x},t)\in` in text.
    # But `\quad` usually appears in display math.
    # So targeting `\quad $(\mathbf{x},t)\in` is safe.
    
    content = content.replace(r'\quad $(\mathbf{x},t)\in', r'\quad (\mathbf{x},t)\in')
    
    # 3. \mathcal{L}$\theta$= -> \mathcal{L}(\theta)=
    # From read output: `\mathcal{L}$\theta$=` (Line 82)
    content = content.replace('\\mathcal{L}$\\theta$=', '\\mathcal{L}(\\theta)=')
    
    # 4. recommendation $`INTER_AREA` for shrinking$
    # From read output: `recommendation $`INTER_AREA` for shrinking$` (Line 156)
    content = content.replace('recommendation $`INTER_AREA` for shrinking$', 'recommendation (`INTER_AREA` for shrinking)')
    
    # 5. Nested big parens: $y = H(u) = D\big$G_{\sigma_{\mathrm{blur}}} * u\big$ + n$
    # From read output: `$y = H(u) = D\big$G_{\sigma_{\mathrm{blur}}} * u\big$ + n` (Line 159)
    # Replace `\big$` with `\big(` or `\big)`.
    # `D\big$G` -> `D\big(G`
    # `u\big$` -> `u\big)`
    content = content.replace(r'D\big$G', r'D\big(G')
    content = content.replace(r'u\big$', r'u\big)')

    # 6. $D$\cdot$$ -> $D(\cdot)$
    # Line 162: `$D$\cdot$$`
    content = content.replace('$D$\\cdot$$', '$D(\\cdot)$')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter2.md")
    else:
        print("No changes needed for chapter2.md")

if __name__ == "__main__":
    fix_file()
