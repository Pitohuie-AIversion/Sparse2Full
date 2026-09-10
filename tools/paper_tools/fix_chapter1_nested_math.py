import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review")

def fix_file(filename):
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Fix nested math H$\hat{u}$ -> H(\hat{u})
    if 'H$\\hat{u}$' in content:
        content = content.replace('H$\\hat{u}$', 'H(\\hat{u})')
        print("Fixed nested math H$\\hat{u}$ in " + filename)
        
    # 2. Fix typo $u$\mathbf{x},t$$ -> $u(\mathbf{x},t)$
    if '$u$\\mathbf{x},t$$' in content:
        content = content.replace('$u$\\mathbf{x},t$$', '$u(\\mathbf{x},t)$')
        print("Fixed typo $u$\\mathbf{x},t$$ in " + filename)
        
    # 3. Fix f_\theta$y -> f_\theta(y
    if 'f_\\theta$y' in content:
        content = content.replace('f_\\theta$y', 'f_\\theta(y')
        print("Fixed f_\\theta$y -> f_\\theta(y in " + filename)

    # 4. Regex fix for \hat{u}=f_\theta$y...$. line
    # Match: \hat{u}=f_\theta$y...$.
    # Replace: \hat{u}=f_\theta(y...). (remove the trailing $ before .)
    # The previous fix might have changed it to `f_\theta(y...$.`
    # So we need to handle both `f_\theta$y` and `f_\theta(y` cases if already partially fixed.
    
    # Current state of file (after simple replace): `\hat{u}=f_\theta(y, m, \mathbf{x};\theta$.`
    # We want: `\hat{u}=f_\theta(y, m, \mathbf{x};\theta).`
    
    # Regex to find `f_\theta(y...$.`
    # Pattern: f_\\theta\(y(.*?)\$\.
    # Replace: f_\\theta(y\1).
    
    content = re.sub(r'f_\\theta\(y(.*?)\$\.', r'f_\\theta(y\1).', content)
    
    # Also handle if it was `f_\theta$y` originally and we want to do it all in one go or cleanup
    # Match: f_\\theta\$y(.*?)\$\.
    # Replace: f_\\theta(y\1).
    content = re.sub(r'f_\\theta\$y(.*?)\$\.', r'f_\\theta(y\1).', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated " + filename)

if __name__ == "__main__":
    md_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
    for f in md_files:
        fix_file(f)
