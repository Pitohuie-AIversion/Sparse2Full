
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter6.md"

def fix_chapter6_v2():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. \mathcal{F}_{2\mathrm{D}}$\tilde u$ -> \mathcal{F}_{2\mathrm{D}}(\tilde u)
    content = content.replace(r'\mathcal{F}_{2\mathrm{D}}$\tilde u$', r'\mathcal{F}_{2\mathrm{D}}(\tilde u)')
    
    # 2. \mathrm{fRMSE}$\mathcal{K}$ -> \mathrm{fRMSE}(\mathcal{K})
    content = content.replace(r'\mathrm{fRMSE}$\mathcal{K}$', r'\mathrm{fRMSE}(\mathcal{K})')
    
    # 3. $\tilde u_{ij}-u_{ij}$^2 -> (\tilde u_{ij}-u_{ij})^2
    # Be careful with context: \sum...$\tilde u_{ij}-u_{ij}$^2
    content = content.replace(r'$\tilde u_{ij}-u_{ij}$^2', r'(\tilde u_{ij}-u_{ij})^2')
    
    # 4. Extra commas in magnitude difference?
    # \left|,|\tilde U_k|-|U_k|,\right|^2 -> \left| |\tilde U_k|-|U_k| \right|^2
    content = content.replace(r'\left|,|\tilde U_k|-|U_k|,\right|^2', r'\left| |\tilde U_k|-|U_k| \right|^2')
    
    # 5. Check for any other $\tilde u$ inside math blocks
    # content = content.replace(r'$\tilde u$', r'(\tilde u)') # Too risky if it's standalone inline math
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter6.md v2")
    else:
        print("No changes needed for chapter6.md v2")

if __name__ == "__main__":
    fix_chapter6_v2()
