
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter4.md"

def fix_chapter4():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. H$\tilde u$-y -> H(\tilde u)-y
    content = content.replace(r'H$\tilde u$-y', r'H(\tilde u)-y')
    
    # 2. H$\tilde u_t$-y_t -> H(\tilde u_t)-y_t
    content = content.replace(r'H$\tilde u_t$-y_t', r'H(\tilde u_t)-y_t')
    
    # 3. L^2$\Omega$^C -> L^2(\Omega)^C
    content = content.replace(r'L^2$\Omega$^C', r'L^2(\Omega)^C')
    
    # 4. H$\tilde u$-H(u) -> H(\tilde u)-H(u)
    content = content.replace(r'H$\tilde u$-H(u)', r'H(\tilde u)-H(u)')
    
    # 5. H$\tilde u$-DC -> H(\tilde u)-DC
    content = content.replace(r'H$\tilde u$-DC', r'H(\tilde u)-DC')
    
    # 6. DC$\tilde u$-y -> DC(\tilde u)-y
    content = content.replace(r'DC$\tilde u$-y', r'DC(\tilde u)-y')
    
    # 7. DC$\tilde u$ -> DC(\tilde u) (Catch residual)
    content = content.replace(r'DC$\tilde u$', r'DC(\tilde u)')
    content = content.replace(r'DC$\tilde u$\|', r'DC(\tilde u)\|') # Safety check
    
    # 8. D_s$G... * u$ -> D_s(G... * u)
    content = content.replace(r'D_s$G', r'D_s(G')
    # Note: * u$ might be risky if $ is end of math.
    # Context: D_s(G_{\sigma_{\mathrm{blur}}} * u$
    # Let's match the specific context
    content = content.replace(r'* u$,', r'* u),')
    content = content.replace(r'* u$', r'* u)') # Fallback if comma not present
    
    # 9. H$\tilde u$\approx -> H(\tilde u)\approx
    content = content.replace(r'H$\tilde u$\approx', r'H(\tilde u)\approx')
    
    # 10. \mathcal{F}$\hat u$ -> \mathcal{F}(\hat u)
    content = content.replace(r'\mathcal{F}$\hat u$', r'\mathcal{F}(\hat u)')
    
    # 11. H$u_t$$ -> H(u_t)$
    content = content.replace(r'H$u_t$$', r'H(u_t)$')
    
    # 12. \mathcal{N}$u_t$$ -> \mathcal{N}(u_t)$
    content = content.replace(r'\mathcal{N}$u_t$$', r'\mathcal{N}(u_t)$')
    
    # 13. Reference title fix
    content = content.replace(r'$recommendation', r'recommendation')
    content = content.replace(r'shrinking$', r'shrinking')
    
    # 14. Fix any remaining H$\tilde u$
    content = content.replace(r'H$\tilde u$', r'H(\tilde u)')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter4.md")
    else:
        print("No changes needed for chapter4.md")

if __name__ == "__main__":
    fix_chapter4()
