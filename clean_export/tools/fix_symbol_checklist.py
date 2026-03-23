
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/symbol_checklist.md"

def fix_symbol_checklist():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. \Phi$\cdot$ -> \Phi(\cdot)
    content = content.replace(r'\Phi$\cdot$', r'\Phi(\cdot)')
    
    # 2. D_s\!\left$G... -> D_s\!\left(G...
    content = content.replace(r'D_s\!\left$G', r'D_s\!\left(G')
    content = content.replace(r'* u\right$', r'* u\right)')
    
    # 3. D_s(\cdot)$$ -> D_s(\cdot)
    # Line 65: D_s(\cdot)$$
    # Wait, if it ends with $$, maybe the $$ was from display math?
    # Context: $D_s(\cdot)$$ 表示...
    content = content.replace(r'D_s(\cdot)$$', r'D_s(\cdot)$')
    
    # 4. \mathrm{Rel\text{-}L2}$\tilde{u},u$ -> \mathrm{Rel\text{-}L2}(\tilde{u},u)
    content = content.replace(r'\mathrm{Rel\text{-}L2}$\tilde{u},u$', r'\mathrm{Rel\text{-}L2}(\tilde{u},u)')
    
    # 5. \mathrm{Rel\text{-}L2}$\hat{u}^{(z$},u^{(z)}$ -> \mathrm{Rel\text{-}L2}(\hat{u}^{(z)},u^{(z)})
    content = content.replace(r'\mathrm{Rel\text{-}L2}$\hat{u}^{(z$},u^{(z)}$', r'\mathrm{Rel\text{-}L2}(\hat{u}^{(z)},u^{(z)})')
    
    # 6. H_{\mathrm{err}}^{$\mathrm{rel}$} -> H_{\mathrm{err}}^{\mathrm{rel}}
    content = content.replace(r'H_{\mathrm{err}}^{$\mathrm{rel}$}', r'H_{\mathrm{err}}^{\mathrm{rel}}')
    
    # 7. \mathrm{fRMSE}$\mathcal{K}$ -> \mathrm{fRMSE}(\mathcal{K})
    content = content.replace(r'\mathrm{fRMSE}$\mathcal{K}$', r'\mathrm{fRMSE}(\mathcal{K})')
    
    # 8. \mathcal{F}_{2D}(\cdot)$$ -> \mathcal{F}_{2D}(\cdot)$
    content = content.replace(r'\mathcal{F}_{2D}(\cdot)$$', r'\mathcal{F}_{2D}(\cdot)$')
    
    # 9. \mathcal{F}_{2D}$\hat{u}^{(z$}) -> \mathcal{F}_{2D}(\hat{u}^{(z)})
    content = content.replace(r'\mathcal{F}_{2D}$\hat{u}^{(z$})', r'\mathcal{F}_{2D}(\hat{u}^{(z)})')
    content = content.replace(r'\mathcal{F}_{2D}$u^{(z$})', r'\mathcal{F}_{2D}(u^{(z)})')
    
    # 10. \{$k_x,k_y$\,:\, -> \{k_x,k_y\,:\,
    content = content.replace(r'\{$k_x,k_y$\,:\,', r'\{k_x,k_y\,:\,')
    
    # 11. \mathrm{MSE}\!\left$H(u^{(i$}),\,y^{(i)}\right) -> \mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right)
    content = content.replace(r'\mathrm{MSE}\!\left$H(u^{(i$}),\,y^{(i)}\right)', r'\mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right)')
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed symbol_checklist.md")
    else:
        print("No changes needed for symbol_checklist.md")

if __name__ == "__main__":
    fix_symbol_checklist()
