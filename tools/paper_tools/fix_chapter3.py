import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter3.md"

def fix_file():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. H$u_t$ -> H(u_t)
    # Be careful with H(u_t) being inside other math.
    # Pattern: H$u_t$
    content = content.replace(r'H$u_t$', r'H(u_t)')
    
    # 2. \Phi_{\omega}\big$y_{1:T} -> \Phi_{\omega}\big(y_{1:T}
    content = content.replace(r'\Phi_{\omega}\big$y_{1:T}', r'\Phi_{\omega}\big(y_{1:T}')
    # Closing \big$ -> \big)
    # The string ends with `p\big$`.
    # Replace `\big$` with `\big)` generally?
    # Wait, check context.
    # Line 37: `\hat{u}_{1:T}=\Phi_{\omega}\big$y_{1:T},\, m_{1:T},\, p\big$,`
    # Replace `\big$` with `\big(`. No, first one is `\big$`. Second is `\big$`.
    # It should be `\big(` and `\big)`.
    # Let's target the specific line content if possible.
    # `\Phi_{\omega}\big$y_{1:T},\, m_{1:T},\, p\big$` -> `\Phi_{\omega}\big(y_{1:T},\, m_{1:T},\, p\big)`
    content = content.replace(r'\Phi_{\omega}\big$y_{1:T},\, m_{1:T},\, p\big$', r'\Phi_{\omega}\big(y_{1:T},\, m_{1:T},\, p\big)')

    # 3. H$\tilde{u}$ -> H(\tilde{u})
    # Line 52: `H_{\mathrm{err}} \triangleq \big\| H$\tilde{u}$-y \big\|_2`
    content = content.replace(r'H$\tilde{u}$', r'H(\tilde{u})')

    # 4. D_s\!\left$G... -> D_s\!\left(G...
    # Line 70: `D_s\!\left$G_{\sigma_{\mathrm{blur}}}\ast u_t\right$`
    content = content.replace(r'D_s\!\left$G', r'D_s\!\left(G')
    content = content.replace(r'\ast u_t\right$', r'\ast u_t\right)')

    # 5. C_{h_c,w_c}$u_t$ -> C_{h_c,w_c}(u_t)
    # Line 89: `C_{h_c,w_c}$u_t$`
    content = content.replace(r'C_{h_c,w_c}$u_t$', r'C_{h_c,w_c}(u_t)')

    # 6. S$\Pi(u_t$) -> S(\Pi(u_t))
    # Line 103: `S$\Pi(u_t$)`
    # Note: it ends with `$)`? No, `$)` -> `))`.
    # Or maybe it was `S$\Pi(u_t)$`.
    # Read output: `y_t = S$\Pi(u_t$) + n_t,`
    # So `$` before `\Pi`, and `$` after `u_t`.
    # Replace `S$\Pi` with `S(\Pi` and `u_t$` with `u_t)`.
    # Be careful with `u_t$`.
    content = content.replace(r'S$\Pi', r'S(\Pi')
    content = content.replace(r'(u_t$)', r'(u_t))') # Replace `(u_t$)` specifically

    # 7. H$u^{(i$})$ -> H(u^{(i)})
    # Line 125: `H$u^{(i$})$`
    # Replace `H$u` -> `H(u`.
    # Replace `i$})$` -> `i)})`.
    content = content.replace(r'H$u^{(i$})$', r'H(u^{(i)})')
    
    # 8. DC$u^{(i$})$ -> DC(u^{(i)})
    # Line 126
    content = content.replace(r'DC$u^{(i$})$', r'DC(u^{(i)})')

    # 9. MSE\big$H... -> MSE\big(H...
    # Line 129: `\mathrm{MSE}\big$H(u^{(i$}),\,DC$u^{(i$})\big)`
    # This is messy.
    # Target: `\mathrm{MSE}\big(H(u^{(i)}),\,DC(u^{(i)})\big)`
    old_mse = r'\mathrm{MSE}\big$H(u^{(i$}),\,DC$u^{(i$})\big)'
    new_mse = r'\mathrm{MSE}\big(H(u^{(i)}),\,DC(u^{(i)})\big)'
    content = content.replace(old_mse, new_mse)

    # 10. Concat\big$... -> Concat\big(...
    # Line 146: `x_t=\mathrm{Concat}\big$\mathrm{baseline}(y_t$,\,m_t,\,\mathrm{coords},\,\mathrm{PE}\_{\mathrm{Fourier}}\big)`
    # Replace `\big$` with `\big(`.
    # Replace `y_t$` with `y_t)`.
    content = content.replace(r'\mathrm{Concat}\big$', r'\mathrm{Concat}\big(')
    content = content.replace(r'baseline}(y_t$', r'baseline}(y_t)')

    # 11. \Phi_{\omega}$\cdot$ -> \Phi_{\omega}(\cdot)
    # Line 209
    content = content.replace(r'\Phi_{\omega}$\cdot$', r'\Phi_{\omega}(\cdot)')

    # 12. \mathcal{F}$\hat{u}^{(z$}) -> \mathcal{F}(\hat{u}^{(z)})
    # Line 228
    content = content.replace(r'\mathcal{F}$\hat{u}^{(z$})', r'\mathcal{F}(\hat{u}^{(z)})')
    content = content.replace(r'\mathcal{F}$u^{(z$})', r'\mathcal{F}(u^{(z)})')

    # 13. L_{\mathrm{dc}} H$\tilde{u}$ -> H(\tilde{u})
    # Line 239: `\left\|H$\tilde{u}$-y\right\|`
    # Already handled by case 3? `H$\tilde{u}$`.
    # Check if case 3 used replace all. Yes.

    # 14. The specific error: $\hat{u}_{t+1}-\hat{u}_t$ inside \left\|
    # Line 250: `\left\| $\hat{u}_{t+1}-\hat{u}_t$ - $u_{t+1}-u_t$ \right\|`
    # Replace `$\hat{u}_{t+1}-\hat{u}_t$` with `(\hat{u}_{t+1}-\hat{u}_t)`
    # Replace `$u_{t+1}-u_t$` with `(u_{t+1}-u_t)`
    content = content.replace(r'$\hat{u}_{t+1}-\hat{u}_t$', r'(\hat{u}_{t+1}-\hat{u}_t)')
    content = content.replace(r'$u_{t+1}-u_t$', r'(u_{t+1}-u_t)')

    # 15. Fix residual p\big$ -> p\big)
    content = content.replace(r'p\big$,', r'p\big),')
    
    # 16. Fix \sum_{$k_x,k_y$\in -> \sum_{k_x,k_y\in
    content = content.replace(r'\sum_{$k_x,k_y$\in', r'\sum_{k_x,k_y\in')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter3.md")
    else:
        print("No changes needed for chapter3.md")

if __name__ == "__main__":
    fix_file()
