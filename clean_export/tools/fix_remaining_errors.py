
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES_TO_FIX = [
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter4.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter5.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter6.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter7.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/template.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/symbol_checklist.md",
    PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/chapter0_abstract.md"
]

def fix_content(content, filename):
    original = content
    
    # 1. Fix \sum_{$k_x,k_y$\in -> \sum_{k_x,k_y\in
    content = content.replace(r'\sum_{$k_x,k_y$\in', r'\sum_{k_x,k_y\in')
    
    # 2. Fix \big$ usage
    # Case: Concat\big$ -> Concat\big(
    content = content.replace(r'\mathrm{Concat}\big$', r'\mathrm{Concat}\big(')
    # Case: baseline$y_t$ -> baseline(y_t)
    content = content.replace(r'\mathrm{baseline}$y_t$', r'\mathrm{baseline}(y_t)')
    content = content.replace(r'\mathrm{baseline}(y_t$', r'\mathrm{baseline}(y_t)') # Just in case
    
    # Case: D_s\big$G -> D_s\big(G
    content = content.replace(r'D_s\big$G', r'D_s\big(G')
    # Case: \ast u\big$+n -> \ast u\big)+n
    content = content.replace(r'\ast u\big$+n', r'\ast u\big)+n')
    # Case: (k_{\sigma} * u_t$ \big) -> (k_{\sigma} * u_t \big)
    content = content.replace(r'* u_t$ \big)', r'* u_t \big)')
    
    # 3. Fix H$u_t$ or H$\cdot$ inside $$ or generally
    # H$u_t$ -> H(u_t)
    content = content.replace(r'H$u_t$', r'H(u_t)')
    content = content.replace(r'H$\tilde{u}$', r'H(\tilde{u})')
    content = content.replace(r'H$\cdot$', r'H(\cdot)')
    
    # 4. Fix $ inside $$...$$
    # This is tricky with simple replace, but let's target specific findings
    
    # chapter7.md: $L_{\mathrm{spec}}$ inside list item (might be $$ if global replace happened?)
    # Scanner said: "Found $ inside $$ line: - 命题2：以 $L_{\mathrm{spec}}$ ..."
    # This line likely has $$L_{\mathrm{spec}}$$ or similar.
    # If it is `- ... $L...$ ... $$k...$$`
    # Let's fix specific detected patterns.
    
    # chapter6.md: ($O$N^2$$) -> ($O(N^2)$)
    content = content.replace(r'($O$N^2$$)', r'($O(N^2)$)')
    content = content.replace(r'($O$N \log N$$)', r'($O(N \log N)$)')
    
    # chapter6.md: $t$\mathrm{df}=...$$
    # $t$\mathrm{df} -> t_{\mathrm{df}}
    content = content.replace(r'$t$\mathrm{df}', r't_{\mathrm{df}}')
    
    # chapter6.md: $$L_{spec}$$ inside text -> $L_{spec}$
    # "引入频谱损失 $$L_{spec}$$" -> "引入频谱损失 $L_{spec}$"
    # This might be tricky if I can't distinguish display math.
    # But usually $$ inside text is wrong.
    # Replace ` $$L_{spec}$$` with ` $L_{spec}$`
    content = content.replace(r' $$L_{spec}$$', r' $L_{spec}$')
    content = content.replace(r'$$L_{spec}$$', r'$L_{spec}$')
    
    # chapter6.md: DC Error $$H_{\mathrm{err}}$$ -> DC Error $H_{\mathrm{err}}$
    content = content.replace(r'$$H_{\mathrm{err}}$$ ', r'$H_{\mathrm{err}}$ ')
    content = content.replace(r' $$H_{\mathrm{err}}$$ ', r' $H_{\mathrm{err}}$ ')
    content = content.replace(r'$$H_{\mathrm{err}}$$)', r'$H_{\mathrm{err}}$)')

    # chapter6.md: $H$\tilde{u}$-y$ -> \|H(\tilde{u})-y\| ?
    # Scanner: `H$\tilde{u}$-y`
    content = content.replace(r'H$\tilde{u}$-y', r'H(\tilde{u})-y')

    # chapter6.md: $$h_c,w_c$$ -> $h_c,w_c$
    content = content.replace(r'$$h_c,w_c$$', r'$h_c,w_c$')

    # chapter6.md: $$\sigma_n$$ -> $\sigma_n$
    content = content.replace(r'$$\sigma_n$$', r'$\sigma_n$')
    
    # chapter6.md: Fixed $T_{out}$$ -> Fixed $T_{out}$
    content = content.replace(r'Fixed $T_{out}$$', r'Fixed $T_{out}$')
    
    # chapter4.md: H$u_t$$ -> H(u_t)
    content = content.replace(r'H$u_t$$', r'H(u_t)$$') # Wait, if it ends with $$?
    content = content.replace(r'H$u_t$', r'H(u_t)')
    
    # chapter4.md: \mathcal{N}$u_t$$ -> \mathcal{N}(u_t)
    content = content.replace(r'\mathcal{N}$u_t$$', r'\mathcal{N}(u_t)$$')
    content = content.replace(r'\mathcal{N}$u_t$', r'\mathcal{N}(u_t)')
    
    # template.md: $u$t,\mathbf{x}$$ -> u(t,\mathbf{x})
    content = content.replace(r'$u$t,\mathbf{x}$$', r'u(t,\mathbf{x})$$')
    content = content.replace(r'$u$t,\mathbf{x}$', r'u(t,\mathbf{x})')
    
    # template.md: \mathcal{F}$u;\boldsymbol{\theta}$ -> \mathcal{F}(u;\boldsymbol{\theta})
    content = content.replace(r'\mathcal{F}$u;\boldsymbol{\theta}$', r'\mathcal{F}(u;\boldsymbol{\theta})')
    
    # template.md: D_s\big$ -> D_s\big(
    content = content.replace(r'D_s\big$', r'D_s\big(')
    
    # template.md: C_{h_c,w_c}$u_t$ -> C_{h_c,w_c}(u_t)
    content = content.replace(r'C_{h_c,w_c}$u_t$', r'C_{h_c,w_c}(u_t)')
    
    # template.md: \mathcal{F}_{2\text{D}}$\hat{u}$ -> \mathcal{F}_{2\text{D}}(\hat{u})
    content = content.replace(r'\mathcal{F}_{2\text{D}}$\hat{u}$', r'\mathcal{F}_{2\text{D}}(\hat{u})')

    # symbol_checklist.md: H$\cdot$$ -> H(\cdot)
    content = content.replace(r'H$\cdot$$', r'H(\cdot)$$')
    
    # symbol_checklist.md: D_s$\cdot$$ -> D_s(\cdot)
    content = content.replace(r'D_s$\cdot$$', r'D_s(\cdot)$$')
    content = content.replace(r'D_s$\cdot$', r'D_s(\cdot)')
    
    # symbol_checklist.md: C_{h_c,w_c}^{$\mathrm{mode}$} -> C_{h_c,w_c}^{\mathrm{mode}}
    content = content.replace(r'^{$\mathrm{mode}$}', r'^{\mathrm{mode}}')
    
    # symbol_checklist.md: \mathcal{F}_{2D}$\cdot$$ -> \mathcal{F}_{2D}(\cdot)
    content = content.replace(r'\mathcal{F}_{2D}$\cdot$$', r'\mathcal{F}_{2D}(\cdot)$$')
    content = content.replace(r'\mathcal{F}_{2D}$\cdot$', r'\mathcal{F}_{2D}(\cdot)')

    # abstract: relative $L_2$ error
    # Scanner: `relative $L_2$ error` inside a line that triggered `Found $ inside $$ line`.
    # This might be because the line had `$$H_{\mathrm{err}}$$` later?
    # Scanner output: `Crucially, the evaluation consistency error $$H_{\mathrm{err}}$$ decreases...`
    # Yes. Fix $$H_{\mathrm{err}}$$ -> $H_{\mathrm{err}}$
    # Already added rule for this.

    return content

def run():
    for file_path in FILES_TO_FIX:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        fixed_content = fix_content(content, os.path.basename(file_path))
        
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed {os.path.basename(file_path)}")
        else:
            print(f"No changes for {os.path.basename(file_path)}")

if __name__ == "__main__":
    run()
