
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter5.md"

def fix_chapter5():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. t\in{1,\dots,T} -> t\in\{1,\dots,T\}
    content = content.replace(r't\in{1,\dots,T}', r't\in\{1,\dots,T\}')
    
    # 2. $$y_t, u_t$$ -> $y_t, u_t$
    content = content.replace(r'$$y_t, u_t$$', r'$y_t, u_t$')
    
    # 3. $$\mu,\sigma_z$$ -> $\mu,\sigma_z$
    content = content.replace(r'$$\mu,\sigma_z$$', r'$\mu,\sigma_z$')
    
    # 4. __init__$in_ch -> __init__(in_ch
    content = content.replace(r'__init__$in_ch', r'__init__(in_ch')
    content = content.replace(r'kwargs$', r'kwargs)')
    
    # 5. forward$x -> forward(x
    content = content.replace(r'forward$x', r'forward(x')
    
    # 6. ([0,1]) -> [0,1]
    # No, keep it if it is text. But if it was `([0,1])` inside math? 
    # Context: "归一化到 ([0,1])" -> "归一化到 $[0,1]$" or just "[0,1]"
    # It seems to be text. "归一化到 ([0,1])" is fine? 
    # Let's check if it was `([0,1])` which looks weird. Maybe `[0,1]` is better.
    content = content.replace(r'([0,1])', r'$[0,1]$')

    # 7. Concat\big(...,,m_t,,... -> Concat\big(...,\,m_t,\,
    content = content.replace(r',,', r',\,')
    content = content.replace(r'\big$.', r'\big).')
    
    # 8. \|H$\tilde u$-y\| -> \|H(\tilde u)-y\|
    content = content.replace(r'\|H$\tilde u$-y\|', r'\|H(\tilde u)-y\|')
    
    # 9. D_s\left$G... -> D_s\left(G...
    content = content.replace(r'D_s\left$G', r'D_s\left(G')
    content = content.replace(r'\ast u_t\right$', r'\ast u_t\right)')
    
    # 10. u^{(i$}) -> u^{(i)}
    content = content.replace(r'u^{(i$})', r'u^{(i)}')
    content = content.replace(r'y^{(i$})', r'y^{(i)}')
    
    # 11. \hat y^{$i$}=H$u^{(i$}$) -> \hat y^{(i)}=H(u^{(i)})
    content = content.replace(r'\hat y^{$i$}', r'\hat y^{(i)}')
    content = content.replace(r'H$u^{(i$}$)', r'H(u^{(i)})')
    content = content.replace(r'H$u^{(i)}$)', r'H(u^{(i)})') # Just in case
    
    # 12. MSE\left$\hat y^{$i$}, y^{(i$} -> MSE\left(\hat y^{(i)}, y^{(i)}
    content = content.replace(r'\mathrm{MSE}\left$\hat y^{$i$}', r'\mathrm{MSE}\left(\hat y^{(i)}')
    content = content.replace(r'y^{(i$}\right)', r'y^{(i)}\right)')
    
    # 13. $$s,\sigma_{\mathrm{blur}},k$$ -> $s,\sigma_{\mathrm{blur}},k$
    content = content.replace(r'$$s,\sigma_{\mathrm{blur}},k$$', r'$s,\sigma_{\mathrm{blur}},k$')
    
    # 14. \mathcal F_{2\mathrm{D}}$\hat u^{(z$}) -> \mathcal F_{2\mathrm{D}}(\hat u^{(z)})
    content = content.replace(r'\mathcal F_{2\mathrm{D}}$\hat u^{(z$})', r'\mathcal F_{2\mathrm{D}}(\hat u^{(z)})')
    content = content.replace(r'\mathcal F_{2\mathrm{D}}$u^{(z$})', r'\mathcal F_{2\mathrm{D}}(u^{(z)})')
    
    # 15. \left|H$\tilde u$-y\right|_2^2 -> \left\|H(\tilde u)-y\right\|_2^2
    content = content.replace(r'\left|H$\tilde u$-y\right|_2^2', r'\left\|H(\tilde u)-y\right\|_2^2')
    
    # 16. Pseudo code function calls with $
    content = content.replace(r'CosineScheduleWithWarmup$opt', r'CosineScheduleWithWarmup(opt')
    content = content.replace(r'warmup$', r'warmup)')
    content = content.replace(r'check_equivalence$H', r'check_equivalence(H')
    content = content.replace(r'eps=1e-8$', r'eps=1e-8)')
    content = content.replace(r'mse$u_hat_z, u_z$', r'mse(u_hat_z, u_z)')
    content = content.replace(r'lowfreq_fft_mse$u_hat_z', r'lowfreq_fft_mse(u_hat_z')
    content = content.replace(r'kmax=cfg.loss.kmax$', r'kmax=cfg.loss.kmax)')
    content = content.replace(r'mse$DC$u_hat$, y$', r'mse(DC(u_hat), y)')
    content = content.replace(r'zero_grad$set_to_none=True$', r'zero_grad(set_to_none=True)')
    
    # 17. References
    content = content.replace(r'guidance $INTER_AREA', r'guidance (INTER_AREA')
    content = content.replace(r'shrinking$', r'shrinking)')

    # 18. Fix forward signature
    # forward$x[B,C_in,H,W]$ -> forward(x[B,C_in,H,W])
    content = content.replace(r'forward$x[B,C_in,H,W]$', r'forward(x[B,C_in,H,W])')
    
    # 19. init signature
    # __init__$in_ch, out_ch, img_size, *_kwargs$
    # Already handled partly?
    content = content.replace(r'*_kwargs$', r'*_kwargs)')

    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter5.md")
    else:
        print("No changes needed for chapter5.md")

if __name__ == "__main__":
    fix_chapter5()
