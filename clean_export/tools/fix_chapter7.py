
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter7.md"

def fix_chapter7():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Fix RelL2 with $ inside
    # \mathrm{RelL2}^{$\text{baseline}
    # User reports: \mathrm{RelL2}^{$\text{baseline}...
    # Expected: \mathrm{RelL2}^{\text{baseline}}
    content = content.replace(r'\mathrm{RelL2}^{$\text{baseline}', r'\mathrm{RelL2}^{\text{baseline}')
    content = content.replace(r'\mathrm{RelL2}^{$\text{ours}', r'\mathrm{RelL2}^{\text{ours}')
    # Check if closing brace is handled. If the user had `\mathrm{RelL2}^{$\text{baseline}$}_j`, then replace `^{$\text{baseline}$}` with `^{\text{baseline}}`
    content = content.replace(r'^{$\text{baseline}$}_j', r'^{\text{baseline}}_j')
    content = content.replace(r'^{$\text{ours}$}_j', r'^{\text{ours}}_j')
    
    # 2. Fix \mathcal F$\tilde u$_k
    # User reports: \left|\mathcal F$\tilde u$_k-\ma...
    # Expected: \left|\mathcal F(\tilde u)_k-\ma...
    content = content.replace(r'\mathcal F$\tilde u$_k', r'\mathcal F(\tilde u)_k')
    
    # 3. Fix Pearson correlation
    # User reports: \text{Pearson}}$\mathrm{RelL2}_
    # Expected: \text{Pearson}}(\mathrm{RelL2}_ or \text{Pearson}(\mathrm{RelL2}_
    # Context: r=\mathrm{corr}_{\text{Pearson}}$\mathrm{RelL2}_j,\,H_{\mathrm{err},j}$
    # Change to: r=\mathrm{corr}_{\text{Pearson}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j})
    content = content.replace(r'\mathrm{corr}_{\text{Pearson}}$\mathrm{RelL2}_j,\,H_{\mathrm{err},j}$', r'\mathrm{corr}_{\text{Pearson}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j})')
    content = content.replace(r'\mathrm{corr}_{\text{Spearman}}$\mathrm{RelL2}_j,\,H_{\mathrm{err},j}$', r'\mathrm{corr}_{\text{Spearman}}(\mathrm{RelL2}_j,\,H_{\mathrm{err},j})')

    # 4. Fix Hu^{(iParseError...
    # Context: H$u^{(i$})$
    # Expected: H(u^{(i)})
    content = content.replace(r'H$u^{(i$})$', r'H(u^{(i)})')
    content = content.replace(r'H$u^{(i)}$)', r'H(u^{(i)})') # Just in case

    # 5. Fix MSE with \left$
    # Context: \mathrm{MSE}\!\left$H(u^{(i$}),\,y^{(i)}\right)
    # Expected: \mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right)
    content = content.replace(r'\mathrm{MSE}\!\left$H(u^{(i$}),\,y^{(i)}\right)', r'\mathrm{MSE}\!\left(H(u^{(i)}),\,y^{(i)}\right)')

    # 6. Fix H_{\mathrm{err}} definition
    # Context: \|H$\tilde u$-y\|_2
    # Expected: \|H(\tilde u)-y\|_2
    content = content.replace(r'\|H$\tilde u$-y\|_2', r'\|H(\tilde u)-y\|_2')
    
    # 7. Additional checks for $ inside math
    content = content.replace(r'H$\tilde u$', r'H(\tilde u)')
    content = content.replace(r'\mathcal F$\tilde u$', r'\mathcal F(\tilde u)')
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed chapter7.md")
    else:
        print("No changes needed for chapter7.md")

if __name__ == "__main__":
    fix_chapter7()
