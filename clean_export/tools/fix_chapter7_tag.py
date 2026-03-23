
import re
import os

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter7.md"

def fix_chapter7_tag():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Context:
    # 43: ... $H(u^{(i)})，并记录
    # 44: $$
    # 45: e^{(i)}=...
    # 47: ... \tag{7-5}
    # 48: $$
    
    # The error "ParseError: KaTeX parse error: \tag works only in display equations$" usually happens if:
    # 1. The environment is NOT recognized as display math by Pandoc/KaTeX.
    # 2. Or there is some weird character interfering.
    
    # However, in standard Markdown, $$...$$ IS display math.
    # But if the previous line "并记录" (Chinese text) is immediately followed by $$, sometimes parsers get confused if there is no newline?
    # My read output shows:
    # 43: ... $H(u^{(i)})，并记录
    # 44: $$
    # This looks correct (newline exists).
    
    # Wait, the user error snippet shows:
    # "...并记录ParseError: KaTeX parse error: \tag works only in display equations$"
    # This implies the parser thinks `\tag` is being used in an inline context or the display block is broken.
    
    # Let's try to remove `\tag{7-5}` if it causes issues, or ensure it's strictly inside `$$`.
    # It IS inside `$$`.
    
    # Maybe the issue is line 43 end: `H(u^{(i)})，并记录`
    # The `)` is closing `H(...)`. The `，` is Chinese comma.
    
    # Another possibility: Pandoc conversion to docx might use a different math engine that struggles with `\tag` inside `$$` if it's not `align` environment?
    # But `\tag` is standard AMS math.
    
    # Let's try to use `\quad (7-5)` instead of `\tag{7-5}` to manually number it, if `\tag` is problematic in this specific pipeline.
    # Or wrap it in `\begin{equation} ... \end{equation}`?
    # But `$$` is usually preferred for Markdown.
    
    # Let's replace `\tag{7-5}` with `\qquad (7-5)` to simulate the tag without triggering the command.
    # This is a safe fallback for docx conversion.
    
    content = content.replace(r'\tag{7-5}', r'\qquad (7-5)')
    content = content.replace(r'\tag{7-1}', r'\qquad (7-1)')
    content = content.replace(r'\tag{7-2}', r'\qquad (7-2)')
    content = content.replace(r'\tag{7-3}', r'\qquad (7-3)')
    content = content.replace(r'\tag{7-4}', r'\qquad (7-4)')
    content = content.replace(r'\tag{7-6}', r'\qquad (7-6)')
    content = content.replace(r'\tag{7-7}', r'\qquad (7-7)')
    content = content.replace(r'\tag{7-8}', r'\qquad (7-8)')
    content = content.replace(r'\tag{7-9}', r'\qquad (7-9)')
    content = content.replace(r'\tag{7-10}', r'\qquad (7-10)')
    
    # Also check if there are any other `\tag` usages.
    
    if content != original_content:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Replaced \\tag with \\qquad manual numbering in chapter7.md")
    else:
        print("No \\tag found to replace in chapter7.md")

if __name__ == "__main__":
    fix_chapter7_tag()
