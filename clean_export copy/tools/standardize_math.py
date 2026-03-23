import os
import re

PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
DOCS_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review")

def standardize_file(filename):
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    
    # 1. Replace \[ ... \] with $$ ... $$
    # We use re.DOTALL to match across lines
    # We need to be careful about escaped backslashes inside.
    # Pattern: literal \[ followed by lazy match until literal \]
    # Replace with $$ \1 $$
    
    # Check if \[ exists
    if '\\[' in content:
        content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
        
    # 2. Replace \( ... \) with $ ... $
    # Pattern: literal \( followed by lazy match until literal \)
    # We assume these are inline and usually single line, but DOTALL is safer?
    # Usually inline math doesn't span paragraphs.
    if '\\(' in content:
        content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content, flags=re.DOTALL)

    # 3. Clean up potential double $$ caused by previous replacements or mixups
    # e.g. $$$$ -> $$
    # content = content.replace('$$$$', '$$') 
    
    if content != original_content:
        print(f"Standardized math in {filename}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        # print(f"No changes in {filename}")
        pass

if __name__ == "__main__":
    # Get all .md files
    md_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
    
    for f in md_files:
        standardize_file(f)
