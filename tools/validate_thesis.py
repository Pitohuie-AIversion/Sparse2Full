import os
import re

PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
DOCS_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review")

MD_FILES = [
    "chapter0_abstract.md",
    "chapter0_notation.md",
    "chapter1.md",
    "chapter2.md",
    "chapter3.md",
    "chapter4.md",
    "chapter5.md",
    "chapter6.md",
    "chapter7.md",
    "chapter8.md",
    "chapter9.md",
    "symbol_checklist.md",
    "template.md",
    "writing_checklist.md"
]

def check_file(filename):
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"❌ File missing: {filename}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []
    warnings = []

    # 1. Check for broken LaTeX braces
    # Pattern: unclosed curly braces in math context is hard to check perfectly with regex,
    # but we can check specific known bad patterns.
    
    # Check for the specific u^{(z error (just in case)
    if re.search(r'\^\{\(z(?!\)\})', content):
         errors.append("Found incomplete superscript pattern: u^{(z without closing )}")

    # Check for general unbalanced braces in lines that look like math
    # Simple heuristic: count { and } in each line. If they don't match, warn.
    # Note: this produces false positives for multi-line math, but useful for inline.
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Ignore lines that are clearly not math or are code blocks
        if line.strip().startswith('```') or line.strip().startswith('    '):
            continue
            
        # Extract potential math parts: $...$ or $$...$$
        # Regex for inline math: \$[^$]+\$
        math_matches = re.findall(r'\$([^$]+)\$', line)
        for m in math_matches:
            if m.count('{') != m.count('}'):
                 warnings.append(f"Line {i+1}: Potential unbalanced braces in inline math: ${m}$")
            if m.count('(') != m.count(')'):
                 warnings.append(f"Line {i+1}: Potential unbalanced parentheses in inline math: ${m}$")
                 
    # 2. Check for non-standard delimiters
    if re.search(r'(?<!\\)\[' + r'.*?' + r'(?<!\\)\]', content, re.DOTALL):
        # Check if [ ... ] is being used as display math (heuristic: contains latex symbols)
        # This is tricky because Markdown links use [text].
        # Look for [ at start of line followed by mathy stuff
        for i, line in enumerate(lines):
             stripped = line.strip()
             if stripped.startswith('[') and ('\\' in line or '_' in line or '=' in line) and not ']' in line:
                  warnings.append(f"Line {i+1}: Possible non-standard display math start '['")

    # 3. Check for image links
    # ![Alt](path)
    img_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
    for img_path in img_matches:
        # Resolve path relative to the markdown file
        # Most paths are likely absolute or relative to project root?
        # Let's try to resolve it.
        if img_path.startswith('http'):
            continue
            
        # Check if path exists
        # 1. Relative to file
        full_path_rel = os.path.join(DOCS_DIR, img_path)
        # 2. Relative to project root (if path starts with / or thesis_paper)
        full_path_abs = os.path.join(PROJECT_ROOT, img_path.lstrip('/'))
        
        if not os.path.exists(full_path_rel) and not os.path.exists(full_path_abs):
             warnings.append(f"Missing image: {img_path}")

    # 4. Check for LaTeX * typos
    if '^*' in content or '_*' in content or '*{' in content:
        # Check specific bad patterns
        if re.search(r'\*\{', content):
             errors.append("Found '*{' which is likely a typo for '_{' or '^{'")
        if re.search(r'[a-zA-Z]\*', content) and '$' in content:
             # letter followed by * inside math might be okay (f*), but often is typo for f_
             # We only warn if it looks like it's trying to be a subscript
             pass

    if errors or warnings:
        print(f"\n📄 Checking {filename}...")
        for e in errors:
            print(f"  🔴 ERROR: {e}")
        for w in warnings:
            # Limit warnings to avoid noise
            if "unbalanced" in w and len(warnings) > 10:
                continue
            print(f"  ⚠️ WARNING: {w}")
    else:
        # print(f"✅ {filename} looks good.")
        pass

if __name__ == "__main__":
    print("Starting Thesis Validation Scan...\n")
    for f in MD_FILES:
        check_file(f)
    print("\nScan Complete.")
