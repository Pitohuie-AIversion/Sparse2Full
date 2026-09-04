import re
import os
import sys
from collections import defaultdict
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "thesis_paper" / "manuscript_5_chapter"
NOTATION_FILE = os.path.join(BASE_DIR, "chapter0_notation.md")
CHAPTER_FILES = [
    "chapter1_intro_related.md",
    "chapter2_problem_framework.md",
    "chapter3_implementation_setup.md",
    "chapter4_results_verification.md",
    "chapter5_discussion_conclusion.md",
    "appendix.md",
    "appendix_proofs.md"
]
REPORT_FILE = os.path.join(BASE_DIR, "symbol_consistency_report.md")

def clean_latex_symbol(symbol_str):
    """
    Cleans a latex symbol string for matching.
    Removes surrounding $, spaces.
    """
    s = symbol_str.strip()
    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1]
    return s.strip()

def normalize_latex(s):
    """
    Normalizes latex string for fuzzy matching.
    Replaces \text, \mathrm, \mathbf, \boldsymbol with nothing or a standard form.
    Actually, mostly we want to treat \text{x} and \mathrm{x} as the same.
    """
    # Replace \text{...} with \mathrm{...}
    s = re.sub(r'\\text\{([^}]+)\}', r'\\mathrm{\1}', s)
    return s

def get_core_symbol(symbol_clean):
    """
    Extracts the core symbol from a definition.
    """
    # Remove (\cdot) or (anything) at the end
    s = re.sub(r'\([^\)]+\)$', '', symbol_clean)
    # Remove ^(z) or similar superscripts for core matching? Maybe not.
    return s.strip()

def extract_symbols(notation_path):
    """
    Extracts symbols from the notation table.
    Returns a dict: {symbol_clean: {'original': symbol_raw, 'desc': description, 'core': core_symbol}}
    """
    symbols = {}
    if not os.path.exists(notation_path):
        print(f"Error: Notation file not found at {notation_path}")
        return {}
        
    with open(notation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith("| 符号"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
            
        symbol_raw = parts[1]
        desc = parts[3]
        
        if "**" in symbol_raw or not symbol_raw or symbol_raw == "Symbol":
            continue
            
        symbol_clean = clean_latex_symbol(symbol_raw)
        
        # Handle comma separated symbols (e.g. \lambda_1, \lambda_2)
        sub_symbols = [s.strip() for s in symbol_clean.split(',')]
        
        for sub_sym in sub_symbols:
            if not sub_sym: continue
            
            core = get_core_symbol(sub_sym)
            symbols[sub_sym] = {
                'original': symbol_raw, # Keep original full string for display
                'desc': desc,
                'core': core,
                'normalized': normalize_latex(sub_sym)
            }
        
    return symbols

def extract_math_expressions(content):
    """
    Extracts all math expressions ($...$ and $$...$$) from markdown content.
    """
    pattern = r'\$\$([\s\S]*?)\$\$|\$([^\n\$]*?)\$'
    matches = re.findall(pattern, content)
    
    exprs = []
    for m in matches:
        if m[0]:
            exprs.append(m[0].strip())
        elif m[1]:
            exprs.append(m[1].strip())
    return exprs

def scan_file(file_path, defined_symbols):
    """
    Scans a file for math expressions and checks usage.
    """
    if not os.path.exists(file_path):
        return set(), set(), [], 0
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    math_exprs = extract_math_expressions(content)
    found_symbols = set()
    found_variations = set()
    
    for sym_clean, info in defined_symbols.items():
        core = info['core']
        norm = info['normalized']
        
        matched = False
        
        for expr in math_exprs:
            # Normalize expression for fuzzy check
            expr_norm = normalize_latex(expr)
            
            # Check 1: Exact match of clean symbol
            if sym_clean in expr:
                found_symbols.add(sym_clean)
                matched = True
                break
            
            # Check 2: Normalized match (handles \text vs \mathrm)
            if norm in expr_norm:
                found_symbols.add(sym_clean) # Mark as found
                found_variations.add(sym_clean) # Mark as variation
                matched = True
                break
                
            # Check 3: Core symbol match (substring)
            # Use negative lookbehind logic for \
            if core in expr:
                idx = expr.find(core)
                # Ensure it's not part of a latex command (e.g. u in \mu)
                if idx > 0 and expr[idx-1] == '\\':
                    pass
                else:
                    found_symbols.add(sym_clean)
                    matched = True
                    break
        
    return found_symbols, found_variations, math_exprs, len(math_exprs)

def generate_report(symbols, file_stats):
    report = ["# 符号一致性检查报告 (Symbol Consistency Report)\n"]
    report.append(f"基准文件: `{os.path.basename(NOTATION_FILE)}`\n")
    report.append(f"检查时间: {os.popen('date').read().strip()}\n")
    
    report.append("## 1. 已定义符号的使用情况 (Defined Symbols Usage)\n")
    report.append("此表列出符号说明表中的符号在各章节中是否被检测到使用。\n")
    report.append("| 符号 (Symbol) | 核心 (Core) | 说明 (Description) | 出现章节 (Found In) | 状态 (Status) |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")
    
    used_count = 0
    unused_count = 0
    
    # Deduplicate original entries if we split commas
    # We want to list unique symbols from the dict keys
    sorted_keys = sorted(symbols.keys())
    
    for sym in sorted_keys:
        info = symbols[sym]
        found_in = []
        is_variation = False
        
        for fname, stats in file_stats.items():
            if sym in stats['found']:
                short_name = fname.split('_')[0].replace('chapter', 'Ch').replace('appendix', 'App')
                if short_name == "thesis": short_name = "Full"
                found_in.append(short_name)
                if sym in stats['variations']:
                    is_variation = True
        
        if found_in:
            status = "✅ Used"
            if is_variation:
                status += " (Var)"
            used_count += 1
            files_str = ", ".join(found_in)
        else:
            status = "⚠️ **Unused**"
            unused_count += 1
            files_str = "-"
            
        # Use the specific symbol string for the first column, not the full original line
        display_sym = f"${sym}$"
        report.append(f"| {display_sym} | {info['core']} | {info['desc']} | {files_str} | {status} |")
        
    report.append(f"\n**统计**: 共定义 {len(symbols)} 个符号，已使用 {used_count} 个，未使用 {unused_count} 个。\n")
    report.append("> 注：(Var) 表示检测到 LaTeX 格式变体（如 `\\text` vs `\\mathrm`）。\n")
    
    report.append("## 2. 章节数学公式统计 (Math Expression Stats)\n")
    report.append("| 文件名 (Chapter) | 公式数量 (Math Count) | 已定义符号覆盖率 (Defined Coverage) |")
    report.append("| :--- | :--- | :--- |")
    
    for fname, stats in file_stats.items():
        count = stats['math_count']
        found = len(stats['found'])
        coverage = f"{found}/{len(symbols)}"
        report.append(f"| {fname} | {count} | {coverage} |")

    report.append("\n## 3. 潜在未定义符号 (Potential Undefined Symbols)\n")
    report.append("*(此功能为实验性功能，列出频繁出现但不在表中的单字母变量)*\n")
    report.append("> 提示：请人工核对以下章节中频繁出现的变量是否遗漏在符号表中。\n")
    
    return "\n".join(report)

def main():
    print(f"Reading notation from {NOTATION_FILE}...")
    symbols = extract_symbols(NOTATION_FILE)
    print(f"Found {len(symbols)} defined symbols.")
    
    file_stats = {}
    
    for fname in CHAPTER_FILES:
        fpath = os.path.join(BASE_DIR, fname)
        print(f"Scanning {fname}...")
        found, variations, math_exprs, count = scan_file(fpath, symbols)
        file_stats[fname] = {'found': found, 'variations': variations, 'math_count': count}
        
    print("Generating report...")
    report_content = generate_report(symbols, file_stats)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report written to {REPORT_FILE}")

if __name__ == "__main__":
    main()
