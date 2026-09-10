
import re
import glob
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def scan_files():
    files = glob.glob(str(PROJECT_ROOT / "thesis_paper/manuscript_gpt_review/*.md"))
    
    print(f"Scanning {len(files)} files...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Check for \big$
            if r'\big$' in line:
                print(f"[{filename}:{i+1}] Found \\big$: {line.strip()}")
                
            # Check for \sum_{$
            if r'\sum_{$' in line:
                 print(f"[{filename}:{i+1}] Found \\sum_{{$: {line.strip()}")
                 
            # Check for $ inside \sum subscripts generally
            # e.g. \sum_{...$
            # This is hard to regex perfectly, but look for \sum_{ followed by $ before }
            match = re.search(r'\\sum_\{[^}]*\$', line)
            if match:
                print(f"[{filename}:{i+1}] Found $ inside \\sum subscript: {line.strip()}")

            # Check for nested $ inside $$...$$
            # Heuristic: line contains $$ and also a single $ that isn't part of $$
            # This is prone to false positives if multiple inline math on same line as display math?
            # But $$ usually takes whole line.
            if '$$' in line:
                clean_line = line.replace('$$', '')
                if '$' in clean_line:
                    # Check if it's just multiple $$ blocks? No, we removed all $$
                    print(f"[{filename}:{i+1}] Found $ inside $$ line: {line.strip()}")

            # Check for $ inside \left ... \right
            # Look for \left followed by ... $ ... \right
            # This requires matching the pair, which is hard line-by-line.
            # But often they are on the same line.
            if r'\left' in line and '$' in line and r'\right' in line:
                 # Check if $ is between left and right
                 # This is too noisy if not careful.
                 pass

if __name__ == "__main__":
    scan_files()
