
import re

FILE_PATH = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/manuscript_gpt_review/chapter3.md"

def check_file():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Scanning {len(lines)} lines...")
    
    for i, line in enumerate(lines):
        # Check for nested $ inside $$...$$
        # This is hard to do line-by-line if $$ spans lines, but let's check for $ inside lines that look like math
        
        # Check for \big$
        if r'\big$' in line:
            print(f"Line {i+1}: Found \\big$: {line.strip()}")
            
        # Check for \sum_{$
        if r'\sum_{$' in line:
             print(f"Line {i+1}: Found \\sum_{{$: {line.strip()}")

        # Check for $ inside \left ... \right
        # Regex for $\w+ inside \left
        
        # Check for any $...$ inside $$...$$
        # Simple heuristic: if line has $$ and also has $ (that is not part of $$)
        # Note: $$...$$ is usually display math. 
        if '$$' in line:
            # Remove $$
            inner = line.replace('$$', '')
            if '$' in inner:
                 print(f"Line {i+1}: Found $ inside $$ line: {line.strip()}")

        # Check for $ inside \sum subscripts
        if r'\sum' in line and '$' in line:
             # Look for \sum_{...$
             pass

if __name__ == "__main__":
    check_file()
