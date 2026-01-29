
import os
import re
import glob

TEMPLATE_DIR = '/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/thesis_paper/figures_nn/templates'

def upgrade_templates():
    files = glob.glob(os.path.join(TEMPLATE_DIR, '*_3d.tex.j2'))
    
    # Pattern to match shift={(X,0,0)}
    # Note the parentheses inside the braces!
    pattern = re.compile(r'shift=\{\((\s*[\d\.]+\s*),\s*0\s*,\s*0\s*\)\}')
    
    for fpath in files:
        print(f"Processing {os.path.basename(fpath)}...")
        with open(fpath, 'r') as f:
            content = f.read()
            
        # Check if already upgraded
        if 'dist_scale' in content:
            print("  Already upgraded.")
            continue
            
        def replacement(match):
            val = match.group(1).strip()
            # If val is 0 or 0.0, don't multiply
            if float(val) == 0:
                return f"shift={{({val},0,0)}}"
            return f"shift={{({val} * {{{{ dist_scale|default(1.0) }}}},0,0)}}"
            
        new_content = pattern.sub(replacement, content)
        
        if new_content != content:
            with open(fpath, 'w') as f:
                f.write(new_content)
            print("  Updated.")
        else:
            print("  No matches found.")

if __name__ == "__main__":
    upgrade_templates()
