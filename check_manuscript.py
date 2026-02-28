import re
import os

files = [
    "thesis_paper/manuscript_5_chapter/chapter0_abstract.md",
    "thesis_paper/manuscript_5_chapter/chapter0_notation.md",
    "thesis_paper/manuscript_5_chapter/chapter1_intro_related.md",
    "thesis_paper/manuscript_5_chapter/revised_manuscript/chapter2_problem_framework.md",
    "thesis_paper/manuscript_5_chapter/chapter3_implementation_setup.md",
    "thesis_paper/manuscript_5_chapter/revised_manuscript/chapter4_results_verification.md",
    "thesis_paper/manuscript_5_chapter/chapter5_discussion_conclusion.md",
    "thesis_paper/manuscript_5_chapter/appendix.md"
]

placeholders = ["TODO", "FIXME", "XX", "待定", "待填", "\[...\]"]
placeholder_re = re.compile("|".join(placeholders))
image_re = re.compile(r"!\[(.*?)\]\((.*?)\)")

for fpath in files:
    if not os.path.exists(fpath):
        print(f"MISSING FILE: {fpath}")
        continue
        
    print(f"--- Checking {fpath} ---")
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.readlines()
        
    for i, line in enumerate(content):
        # Check placeholders
        match = placeholder_re.search(line)
        if match:
            print(f"  Line {i+1}: Placeholder '{match.group()}' found: {line.strip()[:50]}...")
            
        # Check images
        img_match = image_re.search(line)
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            # Basic check if path is likely valid (not empty)
            if not img_path:
                 print(f"  Line {i+1}: Empty image path")
            else:
                 # Check if file exists (relative to md file location)
                 dir_path = os.path.dirname(fpath)
                 full_img_path = os.path.join(dir_path, img_path)
                 if not os.path.exists(full_img_path):
                     print(f"  Line {i+1}: Image not found: {img_path} (Resolved: {full_img_path})")
                 else:
                     print(f"  Line {i+1}: Image OK: {img_path}")

    if len(content) < 10:
        print(f"  WARNING: File seems very short ({len(content)} lines)")

