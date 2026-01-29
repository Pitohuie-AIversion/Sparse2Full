import os
import subprocess
import sys
from pathlib import Path

def convert_md_to_docx(md_files, template_path, output_dir, filter_path, bib_file=None, csl_file=None):
    """
    Convert a list of Markdown files to Docx using Pandoc and a custom template.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return
    if not os.path.exists(filter_path):
        print(f"Error: Filter not found at {filter_path}")
        return

    success_count = 0
    
    for md_file in md_files:
        if not os.path.exists(md_file):
            print(f"Warning: File not found: {md_file}")
            continue
            
        # Determine output filename
        file_name = os.path.basename(md_file)
        name_no_ext = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{name_no_ext}.docx")
        
        print(f"Converting {file_name} -> {output_path}...")
        
        # Construct Pandoc command
        # --reference-doc: Uses styles from the template
        # --lua-filter: Applies custom style mapping
        # --standalone: Produces a standalone document
        cmd = [
            "pandoc",
            md_file,
            "-f", "markdown+tex_math_single_backslash",
            "-o", output_path,
            "--reference-doc", template_path,
            "--lua-filter", filter_path,
            "--standalone"  # Important for applying reference doc styles
        ]

        # Add bibliography support if file exists
        if bib_file and os.path.exists(bib_file):
            cmd.extend([
                "--bibliography", bib_file,
                "--citeproc" # Enable citation processing
            ])
            
            # Add CSL style if provided
            if csl_file and os.path.exists(csl_file):
                cmd.extend(["--csl", csl_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  [OK] Converted {file_name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Error converting {file_name}:")
            print(e.stderr)

    print(f"\nconversion complete. {success_count}/{len(md_files)} files processed.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configuration
    PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
    TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/DMUpapertemplate/template_fixed.docx")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/docx_output")
    FILTER_PATH = os.path.join(PROJECT_ROOT, "tools/thesis_style.lua")
    BIB_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/latex/manuscript/references.bib")
    
    # Files to convert
    MD_FILES = [
        "thesis_paper/manuscript_gpt_review/chapter0_abstract.md",
        "thesis_paper/manuscript_gpt_review/chapter0_notation.md",
        "thesis_paper/manuscript_gpt_review/chapter1.md",
        "thesis_paper/manuscript_gpt_review/chapter2.md",
        "thesis_paper/manuscript_gpt_review/chapter3.md",
        "thesis_paper/manuscript_gpt_review/chapter4.md",
        "thesis_paper/manuscript_gpt_review/chapter5.md",
        "thesis_paper/manuscript_gpt_review/chapter6.md",
        "thesis_paper/manuscript_gpt_review/chapter7.md",
        "thesis_paper/manuscript_gpt_review/chapter8.md",
        "thesis_paper/manuscript_gpt_review/chapter9.md",
        "thesis_paper/manuscript_gpt_review/symbol_checklist.md",
        "thesis_paper/manuscript_gpt_review/template.md",
        "thesis_paper/manuscript_gpt_review/writing_checklist.md"
    ]
    
    # Resolve absolute paths
    ABS_MD_FILES = [os.path.join(PROJECT_ROOT, f) for f in MD_FILES]
    
    convert_md_to_docx(ABS_MD_FILES, TEMPLATE_PATH, OUTPUT_DIR, FILTER_PATH, BIB_PATH, CSL_PATH)
