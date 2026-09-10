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
            "--standalone",  # Important for applying reference doc styles
            "--resource-path=.:" + os.path.dirname(md_file) # Ensure images are found relative to the md file
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

def convert_md_to_pdf(md_files, output_dir, prince_path, filter_path=None, bib_file=None, csl_file=None, css_file=None):
    """
    Convert a list of Markdown files to PDF using Pandoc and PrinceXML via intermediate HTML.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(prince_path):
        print(f"Error: PrinceXML not found at {prince_path}")
        return

    success_count = 0
    
    for md_file in md_files:
        if not os.path.exists(md_file):
            print(f"Warning: File not found: {md_file}")
            continue
            
        # Determine output filename
        file_name = os.path.basename(md_file)
        name_no_ext = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{name_no_ext}.pdf")
        # Generate HTML in the same directory as MD file to preserve relative paths for Prince
        html_path = os.path.join(os.path.dirname(md_file), f"{name_no_ext}.html")
        
        print(f"Converting {file_name} -> {output_path}...")
        
        # Step 1: MD -> HTML
        # Use absolute path for resources to help Prince find them
        # We need to make sure image paths are resolvable from the location of the HTML file
        # or convert them to absolute paths.
        
        cmd_html = [
            "pandoc",
            md_file,
            "-f", "markdown+tex_math_single_backslash",
            "-o", html_path,
            "--standalone",
            "--mathml",
            "--resource-path=.:" + os.path.dirname(md_file) # Ensure images are found relative to the md file
        ]
        
        if filter_path and os.path.exists(filter_path):
            cmd_html.extend(["--lua-filter", filter_path])

        if bib_file and os.path.exists(bib_file):
            cmd_html.extend([
                "--bibliography", bib_file,
                "--citeproc"
            ])
            
            if csl_file and os.path.exists(csl_file):
                cmd_html.extend(["--csl", csl_file])
        
        # Add CSS if provided
        if css_file and os.path.exists(css_file):
            cmd_html.extend(["--css", css_file])
            # We need to make sure the CSS content is embedded or linked correctly for Prince
            # Pandoc's --css adds a link tag. Prince can read local files if path is correct.
            # Alternatively, we can use --include-in-header to embed style.
            # But let's try passing the CSS directly to Prince as well.
        
        try:
            # Generate HTML
            subprocess.run(cmd_html, capture_output=True, text=True, check=True)
            
            # Step 2: HTML -> PDF (using Prince)
            cmd_pdf = [prince_path, html_path, "-o", output_path]
            
            # Pass CSS to Prince explicitly to ensure fonts are loaded
            if css_file and os.path.exists(css_file):
                 cmd_pdf.extend(["--style", css_file])

            # Capture output but print warnings if any
            result = subprocess.run(cmd_pdf, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  [FAIL] Error converting {file_name}:")
                print(result.stderr)
            else:
                # Check for font warnings in stderr even on success
                if "warning: no font for" in result.stderr or "can't open input file" in result.stderr:
                    print(f"  [WARN] Prince warnings for {file_name}:")
                    # Filter only relevant lines to avoid spam
                    for line in result.stderr.splitlines():
                        if "warning:" in line:
                            print(f"    {line}")
                
                print(f"  [OK] Converted {file_name}")
                success_count += 1
            
            # Cleanup
            if os.path.exists(html_path):
                os.remove(html_path)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Error converting {file_name}:")
            print(e.stderr)

    print(f"\nPDF conversion complete. {success_count}/{len(md_files)} files processed.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    # Configuration
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/DMUpapertemplate/template_fixed.docx")
    
    # Target directory: manuscript_5_chapter
    BASE_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_5_chapter")
    DOCX_OUTPUT_DIR = os.path.join(BASE_DIR, "docx_output")
    PDF_OUTPUT_DIR = os.path.join(BASE_DIR, "pdf_output")
    
    FILTER_PATH = os.path.join(PROJECT_ROOT, "tools/thesis_style.lua")
    BIB_PATH = os.path.join(BASE_DIR, "references.bib")
    # CSL_PATH was referenced but not defined in original code, assuming standard path or None
    CSL_PATH = os.path.join(PROJECT_ROOT, "tools/china-national-standard-gb-t-7714-2015-numeric.csl") 
    PRINCE_PATH = os.path.join(PROJECT_ROOT, "tools/prince_local/bin/prince")
    CSS_PATH = os.path.join(PROJECT_ROOT, "tools/prince_style.css")
    
    # Files to convert
    MD_FILES = [
        "chapter0_abstract.md",
        "chapter0_notation.md",
        "chapter1_intro_related.md",
        "chapter2_problem_framework.md",
        "chapter3_implementation_setup.md",
        "chapter4_results_verification.md",
        "chapter5_discussion_conclusion.md",
        "appendix.md",
        "thesis_full.md"
    ]
    
    # Resolve absolute paths
    ABS_MD_FILES = [os.path.join(BASE_DIR, f) for f in MD_FILES]
    
    parser = argparse.ArgumentParser(description="Convert thesis markdown files to DOCX or PDF.")
    parser.add_argument("--format", choices=["docx", "pdf", "all"], default="docx", help="Output format")
    args = parser.parse_args()
    
    if args.format in ["docx", "all"]:
        print("=== Generating DOCX ===")
        convert_md_to_docx(ABS_MD_FILES, TEMPLATE_PATH, DOCX_OUTPUT_DIR, FILTER_PATH, BIB_PATH, CSL_PATH)
        
    if args.format in ["pdf", "all"]:
        print("\n=== Generating PDF ===")
        convert_md_to_pdf(ABS_MD_FILES, PDF_OUTPUT_DIR, PRINCE_PATH, FILTER_PATH, BIB_PATH, CSL_PATH, CSS_PATH)
