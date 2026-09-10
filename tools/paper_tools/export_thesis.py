import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add current directory to sys.path to import generate_reference_doc if it's in the same dir
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from generate_reference_doc import create_dmu_reference_doc
except ImportError:
    print("Warning: Could not import generate_reference_doc. Make sure it is in the same directory.")
    # Fallback or exit? For now, we'll try to proceed assuming the doc might exist.
    pass

def get_pandoc_path():
    # Try the local bin first
    local_bin = os.path.expanduser("~/.local/bin/pandoc")
    if os.path.exists(local_bin):
        return local_bin
    
    # Try system pandoc
    try:
        subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "pandoc"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def export_to_docx(input_file, output_file, reference_doc="dmu_reference.docx"):
    # 1. Ensure Reference Doc exists
    if not os.path.exists(reference_doc):
        print(f"Generating reference document: {reference_doc}")
        try:
            create_dmu_reference_doc(reference_doc)
        except Exception as e:
            print(f"Error generating reference doc: {e}")
            return False
    else:
        print(f"Using existing reference document: {reference_doc}")

    # 2. Check Pandoc
    pandoc_cmd = get_pandoc_path()
    if not pandoc_cmd:
        print("Error: Pandoc not found. Please install pandoc.")
        return False
    
    # 3. Build Command
    # pandoc input.md -o output.docx --reference-doc=ref.docx
    cmd = [
        pandoc_cmd,
        str(input_file),
        "-o", str(output_file),
        f"--reference-doc={reference_doc}"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully exported to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc:\n{e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export Markdown manuscript to Word (DOCX) using DMU thesis template.")
    parser.add_argument("input", nargs="?", default="thesis_paper/abstract_sample.md", help="Input Markdown file path")
    parser.add_argument("-o", "--output", help="Output DOCX file path (default: same basename as input)")
    parser.add_argument("--ref", default="dmu_reference.docx", help="Path to reference docx template")
    parser.add_argument("--regen-ref", action="store_true", help="Force regenerate the reference docx")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
        
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".docx")
        
    if args.regen_ref and os.path.exists(args.ref):
        os.remove(args.ref)
        
    export_to_docx(input_path, output_path, args.ref)

if __name__ == "__main__":
    main()
