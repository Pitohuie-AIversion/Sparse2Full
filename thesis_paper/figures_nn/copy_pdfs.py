import os
import shutil
import glob

source_root = "thesis_paper/figures_nn/build_export"
dest_root = "thesis_paper/figures_nn/build_export_j2"

# Create destination root if it doesn't exist
os.makedirs(dest_root, exist_ok=True)

# Find all PDF files in the source directory matching the pattern
# Pattern: build_export/<model_name>/fig_<model_name>_auto.pdf
# We search recursively
pdf_files = glob.glob(os.path.join(source_root, "**", "fig_*_auto.pdf"), recursive=True)

print(f"Found {len(pdf_files)} PDF files to copy.")

for source_path in pdf_files:
    # Get relative path from source root
    rel_path = os.path.relpath(source_path, source_root)
    
    # Construct destination path
    dest_path = os.path.join(dest_root, rel_path)
    
    # Create destination directory
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    print(f"Copied: {rel_path}")

print("Done.")
