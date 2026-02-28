
import os
import subprocess
from pathlib import Path

# 设置路径
base_dir = Path("thesis_paper/figures_nn/build_export_j2")

# 获取所有PDF文件
pdf_files = list(base_dir.rglob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files.")

# 使用 conda run 在 pdf_tools 环境中执行转换
for pdf_path in pdf_files:
    svg_path = pdf_path.with_suffix('.svg')
    
    # 如果SVG已存在且比PDF新，则跳过
    if svg_path.exists() and svg_path.stat().st_mtime > pdf_path.stat().st_mtime:
        print(f"Skipping {pdf_path} (SVG is up to date)")
        continue
        
    print(f"Converting {pdf_path} -> {svg_path}")
    
    cmd = [
        "conda", "run", "-n", "pdf_tools", 
        "pdf2svg", str(pdf_path), str(svg_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {pdf_path}: {e.stderr.decode()}")
