import os
import glob
import argparse
import base64
import subprocess
from pathlib import Path

def encode_svg_to_base64(filepath):
    with open(filepath, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = str(PROJECT_ROOT / 'drd_paper_1m')
DEFAULT_PRINCE_BIN = str(PROJECT_ROOT / 'tools' / 'prince_local' / 'bin' / 'prince')

def main():
    parser = argparse.ArgumentParser(description='Arrange model SVGs into a paper figure using HTML and Prince.')
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Base directory containing model folders')
    parser.add_argument('--filename', type=str, default='sample_0059_obs_gt_pred_error_t70_1.svg',
                        help='Target SVG filename to compare')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the grid')
    parser.add_argument('--output_pdf', type=str, default=None, help='Output PDF path')
    
    args = parser.parse_args()
    
    search_pattern = os.path.join(args.base_dir, '*', 'test_visualizations', 'visualizations', 'predictions', args.filename)
    svg_files = glob.glob(search_pattern)
    
    if not svg_files:
        print(f"No files found matching {search_pattern}")
        return
        
    model_svgs = []
    for f in svg_files:
        parts = f.split(os.sep)
        model_dir = parts[-5]
        try:
            model_name = model_dir.split('-')[2]
        except IndexError:
            model_name = model_dir
        model_svgs.append((model_name, f))
        
    model_svgs.sort()
    
    # Generate HTML
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  @page { size: 3000px 1500px; margin: 0; }",  # Big page size to fit the grid
        "  body { font-family: 'Arial', sans-serif; margin: 0; padding: 20px; background: white; }",
        "  .grid-container { display: table; border-collapse: separate; border-spacing: 10px; }",
        "  .grid-row { display: table-row; }",
        "  .grid-cell { display: table-cell; text-align: center; vertical-align: top; padding: 10px; }",
        "  .model-title { font-size: 32px; font-weight: bold; margin-bottom: 10px; }",
        "  img { width: 650px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='grid-container'>"
    ]
    
    # 4 columns, dynamic rows
    for i in range(0, len(model_svgs), args.cols):
        html_content.append("  <div class='grid-row'>")
        for j in range(args.cols):
            if i + j < len(model_svgs):
                model_name, svg_path = model_svgs[i + j]
                b64_img = encode_svg_to_base64(svg_path)
                html_content.append(f"    <div class='grid-cell'>")
                html_content.append(f"      <div class='model-title'>{model_name}</div>")
                html_content.append(f"      <img src='{b64_img}' alt='{model_name}'>")
                html_content.append(f"    </div>")
            else:
                html_content.append(f"    <div class='grid-cell'></div>")
        html_content.append("  </div>")
        
    html_content.append("</div>")
    html_content.append("</body></html>")
    
    html_path = os.path.join(args.base_dir, f"temp_{args.filename.replace('.svg', '.html')}")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
        
    if args.output_pdf is None:
        args.output_pdf = os.path.join(args.base_dir, f"paper_figure_prince_{args.filename.replace('.svg', '.pdf')}")
        
    print("Running PrinceXML to generate PDF...")
    prince_bin = DEFAULT_PRINCE_BIN
    subprocess.run([prince_bin, html_path, "-o", args.output_pdf])
    
    # Also generate an SVG by wrapping everything in an SVG file!
    svg_out = args.output_pdf.replace('.pdf', '.svg')
    svg_content = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2800" height="1400">'
    ]
    
    img_width = 657
    img_height = 568
    pad_x = 20
    pad_y = 60
    
    for i, (model_name, svg_path) in enumerate(model_svgs):
        col = i % args.cols
        row = i // args.cols
        x = col * (img_width + pad_x)
        y = row * (img_height + pad_y)
        
        b64_img = encode_svg_to_base64(svg_path)
        
        svg_content.append(f'  <g transform="translate({x}, {y})">')
        svg_content.append(f'    <text x="{img_width/2}" y="30" font-family="Arial" font-size="28" font-weight="bold" text-anchor="middle">{model_name}</text>')
        svg_content.append(f'    <image x="0" y="40" width="{img_width}" height="{img_height}" xlink:href="{b64_img}"/>')
        svg_content.append(f'  </g>')
        
    svg_content.append('</svg>')
    
    with open(svg_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
        
    print(f"Generated PDF: {args.output_pdf}")
    print(f"Generated embedded SVG: {svg_out}")
    os.remove(html_path)

if __name__ == '__main__':
    main()
