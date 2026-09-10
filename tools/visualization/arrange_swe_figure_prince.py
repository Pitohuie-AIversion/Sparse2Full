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
RUN_DIR = PROJECT_ROOT / 'runs'
BASELINE_DIR = PROJECT_ROOT / 'runs_baseline'
DEFAULT_OUTPUT_PDF = str(PROJECT_ROOT / 'runs' / 'paper_figure_swe_sample_0059_t70_prince.pdf')

MODEL_CONFIGS = [
    ('EDSR (Ours)', RUN_DIR / 'AR-SW-10M-edsr' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('NAFNet', RUN_DIR / 'AR-SW-10M-nafnet' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('ResNetLite', RUN_DIR / 'AR-SW-10M-resnetlite' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70.svg'),
    ('UNO', RUN_DIR / 'AR-SW-10M-uno' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('SegFormer', RUN_DIR / 'AR-SW-10M-segformer' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('Bicubic', BASELINE_DIR / 'viz_bicubic' / 'visualizations' / 'predictions' / 'sample_0059_st70_obs_gt_pred_error_t0.svg'),
    ('MLP-Model', RUN_DIR / 'AR-SW-10M-mlpmodel' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('SwinUNet', RUN_DIR / 'AR-SW-10M-swin_unet' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
    ('Bilinear', RUN_DIR / 'AR-SW-10M-bilinear3x3decoder' / 'test_visualizations' / 'visualizations' / 'predictions' / 'sample_0059_obs_gt_pred_error_t70_1.svg'),
]

def main():
    parser = argparse.ArgumentParser(description='Arrange SWE model SVGs into a 3x3 paper figure using HTML and Prince.')
    parser.add_argument('--output_pdf', type=str, default=DEFAULT_OUTPUT_PDF)
    args = parser.parse_args()
    
    model_configs = [(name, str(path)) for name, path in MODEL_CONFIGS]

    cols = 3
    
    # Generate HTML
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  @page { size: 2100px 2000px; margin: 0; }",  # Big page size to fit the grid
        "  body { font-family: 'Arial', sans-serif; margin: 0; padding: 20px; background: white; }",
        "  .grid-container { display: table; border-collapse: separate; border-spacing: 10px; margin: 0 auto; }",
        "  .grid-row { display: table-row; }",
        "  .grid-cell { display: table-cell; text-align: center; vertical-align: top; padding: 10px; }",
        "  .model-title { font-size: 32px; font-weight: bold; margin-bottom: 10px; }",
        "  img { width: 650px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='grid-container'>"
    ]
    
    for i in range(0, len(model_configs), cols):
        html_content.append("  <div class='grid-row'>")
        for j in range(cols):
            if i + j < len(model_configs):
                model_name, svg_path = model_configs[i + j]
                if not os.path.exists(svg_path):
                    print(f"WARNING: File missing for {model_name}: {svg_path}")
                    b64_img = ""
                else:
                    b64_img = encode_svg_to_base64(svg_path)
                    
                html_content.append(f"    <div class='grid-cell'>")
                html_content.append(f"      <div class='model-title'>{model_name}</div>")
                if b64_img:
                    html_content.append(f"      <img src='{b64_img}' alt='{model_name}'>")
                html_content.append(f"    </div>")
            else:
                html_content.append(f"    <div class='grid-cell'></div>")
        html_content.append("  </div>")
        
    html_content.append("</div>")
    html_content.append("</body></html>")
    
    html_path = os.path.join(os.path.dirname(args.output_pdf), f"temp_swe_comparison.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
        
    print("Running PrinceXML to generate PDF...")
    prince_bin = str(PROJECT_ROOT / 'tools' / 'prince_local' / 'bin' / 'prince')
    subprocess.run([prince_bin, html_path, "-o", args.output_pdf])
    
    # Embedded SVG
    svg_out = args.output_pdf.replace('.pdf', '.svg')
    svg_content = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2100" height="2000">'
    ]
    
    img_width = 657
    img_height = 568
    pad_x = 20
    pad_y = 60
    
    for i, (model_name, svg_path) in enumerate(model_configs):
        if not os.path.exists(svg_path):
            continue
        col = i % cols
        row = i // cols
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
    print(f"Saved HTML to: {html_path}")

if __name__ == '__main__':
    main()
