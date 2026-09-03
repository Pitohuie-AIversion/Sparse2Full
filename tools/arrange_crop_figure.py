import os
import glob
import re
import argparse
import svgutils.transform as sg
import tempfile
import base64
import subprocess

def sanitize_svg(content):
    idx = content.find("</svg>")
    if idx != -1:
        return content[:idx + 6]
    return content

def prefix_svg_ids(svg_content, prefix):
    svg_content = sanitize_svg(svg_content)
    id_pattern = re.compile(r'id="([^"]+)"')
    found_ids = set(id_pattern.findall(svg_content))
    
    svg_content = id_pattern.sub(lambda m: f'id="{prefix}_{m.group(1)}"', svg_content)
    
    for old_id in found_ids:
        svg_content = svg_content.replace(f'url(#{old_id})', f'url(#{prefix}_{old_id})')
        svg_content = svg_content.replace(f'href="#{old_id}"', f'href="#{prefix}_{old_id}"')
        svg_content = svg_content.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{prefix}_{old_id}"')
        
    return svg_content

def encode_svg_to_base64(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = sanitize_svg(content)
    encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"

def main():
    sizes = [112, 96, 80, 64, 48, 32, 16, 8, 4, 1]
    models = ['UNet']
    
    # We want a 5x2 grid instead of 1x10
    cols = 5
    
    # Pre-find the files
    grid_files = {}
    for size in sizes:
        for model in models:
            pattern = ''
            if model == 'UNet':
                pattern = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs/UNet_Crop_Scan/AR-DR2D-Crop-Scan-Size{size}-UNet/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0.svg'
            elif model == 'EDSR':
                pattern = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/AR-DR2D-Crop-Scan-Size{size}-model_EDSR-*/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0.svg'
            elif model == 'PartialConvUNet':
                pattern = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size{size}-*/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0.svg'
                
            files = glob.glob(pattern)
            if files:
                grid_files[(size, model)] = files[0]
            else:
                grid_files[(size, model)] = None
                
    # We will generate a PrinceXML based PDF and SVG because it's safer and cleaner.
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  @page { size: 4000px 2000px; margin: 0; }",  # Adjusted page size for 5 columns, 2 rows
        "  body { font-family: 'Arial', sans-serif; margin: 0; padding: 20px; background: white; }",
        "  .grid-container { display: table; border-collapse: separate; border-spacing: 15px; margin: 0 auto; }",
        "  .grid-row { display: table-row; }",
        "  .grid-cell { display: table-cell; text-align: center; vertical-align: middle; padding: 10px; width: 650px; height: 650px; border: 1px dashed #eee; }",
        "  .grid-cell.header { height: auto; border: none; font-size: 36px; font-weight: bold; padding-bottom: 20px; }",
        "  .grid-cell.row-header { width: auto; font-size: 32px; font-weight: bold; padding-right: 30px; text-align: right; border: none; }",
        "  .missing-text { font-size: 48px; color: #999; font-weight: bold; }",
        "  img { width: 650px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='grid-container'>",
    ]
    
    # 5 columns, 2 rows logic
    for i in range(0, len(sizes), cols):
        html_content.append("  <div class='grid-row'>")
        for j in range(cols):
            if i + j < len(sizes):
                size = sizes[i + j]
                svg_path = grid_files[(size, 'UNet')]
                html_content.append(f"    <div class='grid-cell'>")
                html_content.append(f"      <div class='grid-cell header' style='text-align: center;'>Size {size}</div>")
                if svg_path:
                    b64_img = encode_svg_to_base64(svg_path)
                    html_content.append(f"      <img src='{b64_img}' alt='Size {size}'>")
                else:
                    html_content.append(f"      <div class='missing-text'>-</div>")
                html_content.append(f"    </div>")
            else:
                html_content.append(f"    <div class='grid-cell'></div>")
        html_content.append("  </div>")
        
    html_content.append("</div>")
    html_content.append("</body></html>")
    
    out_dir = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs"
    html_path = os.path.join(out_dir, "temp_crop_comparison.html")
    pdf_output = os.path.join(out_dir, "paper_figure_crop_scan_prince.pdf")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
        
    print("Running PrinceXML to generate PDF...")
    prince_bin = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/prince_local/bin/prince"
    subprocess.run([prince_bin, html_path, "-o", pdf_output])
    
    # We will also generate the pure SVG using sg.SVGFigure
    print("Generating pure SVG with prefixed IDs...")
    
    img_width = 743
    img_height = 712
    pad_x = 20
    pad_y = 60
    header_h = 80
    
    rows = (len(sizes) + cols - 1) // cols
    fig_width = cols * (img_width + pad_x)
    fig_height = rows * (img_height + pad_y + header_h)
    
    fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")
    plots = []
    
    temp_dir = tempfile.mkdtemp()
    
    for idx, size in enumerate(sizes):
        col = idx % cols
        row = idx // cols
        
        x = col * (img_width + pad_x)
        y_base = row * (img_height + pad_y + header_h)
        
        # Add header
        txt = sg.TextElement(x + img_width/2, y_base + header_h - 20, f"Size {size}", size=36, weight="bold", anchor="middle", font="Arial")
        plots.append(txt)
        
        y_img = y_base + header_h
        
        svg_path = grid_files[(size, 'UNet')]
        if svg_path:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            prefix = f"crop_s{size}_UNet"
            prefixed_content = prefix_svg_ids(content, prefix)
            temp_path = os.path.join(temp_dir, f"{prefix}.svg")
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(prefixed_content)
            
            svg_plot = sg.fromfile(temp_path).getroot()
            svg_plot.moveto(x, y_img)
            plots.append(svg_plot)
            os.remove(temp_path)
        else:
            txt_missing = sg.TextElement(x + img_width/2, y_img + img_height/2, "-", size=48, weight="bold", anchor="middle", font="Arial", color="#999")
            plots.append(txt_missing)

    os.rmdir(temp_dir)
    fig.append(plots)
    svg_output = os.path.join(out_dir, "paper_figure_crop_scan_fixed.svg")
    fig.save(svg_output)
    
    print("Generating embedded SVG...")
    svg_out_embedded = os.path.join(out_dir, "paper_figure_crop_scan_embedded.svg")
    svg_content = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{fig_width}" height="{fig_height}">'
    ]
    
    for idx, size in enumerate(sizes):
        col = idx % cols
        row = idx // cols
        
        x = col * (img_width + pad_x)
        y_base = row * (img_height + pad_y + header_h)
        
        svg_content.append(f'  <text x="{x + img_width/2}" y="{y_base + header_h - 20}" font-family="Arial" font-size="36" font-weight="bold" text-anchor="middle">Size {size}</text>')
        
        y_img = y_base + header_h
        svg_path = grid_files[(size, 'UNet')]
        
        if svg_path:
            b64_img = encode_svg_to_base64(svg_path)
            svg_content.append(f'  <image x="{x}" y="{y_img}" width="{img_width}" height="{img_height}" xlink:href="{b64_img}"/>')
        else:
            svg_content.append(f'  <text x="{x + img_width/2}" y="{y_img + img_height/2}" font-family="Arial" font-size="48" font-weight="bold" fill="#999" text-anchor="middle">-</text>')
            
    svg_content.append('</svg>')
    with open(svg_out_embedded, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))

    print(f"Generated PDF: {pdf_output}")
    print(f"Generated prefixed SVG: {svg_output}")
    print(f"Generated embedded SVG: {svg_out_embedded}")
    print(f"Saved HTML to: {html_path}")
    
    os.remove(html_path)

if __name__ == '__main__':
    main()
