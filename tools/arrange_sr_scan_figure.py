import os
import glob
import re
import argparse
import svgutils.transform as sg
import tempfile
import base64
import subprocess

def prefix_svg_ids(svg_content, prefix):
    id_pattern = re.compile(r'id="([^"]+)"')
    found_ids = set(id_pattern.findall(svg_content))
    
    svg_content = id_pattern.sub(lambda m: f'id="{prefix}_{m.group(1)}"', svg_content)
    
    for old_id in found_ids:
        svg_content = svg_content.replace(f'url(#{old_id})', f'url(#{prefix}_{old_id})')
        svg_content = svg_content.replace(f'href="#{old_id}"', f'href="#{prefix}_{old_id}"')
        svg_content = svg_content.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{prefix}_{old_id}"')
        
    return svg_content

def sanitize_svg(content):
    idx = content.find("</svg>")
    if idx != -1:
        return content[:idx + 6]
    return content

def encode_svg_to_base64(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = sanitize_svg(content)
    encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"

def main():
    # Scales: 4, 8, 16, 32, 64, 128
    # Input sizes: 32, 16, 8, 4, 2, 1
    scales = [4, 8, 16, 32, 64, 128]
    input_sizes = [32, 16, 8, 4, 2, 1]
    
    # We want a 3x2 grid (6 items)
    cols = 3
    
    grid_files = {}
    for i, size in enumerate(input_sizes):
        if size in [4, 2, 1]:
            # For the last three sizes, use test_sample_1_obs_gt_pred_error_t0_113.svg
            pattern = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/sr_scan_batch/AR-DR2D-SR-Scan-Input{size}/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0_113.svg'
            files = glob.glob(pattern)
            if files:
                grid_files[size] = files[0]
            else:
                grid_files[size] = None
                print(f"Warning: File missing for input size {size}")
        else:
            # For larger sizes, use sample_0027 or fallback
            pattern_27 = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/sr_scan_batch/AR-DR2D-SR-Scan-Input{size}/test_visualizations/visualizations/predictions/sample_0027_obs_gt_pred_error_t0_t0_1.svg'
            pattern_1 = f'/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper/sr_scan_batch/AR-DR2D-SR-Scan-Input{size}/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0_1.svg'
            
            files_27 = glob.glob(pattern_27)
            files_1 = glob.glob(pattern_1)
            
            if files_27:
                grid_files[size] = files_27[0]
            elif files_1:
                grid_files[size] = files_1[0]
                print(f"Info: Using fallback test_sample_1 for input size {size}")
            else:
                grid_files[size] = None
                print(f"Warning: File missing for input size {size}")
            
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  @page { size: 2500px 2000px; margin: 0; }",
        "  body { font-family: 'Arial', sans-serif; margin: 0; padding: 20px; background: white; }",
        "  .grid-container { display: table; border-collapse: separate; border-spacing: 15px; margin: 0 auto; }",
        "  .grid-row { display: table-row; }",
        "  .grid-cell { display: table-cell; text-align: center; vertical-align: middle; padding: 10px; width: 650px; height: 650px; border: 1px dashed #eee; }",
        "  .grid-cell.header { height: auto; border: none; font-size: 36px; font-weight: bold; padding-bottom: 20px; }",
        "  .missing-text { font-size: 48px; color: #999; font-weight: bold; }",
        "  img { width: 650px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='grid-container'>"
    ]
    
    for i in range(0, len(input_sizes), cols):
        html_content.append("  <div class='grid-row'>")
        for j in range(cols):
            if i + j < len(input_sizes):
                size = input_sizes[i + j]
                scale = scales[i + j]
                svg_path = grid_files[size]
                
                html_content.append(f"    <div class='grid-cell'>")
                html_content.append(f"      <div class='grid-cell header' style='text-align: center;'>Scale x{scale} (Input {size}x{size})</div>")
                
                if svg_path:
                    b64_img = encode_svg_to_base64(svg_path)
                    html_content.append(f"      <img src='{b64_img}' alt='Scale x{scale}'>")
                else:
                    html_content.append(f"      <div class='missing-text'>-</div>")
                html_content.append(f"    </div>")
            else:
                html_content.append(f"    <div class='grid-cell'></div>")
        html_content.append("  </div>")
        
    html_content.append("</div>")
    html_content.append("</body></html>")
    
    out_dir = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd_paper"
    html_path = os.path.join(out_dir, "temp_sr_comparison.html")
    pdf_output = os.path.join(out_dir, "paper_figure_sr_scan_edsr_prince.pdf")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
        
    print("Running PrinceXML to generate PDF...")
    prince_bin = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/prince_local/bin/prince"
    subprocess.run([prince_bin, html_path, "-o", pdf_output])
    
    print("Generating pure SVG with prefixed IDs...")
    
    img_width = 743
    img_height = 712
    pad_x = 20
    pad_y = 60
    header_h = 80
    
    rows = (len(input_sizes) + cols - 1) // cols
    fig_width = cols * (img_width + pad_x)
    fig_height = rows * (img_height + pad_y + header_h)
    
    fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")
    plots = []
    
    temp_dir = tempfile.mkdtemp()
    
    for idx, size in enumerate(input_sizes):
        scale = scales[idx]
        col = idx % cols
        row = idx // cols
        
        x = col * (img_width + pad_x)
        y_base = row * (img_height + pad_y + header_h)
        
        # Add header
        txt = sg.TextElement(x + img_width/2, y_base + header_h - 20, f"Scale x{scale} (Input {size}x{size})", size=36, weight="bold", anchor="middle", font="Arial")
        plots.append(txt)
        
        y_img = y_base + header_h
        
        svg_path = grid_files[size]
        if svg_path:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            prefix = f"sr_s{size}_EDSR"
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
    svg_output = os.path.join(out_dir, "paper_figure_sr_scan_edsr_fixed.svg")
    fig.save(svg_output)
    
    print("Generating embedded SVG...")
    svg_out_embedded = os.path.join(out_dir, "paper_figure_sr_scan_edsr_embedded.svg")
    svg_content = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{fig_width}" height="{fig_height}">'
    ]
    
    for idx, size in enumerate(input_sizes):
        scale = scales[idx]
        col = idx % cols
        row = idx // cols
        
        x = col * (img_width + pad_x)
        y_base = row * (img_height + pad_y + header_h)
        
        svg_content.append(f'  <text x="{x + img_width/2}" y="{y_base + header_h - 20}" font-family="Arial" font-size="36" font-weight="bold" text-anchor="middle">Scale x{scale} (Input {size}x{size})</text>')
        
        y_img = y_base + header_h
        svg_path = grid_files[size]
        
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