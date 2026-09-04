import os
import re
import argparse
import svgutils.transform as sg
import tempfile
from pathlib import Path

def prefix_svg_ids(svg_content, prefix):
    """
    Prefixes all IDs in the SVG content to prevent clashes when merging multiple SVGs.
    """
    id_pattern = re.compile(r'id="([^"]+)"')
    found_ids = set(id_pattern.findall(svg_content))
    
    svg_content = id_pattern.sub(lambda m: f'id="{prefix}_{m.group(1)}"', svg_content)
    
    for old_id in found_ids:
        svg_content = svg_content.replace(f'url(#{old_id})', f'url(#{prefix}_{old_id})')
        svg_content = svg_content.replace(f'href="#{old_id}"', f'href="#{prefix}_{old_id}"')
        svg_content = svg_content.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{prefix}_{old_id}"')
        
    return svg_content


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / 'runs'
BASELINE_DIR = PROJECT_ROOT / 'runs_baseline'
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
    parser = argparse.ArgumentParser(description='Arrange SWE model SVGs into a 3x3 paper figure.')
    parser.add_argument('--output', type=str, default=str(PROJECT_ROOT / 'runs' / 'paper_figure_swe_sample_0059_t70.svg'))
    args = parser.parse_args()
    
    # Define exact paths and titles for the 3x3 grid
    model_configs = [(name, str(path)) for name, path in MODEL_CONFIGS]
    
    cols = 3
    rows = 3
    
    temp_dir = tempfile.mkdtemp()
    prefixed_svgs = []
    
    for i, (model_name, svg_path) in enumerate(model_configs):
        if not os.path.exists(svg_path):
            print(f"WARNING: File missing for {model_name}: {svg_path}")
            continue
            
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        prefix = f"swe_m{i}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        prefixed_content = prefix_svg_ids(content, prefix)
        
        temp_path = os.path.join(temp_dir, f"{prefix}.svg")
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(prefixed_content)
            
        prefixed_svgs.append((model_name, temp_path))

    if not prefixed_svgs:
        print("No valid SVGs found. Exiting.")
        return

    first_svg = sg.fromfile(prefixed_svgs[0][1])
    def parse_dim(d):
        if isinstance(d, str) and d.endswith('pt'):
            return float(d[:-2])
        elif isinstance(d, str) and d.endswith('px'):
            return float(d[:-2])
        return float(d)
        
    width = parse_dim(first_svg.width)
    height = parse_dim(first_svg.height)
    
    scale = 1.0
    padding_x = 20
    padding_y = 60  # space for title
    
    fig_width = cols * (width * scale + padding_x)
    fig_height = rows * (height * scale + padding_y)
    
    fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")
    
    plots = []
    for i, (model_name, temp_svg_path) in enumerate(prefixed_svgs):
        col = i % cols
        row = i // cols
        
        x = col * (width * scale + padding_x)
        y = row * (height * scale + padding_y)
        
        svg_plot = sg.fromfile(temp_svg_path).getroot()
        svg_plot.moveto(x, y + 40)
        if scale != 1.0:
            svg_plot.scale(scale)
            
        txt = sg.TextElement(x + width/2, y + 30, model_name, size=28, weight="bold", anchor="middle", font="Arial")
        
        plots.append(svg_plot)
        plots.append(txt)
        
    fig.append(plots)
    
    fig.save(args.output)
    print(f"Saved SWE combined SVG to {args.output}")
    
    # Try converting to PDF as well
    pdf_output = args.output.replace('.svg', '.pdf')
    os.system(f"rsvg-convert -f pdf {args.output} -o {pdf_output}")
    print(f"Also converted to PDF: {pdf_output}")
    
    # Cleanup
    for _, temp_svg_path in prefixed_svgs:
        os.remove(temp_svg_path)
    os.rmdir(temp_dir)

if __name__ == '__main__':
    main()
