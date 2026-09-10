import os
import glob
import re
import argparse
import svgutils.transform as sg
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = str(PROJECT_ROOT / 'drd_paper_1m')

def prefix_svg_ids(svg_content, prefix):
    """
    Prefixes all IDs in the SVG content to prevent clashes when merging multiple SVGs.
    """
    # Find all id="something"
    id_pattern = re.compile(r'id="([^"]+)"')
    
    # We need to collect all IDs so we can also replace url(#...) and href="#..."
    found_ids = set(id_pattern.findall(svg_content))
    
    # Replace id="something" with id="prefix_something"
    svg_content = id_pattern.sub(lambda m: f'id="{prefix}_{m.group(1)}"', svg_content)
    
    # Replace url(#something) with url(#prefix_something)
    for old_id in found_ids:
        # We need to be careful with exact matches
        svg_content = svg_content.replace(f'url(#{old_id})', f'url(#{prefix}_{old_id})')
        svg_content = svg_content.replace(f'href="#{old_id}"', f'href="#{prefix}_{old_id}"')
        svg_content = svg_content.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{prefix}_{old_id}"')
        
    return svg_content

def main():
    parser = argparse.ArgumentParser(description='Arrange model SVGs into a paper figure without ID clashes.')
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Base directory containing model folders')
    parser.add_argument('--filename', type=str, default='sample_0059_obs_gt_pred_error_t70_1.svg',
                        help='Target SVG filename to compare')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the grid')
    parser.add_argument('--output', type=str, default=None, help='Output combined SVG path')
    
    args = parser.parse_args()
    
    search_pattern = os.path.join(args.base_dir, '*', 'test_visualizations', 'visualizations', 'predictions', args.filename)
    svg_files = glob.glob(search_pattern)
    
    if not svg_files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(svg_files)} files.")
    
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
    
    # Create temp directory for prefixed SVGs
    temp_dir = tempfile.mkdtemp()
    prefixed_svgs = []
    
    for i, (model_name, svg_path) in enumerate(model_svgs):
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create a unique prefix using index to ensure it's completely safe
        prefix = f"m{i}_{model_name.replace('-', '_')}"
        prefixed_content = prefix_svg_ids(content, prefix)
        
        temp_path = os.path.join(temp_dir, f"{prefix}.svg")
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(prefixed_content)
            
        prefixed_svgs.append((model_name, temp_path))

    # Base dimensions
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
    
    rows = (len(prefixed_svgs) + args.cols - 1) // args.cols
    
    fig_width = args.cols * (width * scale + padding_x)
    fig_height = rows * (height * scale + padding_y)
    
    fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")
    
    plots = []
    for i, (model_name, temp_svg_path) in enumerate(prefixed_svgs):
        col = i % args.cols
        row = i // args.cols
        
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
    
    if args.output is None:
        args.output = os.path.join(args.base_dir, f"paper_figure_fixed_{args.filename}")
        
    fig.save(args.output)
    print(f"Saved combined SVG to {args.output}")
    
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
