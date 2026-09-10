import os
import glob
import argparse
import svgutils.transform as sg
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = str(PROJECT_ROOT / 'drd_paper_1m')

def main():
    parser = argparse.ArgumentParser(description='Arrange model SVGs into a paper figure.')
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
    
    # Base dimensions (assuming all SVGs are roughly the same size)
    # We load the first one to get exact dimensions
    first_svg = sg.fromfile(model_svgs[0][1])
    # dimensions are often like '657.23pt'
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
    
    rows = (len(model_svgs) + args.cols - 1) // args.cols
    
    fig_width = args.cols * (width * scale + padding_x)
    fig_height = rows * (height * scale + padding_y)
    
    fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")
    
    # White background (using a rect trick by injecting raw svg later if needed, but usually default is fine)
    plots = []
    for i, (model_name, svg_path) in enumerate(model_svgs):
        col = i % args.cols
        row = i // args.cols
        
        x = col * (width * scale + padding_x)
        y = row * (height * scale + padding_y)
        
        svg_plot = sg.fromfile(svg_path).getroot()
        svg_plot.moveto(x, y + 40)
        if scale != 1.0:
            svg_plot.scale(scale)
            
        txt = sg.TextElement(x + width/2, y + 30, model_name, size=28, weight="bold", anchor="middle", font="Arial")
        
        plots.append(svg_plot)
        plots.append(txt)
        
    fig.append(plots)
    
    if args.output is None:
        args.output = os.path.join(args.base_dir, f"paper_figure_{args.filename}")
        
    fig.save(args.output)
    print(f"Saved combined SVG to {args.output}")
    
    # Try converting to PDF as well
    pdf_output = args.output.replace('.svg', '.pdf')
    os.system(f"rsvg-convert -f pdf {args.output} -o {pdf_output}")
    print(f"Also converted to PDF: {pdf_output}")

if __name__ == '__main__':
    main()
