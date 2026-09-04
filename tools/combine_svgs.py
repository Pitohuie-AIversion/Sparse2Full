import os
import glob
import svgutils.transform as sg
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Search for the SVG files
base_dir = PROJECT_ROOT / 'drd_paper_1m'
filename = 'sample_0059_obs_gt_pred_error_t70_1.svg'

search_pattern = os.path.join(base_dir, '*', 'test_visualizations', 'visualizations', 'predictions', filename)
svg_files = glob.glob(search_pattern)

print(f"Found {len(svg_files)} files.")

# Extract model names
model_svgs = []
for f in svg_files:
    # Example path: .../drd_paper_1m/AR-DR2D-UNet-SRx4-1M-100ep/...
    parts = f.split('/')
    model_dir = parts[-5]  # The directory like 'AR-DR2D-UNet-SRx4-1M-100ep'
    model_name = model_dir.split('-')[2]  # Extracts 'UNet'
    model_svgs.append((model_name, f))

# Sort to have a consistent order
model_svgs.sort()

# SVG dimensions
width = 657
height = 568

# Grid layout: 4 columns, 2 rows
cols = 4
rows = (len(model_svgs) + cols - 1) // cols

scale = 1.0
padding_x = 20
padding_y = 60

fig_width = cols * (width * scale + padding_x)
fig_height = rows * (height * scale + padding_y)

# Create new SVG figure
fig = sg.SVGFigure(f"{fig_width}pt", f"{fig_height}pt")

plots = []
for i, (model_name, svg_path) in enumerate(model_svgs):
    col = i % cols
    row = i // cols
    
    x = col * (width * scale + padding_x)
    y = row * (height * scale + padding_y)
    
    # Load the svg
    svg_plot = sg.fromfile(svg_path).getroot()
    svg_plot.moveto(x, y + 30)
    if scale != 1.0:
        svg_plot.scale(scale)

    
    # Add a text label
    txt = sg.TextElement(x + width/2, y + 25, model_name, size=24, weight="bold", anchor="middle")
    
    plots.append(svg_plot)
    plots.append(txt)

fig.append(plots)

output_path = os.path.join(base_dir, 'combined_sample_0059.svg')
fig.save(output_path)
print(f"Saved combined SVG to {output_path}")
