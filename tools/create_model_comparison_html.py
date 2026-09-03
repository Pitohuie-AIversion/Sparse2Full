import os
import glob
import argparse

def generate_html_grid(base_dir, filename, output_html):
    search_pattern = os.path.join(base_dir, '*', 'test_visualizations', 'visualizations', 'predictions', filename)
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
    
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  body { font-family: Arial, sans-serif; background: #fff; margin: 20px; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }",
        "  .figure-container { border: 1px solid #ddd; padding: 10px; text-align: center; background: #fafafa; border-radius: 8px; page-break-inside: avoid; }",
        "  .model-title { font-size: 24px; font-weight: bold; margin-bottom: 15px; color: #333; }",
        "  img { max-width: 100%; height: auto; }",
        "  @media print {",
        "    .figure-container { border: none; background: #fff; }",
        "    body { margin: 0; }",
        "  }",
        "</style>",
        "</head>",
        "<body>",
        f"<h2>Model Comparison: {filename}</h2>",
        "<div class='grid'>"
    ]
    
    for model_name, svg_path in model_svgs:
        # Create a relative path for the HTML file if they are in the same or sub-directory
        rel_path = os.path.relpath(svg_path, os.path.dirname(output_html))
        html_content.append("  <div class='figure-container'>")
        html_content.append(f"    <div class='model-title'>{model_name}</div>")
        html_content.append(f"    <img src='{rel_path}' alt='{model_name}'>")
        html_content.append("  </div>")
        
    html_content.append("</div>")
    html_content.append("</body></html>")
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
        
    print(f"Generated HTML figure at: {output_html}")
    print(f"You can open this HTML file in your browser and print to PDF.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a comparison HTML figure for models.')
    parser.add_argument('--base_dir', type=str, default='/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/drd_paper_1m',
                        help='Base directory containing model folders')
    parser.add_argument('--filename', type=str, default='sample_0059_obs_gt_pred_error_t70_1.svg',
                        help='Target SVG filename to compare')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path')
                        
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(args.base_dir, f"comparison_{args.filename.replace('.svg', '.html')}")
        
    generate_html_grid(args.base_dir, args.filename, args.output)
