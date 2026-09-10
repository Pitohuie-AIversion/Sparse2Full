
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# List of missing images to generate placeholders for
MISSING_IMAGES = [
    "fig1-1_digital_twin_bottleneck.png",
    "fig1-2_reconstruction_challenge.png",
    "fig1-3_sciml_comparison.png",
    "fig1-4_operator_mismatch.png",
    # fig1-5 is handled by copy/symlink logic
    "fig2-1_error_bound_theory.png",
    "fig2-2_operator_decomposition.png",
    "fig2-3_null_space_illposedness.png",
    "fig2-4_bayesian_consistency.png",
    "fig4-2_power_spectrum.png",
    "fig4-3_pareto_frontier.png",
    "fig5-2_active_sensing_rl.png",
    "fig5-3_foundation_model_pde.png",
    "fig5-4_trustworthy_geometric_sciml.png"
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "thesis_paper/manuscript_5_chapter/images"

def create_placeholder(filename):
    width, height = 800, 600
    # Light gray background
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    
    # Draw border
    d.rectangle([0, 0, width-1, height-1], outline=(100, 100, 100), width=5)
    
    # Draw X
    d.line([0, 0, width, height], fill=(200, 200, 200), width=2)
    d.line([0, height, width, 0], fill=(200, 200, 200), width=2)
    
    # Draw text
    text = f"Placeholder\n{filename}\n(To be replaced)"
    
    # Try to load a font, otherwise use default
    try:
        # Check if we can load a truetype font
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()
        
    # Calculate text position (approximate centering)
    try:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        # Fallback for older Pillow versions
        text_w, text_h = d.textsize(text, font=font)
        
    position = ((width-text_w)/2, (height-text_h)/2)
    
    # Draw text in red
    d.text(position, text, fill=(255, 0, 0), font=font, align="center")
    
    save_path = os.path.join(TARGET_DIR, filename)
    img.save(save_path)
    print(f"Generated placeholder: {save_path}")

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    for img_name in MISSING_IMAGES:
        file_path = os.path.join(TARGET_DIR, img_name)
        if not os.path.exists(file_path):
            create_placeholder(img_name)
        else:
            print(f"Skipping existing: {img_name}")

if __name__ == "__main__":
    main()
