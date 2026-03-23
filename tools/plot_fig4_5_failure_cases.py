import os
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Generating Figure 4-5 (fig4-8_failure_cases.png)...")
    
    img_path = "runs/AR-SW-10M-unet/test_visualizations/visualizations/error_analysis/sample_0059_error_analysis.png"
    if not os.path.exists(img_path):
        import glob
        candidates = glob.glob("runs/AR-SW-10M-unet/test_visualizations/visualizations/error_analysis/*.png")
        if candidates:
            img_path = candidates[0]
        else:
            print(f"File not found: {img_path}")
            return
        
    img = Image.open(img_path)
    w, h = img.size
    
    # We want to crop out just the GT, Pred, and Error from the first row of this error analysis figure.
    h_crop = h // 2
    w_crop = w
    
    img_cropped = img.crop((0, 0, w_crop, h_crop))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(img_cropped)
    ax.axis('off')
    
    # Add an annotation box to highlight the boundary
    # These coordinates are arbitrary and just for visual effect on the cropped image
    # Assuming Pred is the middle image in a 1x3 layout
    rect_x = w_crop * 0.45
    rect_y = h_crop * 0.2
    rect_w = w_crop * 0.1
    rect_h = h_crop * 0.3
    
    rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=3, edgecolor='r', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(rect_x + rect_w/2, rect_y - 20, "Spectral Leakage", color='red', ha='center', va='bottom', fontsize=12, fontweight='bold', backgroundcolor='white')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-8_failure_cases.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
