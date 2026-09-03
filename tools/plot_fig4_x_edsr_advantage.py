import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5
})

def main():
    print("Generating Comprehensive EDSR Advantage Figure...")
    
    fig = plt.figure(figsize=(7.16, 5.5), dpi=600)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # (a) Reconstruction Image Comparison
    ax_a = fig.add_subplot(gs[0, 0])
    try:
        img_path = "runs/AR-SW-10M-edsr/test_visualizations/visualizations/predictions/sample_0059_obs_gt_pred_error_t70.png"
        unet_path = "runs/AR-SW-10M-unet/test_visualizations/visualizations/error_analysis/sample_0059_error_analysis.png"
        if os.path.exists(img_path) and os.path.exists(unet_path):
            img = Image.open(img_path)
            unet_img = Image.open(unet_path)
            w, h = img.size
            w2, h2 = w//2, h//2
            gt = img.crop((w2, 0, w, h2))
            edsr_pred = img.crop((0, h2, w2, h))
            
            uw, uh = unet_img.size
            uw2, uh2 = uw//2, uh//2
            unet_pred = unet_img.crop((0, uh2, uw2, uh))
            
            target_w = 200
            target_h = int(200 * (gt.height / gt.width))
            
            gt = gt.resize((target_w, target_h))
            unet_pred = unet_pred.resize((target_w, target_h))
            edsr_pred = edsr_pred.resize((target_w, target_h))
            
            combined = Image.new('RGB', (target_w * 3, target_h))
            combined.paste(gt, (0, 0))
            combined.paste(unet_pred, (target_w, 0))
            combined.paste(edsr_pred, (target_w * 2, 0))
            
            ax_a.imshow(combined)
            ax_a.axis('off')
            ax_a.set_title("(a) Reconstruction (GT | UNet | EDSR)", fontsize=10, loc='left')
            
            # (c) Visual Detail Zoom-in
            ax_c = fig.add_subplot(gs[1, 0])
            zoom_w, zoom_h = int(target_w * 0.3), int(target_h * 0.3)
            center_x, center_y = target_w // 2 + 20, target_h // 2 - 20
            
            zoom_gt = gt.crop((center_x - zoom_w, center_y - zoom_h, center_x + zoom_w, center_y + zoom_h)).resize((target_w, target_h))
            zoom_unet = unet_pred.crop((center_x - zoom_w, center_y - zoom_h, center_x + zoom_w, center_y + zoom_h)).resize((target_w, target_h))
            zoom_edsr = edsr_pred.crop((center_x - zoom_w, center_y - zoom_h, center_x + zoom_w, center_y + zoom_h)).resize((target_w, target_h))
            
            combined_zoom = Image.new('RGB', (target_w * 3, target_h))
            combined_zoom.paste(zoom_gt, (0, 0))
            combined_zoom.paste(zoom_unet, (target_w, 0))
            combined_zoom.paste(zoom_edsr, (target_w * 2, 0))
            
            ax_c.imshow(combined_zoom)
            ax_c.axis('off')
            ax_c.set_title("(c) Detail Amplification (GT | UNet | EDSR)", fontsize=10, loc='left')
        else:
            ax_a.text(0.5, 0.5, "Image not found", ha='center', va='center')
            ax_c = fig.add_subplot(gs[1, 0])
            ax_c.text(0.5, 0.5, "Image not found", ha='center', va='center')
            ax_c.axis('off')
    except Exception as e:
        ax_a.text(0.5, 0.5, "Error loading image", ha='center', va='center')
        ax_a.axis('off')
        
    # (b) Pareto Frontier (PSNR vs Params)
    ax_b = fig.add_subplot(gs[0, 1])
    models = ["EDSR (Ours)", "NAFNet", "ResNetLite", "UNO", "SwinUNet", "SegFormer", "MLP-Model"]
    params = [1.22, 8.15, 9.99, 28.05, 3.52, 23.21, 0.01]
    psnr = [71.05, 52.19, 46.52, 48.77, 31.96, 32.36, 39.52]
    
    ax_b.scatter(params, psnr, s=50, c='tab:blue', alpha=0.7, edgecolors='k')
    idx_ours = models.index("EDSR (Ours)")
    ax_b.scatter(params[idx_ours], psnr[idx_ours], s=100, c='tab:red', marker='*', edgecolors='k', label='Ours')
    
    for i, txt in enumerate(models):
        if txt == "EDSR (Ours)":
            ax_b.annotate(txt, (params[i], psnr[i]), xytext=(5, -10), textcoords='offset points', fontweight='bold', color='tab:red', fontsize=8)
        else:
            ax_b.annotate(txt, (params[i], psnr[i]), xytext=(5, 5), textcoords='offset points', fontsize=7)
            
    ax_b.set_xlabel("Parameters (M) - Log Scale")
    ax_b.set_ylabel("PSNR (dB)")
    ax_b.set_xscale('log')
    ax_b.set_title("(b) Pareto Frontier (PSNR vs Params)", fontsize=10, loc='left')
    ax_b.grid(True, which="both", ls="-", alpha=0.2)
        
    # (d) Ablation / Metric Bar Chart
    ax_d = fig.add_subplot(gs[1, 1])
    bar_models = ["Bicubic", "MLP", "UNet", "EDSR (Ours)"]
    bar_rel_l2 = [0.1480, 0.0182, 0.0327, 0.0023] 
    
    x_pos = np.arange(len(bar_models))
    bars = ax_d.bar(x_pos, bar_rel_l2, color=['gray', 'tab:purple', 'tab:blue', 'tab:red'], alpha=0.8)
    
    ax_d.set_yscale('log')
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(bar_models, rotation=15, ha='right')
    ax_d.set_ylabel("Rel-L2 Error (Log Scale)")
    ax_d.set_title("(d) Reconstruction Error Comparison", fontsize=10, loc='left')
    
    ax_d.text(3, 0.0025, "* p<0.05", ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
    
    output_path = os.path.join(OUTPUT_DIR, "fig4_edsr_advantage.png")
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
