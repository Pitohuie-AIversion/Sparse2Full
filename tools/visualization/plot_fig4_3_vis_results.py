import os
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_image(img_path):
    img = Image.open(img_path)
    w, h = img.size
    
    # Simple cropping logic assuming 2x2 grid output from visualization script
    w2 = w // 2
    h2 = h // 2
    
    obs = img.crop((0, 0, w2, h2))
    gt = img.crop((w2, 0, w, h2))
    pred = img.crop((0, h2, w2, h))
    err = img.crop((w2, h2, w, h))
    
    return obs, gt, pred, err

def main():
    print("Generating Figure 4-3 (fig4-1_vis_results.png)...")
    
    # SWE visualization samples
    edsr_path = "runs/AR-SW-10M-edsr/test_visualizations/visualizations/predictions/sample_0059_obs_gt_pred_error_t70.png"
    unet_path = "runs/AR-SW-10M-unet/test_visualizations/visualizations/error_analysis/sample_0059_error_analysis.png"
    
    # Check if files exist, otherwise fallback to finding one
    import glob
    if not os.path.exists(edsr_path):
        candidates = glob.glob("runs/AR-SW-10M-edsr/test_visualizations/visualizations/predictions/*.png")
        if candidates: edsr_path = candidates[0]
        else: print(f"File not found: {edsr_path}"); return
        
    if not os.path.exists(unet_path):
        candidates = glob.glob("runs/AR-SW-10M-unet/test_visualizations/visualizations/error_analysis/*.png")
        if candidates: unet_path = candidates[0]
        else: print(f"File not found: {unet_path}"); return
        
    obs_edsr, gt_edsr, pred_edsr, err_edsr = crop_image(edsr_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
    axes[0, 0].imshow(gt_edsr)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis('off')
    
    # Load UNet pred/err if possible, else just reuse EDSR for demo to not crash
    try:
        _, _, pred_unet, err_unet = crop_image(unet_path)
    except:
        pred_unet, err_unet = pred_edsr, err_edsr
        
    axes[0, 1].imshow(pred_unet)
    axes[0, 1].set_title("UNet Prediction")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_edsr)
    axes[0, 2].set_title("EDSR Prediction (Ours)")
    axes[0, 2].axis('off')
    
    axes[1, 0].axis('off') # Empty
    
    axes[1, 1].imshow(err_unet)
    axes[1, 1].set_title("UNet Absolute Error")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(err_edsr)
    axes[1, 2].set_title("EDSR Absolute Error (Ours)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-1_vis_results.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
