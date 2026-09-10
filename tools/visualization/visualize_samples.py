
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os
import random
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def save_animation(fig, ims, output_path):
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=10, bitrate=1800)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved to {output_path}")

def visualize_drd(h5_path, output_dir, stride=10, num_samples=5):
    print(f"Processing DRD from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        # Filter keys that look like sample indices (digits)
        sample_keys = [k for k in keys if k.isdigit()]
        if not sample_keys:
             # Fallback if keys are not just digits or different structure
             sample_keys = keys
        
        total_samples = len(sample_keys)
        print(f"Total samples found: {total_samples}")
        
        selected_keys = random.sample(sample_keys, min(num_samples, total_samples))
        
        for key in selected_keys:
            data = np.array(f[f"{key}/data"], dtype="f") # Shape: [T, H, W, C]
            # Apply time stride
            data = data[::stride]
            
            # DRD has 2 channels
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ims = []
            
            for i in range(data.shape[0]):
                im1 = ax[0].imshow(data[i, ..., 0].squeeze(), animated=True, vmin=data[..., 0].min(), vmax=data[..., 0].max())
                im2 = ax[1].imshow(data[i, ..., 1].squeeze(), animated=True, vmin=data[..., 1].min(), vmax=data[..., 1].max())
                
                # Add title or text if needed
                if i == 0:
                    ax[0].imshow(data[0, ..., 0].squeeze(), vmin=data[..., 0].min(), vmax=data[..., 0].max())
                    ax[1].imshow(data[0, ..., 1].squeeze(), vmin=data[..., 1].min(), vmax=data[..., 1].max())
                
                ims.append([im1, im2])
            
            ax[0].set_title(f"Sample {key} - Channel U")
            ax[1].set_title(f"Sample {key} - Channel V")
            
            output_path = os.path.join(output_dir, f"drd_sample_{key}_stride{stride}.gif")
            save_animation(fig, ims, output_path)

def visualize_swe(h5_path, output_dir, stride=10, num_samples=5):
    print(f"Processing SWE from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        sample_keys = [k for k in keys if k.isdigit()]
        if not sample_keys:
             sample_keys = keys
             
        total_samples = len(sample_keys)
        print(f"Total samples found: {total_samples}")
        
        selected_keys = random.sample(sample_keys, min(num_samples, total_samples))
        
        for key in selected_keys:
            data = np.array(f[f"{key}/data"], dtype="f") # Shape: [T, H, W, 1] usually
            data = data[::stride]
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ims = []
            
            # Check if channel dim exists
            if data.ndim == 4:
                plot_data = data[..., 0]
            else:
                plot_data = data
                
            vmin, vmax = plot_data.min(), plot_data.max()
            
            for i in range(plot_data.shape[0]):
                im = ax.imshow(plot_data[i].squeeze(), animated=True, vmin=vmin, vmax=vmax)
                if i == 0:
                    ax.imshow(plot_data[0].squeeze(), vmin=vmin, vmax=vmax)
                ims.append([im])
            
            ax.set_title(f"SWE Sample {key}")
            output_path = os.path.join(output_dir, f"swe_sample_{key}_stride{stride}.gif")
            save_animation(fig, ims, output_path)

if __name__ == "__main__":
    base_dir = PROJECT_ROOT / "data/2D"
    output_dir = PROJECT_ROOT / "paper_package/figs/vis_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths based on user request and file system exploration
    drd_path = os.path.join(base_dir, "diffusion-reaction", "2D_diff-react_NA_NA.h5")
    swe_path = os.path.join(base_dir, "shallow-water", "2D_rdb_NA_NA.h5")
    
    # Run visualizations
    if os.path.exists(drd_path):
        visualize_drd(drd_path, output_dir, stride=10, num_samples=5)
    else:
        print(f"File not found: {drd_path}")
        
    if os.path.exists(swe_path):
        visualize_swe(swe_path, output_dir, stride=10, num_samples=5)
    else:
        print(f"File not found: {swe_path}")

    print(f"All visualizations saved to {output_dir}")
