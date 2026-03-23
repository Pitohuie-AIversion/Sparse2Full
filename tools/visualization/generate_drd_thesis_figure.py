
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def generate_drd_figure():
    # Configuration
    # Prefer the file in data/DR2D/ first as it is likely the correct one
    data_path = Path("data/DR2D/2D_diff-react_NA_NA.h5")
    if not data_path.exists():
        data_path = Path("data/2D_diff-react_NA_NA.h5")
        
    output_dir = Path("thesis_paper/manuscript_5_chapter/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not data_path.exists():
        # Try relative to project root
        project_root = Path("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full")
        data_path = project_root / "data/2D_diff-react_NA_NA.h5"
        if not data_path.exists():
            data_path = project_root / "data/DR2D/2D_diff-react_NA_NA.h5"
            if not data_path.exists():
                print(f"Error: Data file not found at {data_path}")
                return

    print(f"Loading data from {data_path}...")
    
    try:
        with h5py.File(data_path, "r") as f:
            # Pick a sample. Let's try seed "0000"
            seed = "0000"
            if seed not in f:
                seed = list(f.keys())[0]
                print(f"Seed '0000' not found, using '{seed}' instead.")
            
            # Data shape: [T, H, W, C] -> usually [101, 128, 128, 2] for Reaction-Diffusion
            data = f[f"{seed}/data"][:]
            print(f"Data shape: {data.shape}")
            
            # Select channel: 0 for u, 1 for v. Let's visualize 'u' (channel 0)
            channel_idx = 0
            if data.ndim == 4:
                data = data[..., channel_idx] # [T, H, W]
            
            # Select time steps
            time_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            frames = [data[t] for t in time_steps]
            
            # Determine global vmin/vmax for consistent color mapping
            vmin = np.percentile(data, 2)
            vmax = np.percentile(data, 98)
            
            # Setup Plot: 2 rows, 5 columns
            fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
            
            # Flatten axes for easy iteration
            axes_flat = axes.flatten()
            
            images = []
            for i, ax in enumerate(axes_flat):
                t = time_steps[i]
                frame = frames[i]
                
                # Plot heatmap
                im = ax.imshow(frame, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                
                # Title
                ax.set_title(f"t = {t}", fontsize=14, pad=5)
                
                # Clean style: remove ticks/labels but keep border
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Ensure clear black border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
                
                images.append(im)

            # Add shared colorbar on the right
            cbar = fig.colorbar(images[-1], ax=axes, orientation='vertical', fraction=0.02, pad=0.02, aspect=30)
            cbar.ax.tick_params(labelsize=12)
            cbar.outline.set_linewidth(1.0)
            # Label for colorbar (optional, maybe just 'u')
            # cbar.set_label('u', rotation=0, labelpad=10)
            
            # Add subtle seed label in top-left corner
            fig.text(0.01, 0.98, f"Seed: {seed}", fontsize=10, color='gray', ha='left', va='top', fontfamily='monospace')
            
            # Save
            output_path_png = output_dir / "drd_evolution_sample.png"
            output_path_pdf = output_dir / "drd_evolution_sample.pdf"
            output_path_svg = output_dir / "drd_evolution_sample.svg"
            
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_path_pdf, bbox_inches='tight')
            plt.savefig(output_path_svg, bbox_inches='tight')
            
            print(f"Figure saved to:\n{output_path_png}\n{output_path_pdf}\n{output_path_svg}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_drd_figure()
