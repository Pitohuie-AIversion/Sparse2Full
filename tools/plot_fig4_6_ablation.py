import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4),
    "lines.linewidth": 2
})

def extract_scalar(log_dir, tags):
    """Extract scalar data from the largest event file in log_dir."""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None

    event_file = max(event_files, key=os.path.getsize)
    print(f"Reading {event_file}...")
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    target_tag = None
    for t in tags:
        if t in available_tags:
            target_tag = t
            break
            
    if not target_tag:
        print(f"None of {tags} found in {available_tags}")
        return None, None
        
    events = ea.Scalars(target_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def main():
    print("Generating Figure 4-6 (fig4-6_ablation_curves.png)...")
    
    # Use runs_3loss_ablation_unet for UNet ablation results
    paths = {
        "MSE Only ($\mathcal{L}_{rec}$)": "runs_3loss_ablation_unet/A0_Baseline/tensorboard",
        "Full Loss ($\mathcal{L}_{rec} + \mathcal{L}_{spec} + \mathcal{L}_{dc}$)": "runs_3loss_ablation_unet/A3_Full/tensorboard"
    }
    
    fig, ax = plt.subplots()
    
    colors = {
        "MSE Only ($\mathcal{L}_{rec}$)": "tab:blue", 
        "Full Loss ($\mathcal{L}_{rec} + \mathcal{L}_{spec} + \mathcal{L}_{dc}$)": "tab:green"
    }
    linestyles = {
        "MSE Only ($\mathcal{L}_{rec}$)": "--", 
        "Full Loss ($\mathcal{L}_{rec} + \mathcal{L}_{spec} + \mathcal{L}_{dc}$)": "-"
    }
    
    metric_tags = ['Val/RelL2', 'val/rel_l2', 'rel_l2', 'Val/Loss', 'val_loss', 'val/loss']
    
    last_mse_val = None
    last_full_val = None
    max_epoch = 0
    
    for name, path in paths.items():
        steps, values = extract_scalar(path, metric_tags)
        if steps:
            # We want epochs on x-axis.
            if len(steps) > 500:
                epochs = [s / (steps[-1]/100) for s in steps] # Assuming ~100 epochs for ablation
            else:
                epochs = steps
                
            ax.plot(epochs, values, label=name, color=colors.get(name), linestyle=linestyles.get(name))
            
            if "MSE" in name:
                last_mse_val = values[-1]
            elif "Full" in name:
                last_full_val = values[-1]
                max_epoch = epochs[-1]
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Rel-L2")
    ax.set_title("Ablation Study: Impact of Physical Consistency Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if last_mse_val and last_full_val and last_mse_val > 0:
        gain = (last_mse_val - last_full_val) / last_mse_val * 100
        ax.annotate(f"Performance Gain: -{gain:.1f}%", xy=(max_epoch, last_full_val), xytext=(max_epoch*0.6, last_full_val*2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-6_ablation_curves.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
