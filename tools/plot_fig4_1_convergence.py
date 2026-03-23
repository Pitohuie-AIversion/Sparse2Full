import json
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
    print("Generating Figure 4-1 (fig4-4_training_convergence.png)...")
    
    # Update paths to SWE models
    paths = {
        "EDSR (Ours)": "runs/AR-SW-10M-edsr/tensorboard",
        "UNet": "runs/AR-SW-10M-unet/tensorboard",
        "FNO": "runs/AR-SW-10M-fno2d/tensorboard"
    }
    
    fig, ax = plt.subplots()
    
    colors = {"EDSR (Ours)": "tab:red", "UNet": "tab:blue", "FNO": "tab:orange"}
    metric_tags = ['Val/RelL2', 'val/rel_l2', 'rel_l2', 'Val/Loss', 'val_loss', 'val/loss']
    
    for name, path in paths.items():
        steps, values = extract_scalar(path, metric_tags)
        if steps:
            if len(steps) > 500:
                epochs = [s / (steps[-1]/300) for s in steps]
            else:
                epochs = steps
                
            ax.plot(epochs, values, label=name, color=colors.get(name))
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Rel-L2 (Log Scale)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_title("Convergence Comparison on SWE Dataset")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-4_training_convergence.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
