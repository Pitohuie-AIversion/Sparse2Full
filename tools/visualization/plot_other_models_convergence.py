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
    "figure.figsize": (8, 6),
    "lines.linewidth": 2
})

def extract_scalar(log_dir, tags):
    """Extract scalar data from the largest event file in log_dir."""
    if not os.path.exists(log_dir):
        return None, None
        
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
    
    available_tags = ea.Tags().get('scalars', [])
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
    print("Generating training curves for other models...")
    
    paths = {
        "EDSR (Ours)": "runs/AR-SW-10M-edsr/tensorboard",
        "NAFNet": "runs/AR-SW-10M-nafnet/tensorboard",
        "ResNetLite": "runs/AR-SW-10M-resnetlite/tensorboard",
        "UNO": "runs/AR-SW-10M-uno/tensorboard",
        "SwinUNet": "runs/AR-SW-10M-swinunet/tensorboard",
        "SegFormer": "runs/AR-SW-10M-segformer/tensorboard",
        "MLP-Model": "runs/AR-SW-10M-mlpmodel/tensorboard"
    }
    
    fig, ax = plt.subplots()
    
    colors = {
        "EDSR (Ours)": "tab:red", 
        "NAFNet": "tab:blue", 
        "ResNetLite": "tab:green",
        "UNO": "tab:orange",
        "SwinUNet": "tab:purple",
        "SegFormer": "tab:brown",
        "MLP-Model": "tab:pink"
    }
    metric_tags = ['Val/RelL2', 'val/rel_l2', 'rel_l2', 'Val/Loss', 'val_loss', 'val/loss']
    
    for name, path in paths.items():
        steps, values = extract_scalar(path, metric_tags)
        if steps:
            if len(steps) > 500:
                epochs = [s / (steps[-1]/300) for s in steps]
            else:
                epochs = steps
                
            ax.plot(epochs, values, label=name, color=colors.get(name))
        else:
            print(f"Could not load data for {name}")
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Rel-L2 (Log Scale)")
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_title("Convergence Comparison on SWE Dataset (Extended Models)")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-4_training_convergence_extended.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
