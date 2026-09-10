
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# Set style for academic plot
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
        return None

    # Use the largest event file
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
        return None
        
    events = ea.Scalars(target_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_real_curve():
    # Define one experiment to test
    exp_path = "runs/AR-SW-10M-pconvunet"
    exp_name = "PConvUNet (Real Data)"
    
    # Possible tags for validation loss/metric
    loss_tags = ['Val/Loss', 'val_loss', 'val/loss']
    metric_tags = ['Val/RelL2', 'val/rel_l2', 'rel_l2']
    
    steps, loss_values = extract_scalar(exp_path, loss_tags) or ([], [])
    steps_m, metric_values = extract_scalar(exp_path, metric_tags) or ([], [])
    
    if not steps:
        print("Failed to extract data.")
        return

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch / Step')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(steps, loss_values, color=color, label='Val Loss', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # If we have metric data, plot it on twin axis or same
    if metric_values:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Rel-L2 Error', color=color)
        ax2.plot(steps_m, metric_values, color=color, label='Rel-L2', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Real Training Curve: {exp_name}")
    plt.tight_layout()
    
    output_path = "thesis_paper/manuscript_5_chapter/images/test_real_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_real_curve()
