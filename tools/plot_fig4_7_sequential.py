import os
import matplotlib.pyplot as plt
import json
import numpy as np

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

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    print("Generating Figure 4-7 (fig4-5_sequential_evolution.png)...")
    
    stage2_path = "runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/training_history.json"
    stage3_path = "runs_drd_paper/AR-DR2D-Stage3-VideoSwin-SRx4-JointFineTune-model_unknown-s2025-20260226/training_history.json"
    
    d2 = load_json(stage2_path)
    d3 = load_json(stage3_path)
    
    if not d2 or not d3:
        print("Data not found, please check paths.")
        return
        
    m2 = d2.get('val_metrics', [])
    m3 = d3.get('val_metrics', [])
    
    rel_l2_2 = [m.get('rel_l2', np.nan) for m in m2]
    rel_l2_3 = [m.get('rel_l2', np.nan) for m in m3]
    
    # We might not have frmse_high in this JSON. Let's check if we have it, else use spectral_loss or fallback.
    high_err_2 = [m.get('frmse_high', m.get('spectral_loss', np.nan)) for m in m2]
    high_err_3 = [m.get('frmse_high', m.get('spectral_loss', np.nan)) for m in m3]
    
    epochs_2 = np.arange(1, len(rel_l2_2) + 1)
    epochs_3 = np.arange(len(rel_l2_2) + 1, len(rel_l2_2) + len(rel_l2_3) + 1)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Rel-L2 (Global Error)', color=color)
    ax1.plot(epochs_2, rel_l2_2, color=color, label='Rel-L2 (Stage 2)')
    ax1.plot(epochs_3, rel_l2_3, color=color, label='Rel-L2 (Stage 3)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('High-Freq Error Indicator', color=color)
    ax2.plot(epochs_2, high_err_2, color=color, linestyle='--', label='High-Freq Err (Stage 2)')
    ax2.plot(epochs_3, high_err_3, color=color, linestyle='--', label='High-Freq Err (Stage 3)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add vertical line for Stage transition
    transition_epoch = len(rel_l2_2)
    plt.axvline(x=transition_epoch, color='gray', linestyle=':', label='Unfreeze All Layers')
    ax1.text(transition_epoch + 2, max(rel_l2_2)*0.8, "Stage 3: Joint Fine-tuning", fontsize=9)
    ax1.text(transition_epoch / 2, max(rel_l2_2)*0.8, "Stage 2: Spatial Freeze", fontsize=9, ha='center')
    
    plt.title("Evolution of Global vs. Local Error during Sequential Training")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-5_sequential_evolution.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
