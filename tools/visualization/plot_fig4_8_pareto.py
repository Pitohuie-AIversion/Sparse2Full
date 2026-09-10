import os
import matplotlib.pyplot as plt
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
    "figure.figsize": (6, 5),
    "lines.linewidth": 2
})

def main():
    print("Generating Figure 4-8 (fig4-3_pareto_frontier.png)...")
    
    # Real data extracted from runs_drd models (DRD dataset SRx4)
    # Extracted from corresponding model_resources.json and test_results.json
    models = ["EDSR (Ours)", "UNet", "FNO", "ResNetLite", "SegFormer", "UNetPlusPlus"]
    # FLOPs (G)
    flops = [19.95, 161.84, 15.3, 163.62, 88.62, 180.5]
    # Rel-L2
    rel_l2 = [0.0180, 0.0210, 0.0351, 0.0376, 0.1008, 0.0195]
    
    fig, ax = plt.subplots()
    
    # Scatter plot
    ax.scatter(flops, rel_l2, s=100, c='tab:blue', alpha=0.7, edgecolors='k')
    
    # Highlight Ours
    idx_ours = models.index("EDSR (Ours)")
    ax.scatter(flops[idx_ours], rel_l2[idx_ours], s=150, c='tab:red', marker='*', edgecolors='k', label='Ours')
    
    # Annotate points
    for i, model in enumerate(models):
        if model == "EDSR (Ours)":
            ax.annotate(model, (flops[i], rel_l2[i]), xytext=(5, -10), textcoords='offset points', fontweight='bold', color='tab:red')
        else:
            ax.annotate(model, (flops[i], rel_l2[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
    # Draw Pareto frontier
    sorted_indices = np.argsort(flops)
    pareto_flops = []
    pareto_err = []
    min_err = float('inf')
    
    for idx in sorted_indices:
        if rel_l2[idx] < min_err:
            pareto_flops.append(flops[idx])
            pareto_err.append(rel_l2[idx])
            min_err = rel_l2[idx]
            
    ax.plot(pareto_flops, pareto_err, 'k--', alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel("Computational Cost (GFLOPs) - Log Scale")
    ax.set_ylabel("Relative L2 Error - Log Scale")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Resource-Accuracy Trade-off (Pareto Frontier)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-3_pareto_frontier.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
