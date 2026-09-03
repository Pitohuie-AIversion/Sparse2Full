import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5
})

def main():
    print("Generating Figure: Latency-Error Trade-off for 1M Parameter Budget...")
    
    # Data from Table 4-3
    models = ["EDSR (Ours)", "ConvUNetLite", "UNet", "StableFNO2d", "NAFNet"]
    # We plot Latency on x-axis and Rel-L2 on y-axis
    latency = [20.25, 0.77, 1.11, 5.00, 15.91]
    rel_l2 = [0.0046, 0.0082, 0.0327, 0.0351, 0.0072]
    # We can use Params to scale the bubble size
    params = [0.93, 1.00, 0.92, 1.19, 8.15]
    
    # Scale sizes for better visibility, NAFNet will be notably larger
    sizes = [p * 100 for p in params]
    
    fig, ax = plt.subplots(figsize=(7.16, 5), dpi=600)
    
    # Define colors
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    markers = ['*', 'o', 's', '^', 'D']
    
    for i, model in enumerate(models):
        if model == "EDSR (Ours)":
            ax.scatter(latency[i], rel_l2[i], s=sizes[i]*1.5, c=colors[i], marker=markers[i], edgecolors='k', label=f"{model} ({params[i]:.2f}M)", zorder=5)
        else:
            ax.scatter(latency[i], rel_l2[i], s=sizes[i], c=colors[i], marker=markers[i], edgecolors='k', label=f"{model} ({params[i]:.2f}M)", alpha=0.8, zorder=4)
            
    # Add annotations
    # EDSR
    ax.annotate("Best Accuracy\nHigh Latency", (latency[0], rel_l2[0]), xytext=(-10, 15), textcoords='offset points', ha='center', va='bottom', fontsize=10, color='tab:red')
    # ConvUNetLite
    ax.annotate("Fastest\nGood Accuracy", (latency[1], rel_l2[1]), xytext=(15, -15), textcoords='offset points', ha='left', va='top', fontsize=10, color='tab:green')
    # UNet
    ax.annotate("Fast\nFair Accuracy", (latency[2], rel_l2[2]), xytext=(15, 10), textcoords='offset points', ha='left', va='bottom', fontsize=10, color='tab:blue')
    # StableFNO2d
    ax.annotate("Moderate Speed\nFair Accuracy", (latency[3], rel_l2[3]), xytext=(15, 10), textcoords='offset points', ha='left', va='bottom', fontsize=10, color='tab:orange')
    # NAFNet
    ax.annotate("Budget Exceeded\n(8.15M)", (latency[4], rel_l2[4]), xytext=(-15, -20), textcoords='offset points', ha='right', va='top', fontsize=10, color='tab:purple')

    # Draw Pareto frontier line (EDSR -> ConvUNetLite)
    # The optimal trade-off between speed and accuracy
    pareto_x = [latency[1], latency[0]]
    pareto_y = [rel_l2[1], rel_l2[0]]
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.4, zorder=1, label='Pareto Frontier')
    
    # Highlight the 1M parameter budget zone implicitly
    
    ax.set_xlabel("Inference Latency (ms) - Log Scale")
    ax.set_ylabel("Relative L2 Error - Log Scale")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(title="Models (Circle Size $\propto$ Params)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Latency vs Accuracy Trade-off under ~1M Parameter Budget", pad=15)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4_latency_accuracy_tradeoff.png")
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
