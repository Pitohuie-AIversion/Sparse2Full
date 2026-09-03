import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# 1. Data gathered from ./drd_paper_1m logs
models = ["EDSR", "ConvUNetLite", "UNet", "StableFNO2d", "NAFNet"]
latencies = [20.247, 0.768, 1.107, 4.997, 15.911]
rel_l2s = [0.00455, 0.00820, 0.03273, 0.03506, 0.00719]
params = [0.934, 1.002, 0.915, 1.186, 8.147]

# 2. Matplotlib configuration for IEEE Transactions standard
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})

fig, ax = plt.subplots(figsize=(5, 3.8))

colors = sns.color_palette("Set2", len(models))
markers = ['o', 'o', 'o', 'o', 'o']

# 3. Plot points
scatter_plots = []
for i in range(len(models)):
    # Calculate bubble size based on Params (M)
    # Base size 30, scaled by params.
    # Since NAFNet is 8M, it will be noticeably larger.
    size = params[i] * 20 + 40 
    
    sc = ax.scatter(latencies[i], rel_l2s[i], 
               s=size, 
               color=colors[i], 
               marker=markers[i], 
               alpha=0.85,
               edgecolors='k',
               linewidth=0.6,
               zorder=3)
    scatter_plots.append(sc)
    
    # Text annotation offsets
    offset_x = 0.5
    offset_y = 0.001
    
    if models[i] == "EDSR":
        offset_x, offset_y = -0.5, -0.0025
        ha = 'right'
    elif models[i] == "NAFNet":
        offset_x, offset_y = -0.8, -0.0025
        ha = 'right'
    elif models[i] == "StableFNO2d":
        offset_x, offset_y = 0.5, 0.001
        ha = 'left'
    elif models[i] == "UNet":
        offset_x, offset_y = 0.5, 0.001
        ha = 'left'
    elif models[i] == "ConvUNetLite":
        offset_x, offset_y = 0.5, 0.001
        ha = 'left'
        
    ax.text(latencies[i] + offset_x, rel_l2s[i] + offset_y, models[i], 
            fontsize=9, ha=ha, zorder=4)

# 4. Draw Pareto Frontier
# Sorted by latency: ConvUNetLite -> NAFNet -> EDSR
pareto_latencies = [latencies[1], latencies[4], latencies[0]]
pareto_errors = [rel_l2s[1], rel_l2s[4], rel_l2s[0]]
ax.plot(pareto_latencies, pareto_errors, '--', color='gray', zorder=2, label='Pareto Frontier', alpha=0.7)

# 5. Formatting
ax.set_xlabel('Inference Latency (ms)')
ax.set_ylabel('Relative L2 Error (Rel-L2)')

ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

# Add custom legends
# Legend 1: Pareto frontier
handles, labels = ax.get_legend_handles_labels()
# Add dummy scatter for Size legend
legend_elements = [
    Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='Pareto Frontier'),
    plt.scatter([], [], s=1*20+40, color='none', edgecolors='k', label='~1M Params'),
    plt.scatter([], [], s=8*20+40, color='none', edgecolors='k', label='~8M Params')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, edgecolor='k')

# Limits to give text some room
ax.set_xlim(-2, 23)
ax.set_ylim(0.000, 0.040)

plt.tight_layout()

# Save
save_dir = "thesis_paper/figures/edsr"
plt.savefig(os.path.join(save_dir, "efficiency_accuracy_tradeoff.png"))
plt.savefig(os.path.join(save_dir, "efficiency_accuracy_tradeoff.pdf"))
plt.savefig(os.path.join(save_dir, "efficiency_accuracy_tradeoff.svg"), format='svg')
plt.close()

print("Plot saved to thesis_paper/figures/edsr/efficiency_accuracy_tradeoff.png/pdf/svg")
