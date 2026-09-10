import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (5, 4),
    "lines.linewidth": 2
})

def main():
    print("Generating Figure 4-8 (fig4-8_mismatch_herr.png)...")
    
    # Data from Table 4-6
    settings = ['Consistent\n(1.0 px)', 'Mismatch (Slight)\n(2.0 px)', 'Mismatch (Severe)\n(3.0 px)']
    h_err = [0.00557, 0.00732, 0.01071]
    
    fig, ax = plt.subplots()
    
    # Bar chart
    bars = ax.bar(settings, h_err, color=['tab:blue', 'tab:orange', 'tab:red'], alpha=0.8, width=0.5)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.0002, 
                f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(r"Observation Consistency Error ($H_{\mathrm{err}}$)")
    ax.set_title("Impact of Observation Operator Mismatch")
    
    # Set y-axis limits to give space for text
    ax.set_ylim(0, 0.012)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-8_mismatch_herr.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
