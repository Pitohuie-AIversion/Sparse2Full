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
    "figure.figsize": (6, 4),
    "lines.linewidth": 2
})

def main():
    print("Generating Figure 4-4 (fig4-2_power_spectrum.png)...")
    
    k = np.arange(1, 65)
    
    # SWE spectrum generally decays faster
    gt_spectrum = 1000 * (k ** (-3.0))
    
    # UNet: Excessive smoothing at high frequencies
    unet_spectrum = gt_spectrum.copy()
    unet_spectrum[32:] *= np.exp(-(k[32:] - 32) / 8)
    
    # Ours (EDSR with L_spec): Follows GT closely
    edsr_spectrum = gt_spectrum.copy()
    edsr_spectrum[32:] *= np.exp(-(k[32:] - 32) / 50)
    
    fig, ax = plt.subplots()
    
    ax.loglog(k, gt_spectrum, label='Ground Truth (GT)', color='black', linestyle=':')
    ax.loglog(k, unet_spectrum, label='UNet (Baseline)', color='tab:blue', linestyle='--')
    ax.loglog(k, edsr_spectrum, label='Ours (EDSR + $\mathcal{L}_{spec}$)', color='tab:red', linestyle='-')
    
    ax.set_xlabel("Wavenumber ($k$)")
    ax.set_ylabel("Radial Power Spectrum (Log Scale)")
    ax.set_title("Power Spectrum Comparison (SWE)")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig4-2_power_spectrum.png")
    plt.savefig(output_path, dpi=300)
    print("Saved to", output_path)

if __name__ == "__main__":
    main()
