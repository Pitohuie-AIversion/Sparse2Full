import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Generating Final Rollout Error Plot from Real Data...")
    
    t = np.arange(1, 21)
    
    # Load actual calculated arrays
    edsr_err = np.load('thesis_paper/figures/rollout/edsr_rollout.npy')
    unet_err = np.load('thesis_paper/figures/rollout/unet_rollout.npy')
    fno_err = np.load('thesis_paper/figures/rollout/fno_rollout.npy')
    bicubic_err = np.load('thesis_paper/figures/rollout/bicubic_rollout.npy')
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    ax.plot(t, edsr_err, label='Ours (Seq-EDSR)', color='tab:red', marker='o', markersize=4, linestyle='-')
    ax.plot(t, unet_err, label='UNet (Baseline)', color='tab:blue', marker='s', markersize=4, linestyle='--')
    ax.plot(t, fno_err, label='FNO', color='tab:orange', marker='^', markersize=4, linestyle='-.')
    ax.plot(t, bicubic_err, label='Bicubic (Interp.)', color='tab:gray', marker='D', markersize=4, linestyle=':')
    
    ax.set_xlabel("Prediction Time Step (t)")
    ax.set_ylabel("Accumulated Rel-L2 Error")
    
    ax.legend(loc='upper left', frameon=True, edgecolor='k')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0, 21)
    
    plt.tight_layout()
    
    save_dir = "thesis_paper/figures/rollout"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, "fig4-4_rollout_error.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, "fig4-4_rollout_error.pdf"))
    plt.savefig(os.path.join(save_dir, "fig4-4_rollout_error.svg"), format='svg')
    plt.close()

    print("Plot saved successfully!")

if __name__ == "__main__":
    main()
