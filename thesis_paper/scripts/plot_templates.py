import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern for math
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

def plot_qualitative_comparison(gt, pred, baseline, title="Qualitative Comparison", save_path="qualitative.pdf"):
    """
    绘制定性对比图：GT, Ours, Baseline 以及对应的 Error Map
    Args:
        gt: (H, W) numpy array
        pred: (H, W) numpy array (Ours)
        baseline: (H, W) numpy array
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    # 统一 Value Range 以便于比较
    vmin = min(gt.min(), pred.min(), baseline.min())
    vmax = max(gt.max(), pred.max(), baseline.max())
    
    # 第一行：Fields
    im0 = axes[0, 0].imshow(gt, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ours")
    axes[0, 1].axis('off')
    
    im2 = axes[0, 2].imshow(baseline, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Baseline")
    axes[0, 2].axis('off')
    
    # Colorbar for fields
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    # 第二行：Error Maps (|Pred - GT|)
    err_pred = np.abs(pred - gt)
    err_baseline = np.abs(baseline - gt)
    
    # Error map range (usually start from 0)
    err_max = max(err_pred.max(), err_baseline.max())
    
    # Placeholder for alignment
    axes[1, 0].axis('off')
    
    im4 = axes[1, 1].imshow(err_pred, cmap='bwr', vmin=0, vmax=err_max)
    axes[1, 1].set_title("Ours Error")
    axes[1, 1].axis('off')
    
    im5 = axes[1, 2].imshow(err_baseline, cmap='bwr', vmin=0, vmax=err_max)
    axes[1, 2].set_title("Baseline Error")
    axes[1, 2].axis('off')
    
    # Colorbar for error
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im5, cax=cax)
    cbar.formatter.set_powerlimits((0, 0)) # Use scientific notation if needed

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

def plot_spectral_analysis(gt_spec, pred_specs, labels, save_path="spectral.pdf"):
    """
    绘制径向平均功率谱
    Args:
        gt_spec: (N,) 1D array of GT power spectrum
        pred_specs: List of (N,) arrays for models
        labels: List of names (e.g., ["Ours", "U-Net"])
    """
    plt.figure(figsize=(6, 5))
    
    wavenumbers = np.arange(len(gt_spec))
    
    plt.loglog(wavenumbers, gt_spec, 'k-', linewidth=2, label='Ground Truth')
    
    colors = ['r', 'b', 'g', 'm']
    markers = ['o', 's', '^', 'v']
    
    for i, (spec, label) in enumerate(zip(pred_specs, labels)):
        plt.loglog(wavenumbers, spec, color=colors[i % len(colors)], 
                   linestyle='--', linewidth=1.5, label=label)
        
    plt.xlabel(r'Wavenumber $k$')
    plt.ylabel(r'Power Spectrum $E(k)$')
    plt.title('Radially Averaged Power Spectrum')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

def plot_robustness_curve(x_values, metric_dict, x_label="Sparsity (%)", y_label="Rel L2 Error", save_path="robustness.pdf"):
    """
    绘制鲁棒性曲线
    Args:
        x_values: list or array of x-axis values
        metric_dict: dict {model_name: [y_values]}
    """
    plt.figure(figsize=(6, 5))
    
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, (name, y_values) in enumerate(metric_dict.items()):
        plt.plot(x_values, y_values, marker=markers[i % len(markers)], 
                 linewidth=2, markersize=6, label=name)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

def plot_pareto_frontier(params, errors, names, save_path="pareto.pdf"):
    """
    绘制参数量 vs 误差的 Pareto 图
    """
    plt.figure(figsize=(7, 6))
    
    plt.scatter(params, errors, s=100, alpha=0.7, c='tab:blue', edgecolors='k')
    
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], errors[i]), xytext=(5, 5), 
                     textcoords='offset points')
        
    plt.xlabel('Parameters (M)')
    plt.ylabel('Rel L2 Error')
    plt.title('Efficiency vs. Performance')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight Ours if name contains "Ours"
    for i, name in enumerate(names):
        if "Ours" in name:
            plt.scatter([params[i]], [errors[i]], s=150, c='red', marker='*', label='Ours')
            
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Mock data for demonstration
    H, W = 128, 128
    gt = np.random.randn(H, W)
    pred = gt + np.random.randn(H, W) * 0.1
    baseline = gt + np.random.randn(H, W) * 0.3
    
    print("Generating demo plots...")
    plot_qualitative_comparison(gt, pred, baseline, save_path="demo_qualitative.png") # png for preview
    
    k = np.arange(1, 64)
    gt_spec = k ** (-3.0)
    pred_spec = k ** (-3.0) * (1 + 0.1 * np.random.randn(63))
    unet_spec = k ** (-3.5) # decays faster
    
    plot_spectral_analysis(gt_spec, [pred_spec, unet_spec], ["Ours", "U-Net"], save_path="demo_spectral.png")
    
    x = [0.1, 1, 5, 10]
    y_ours = [0.05, 0.02, 0.01, 0.005]
    y_base = [0.2, 0.1, 0.05, 0.03]
    plot_robustness_curve(x, {"Ours": y_ours, "Baseline": y_base}, save_path="demo_robustness.png")
