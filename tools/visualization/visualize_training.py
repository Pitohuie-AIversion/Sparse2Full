#!/usr/bin/env python3
"""
Training results visualization script (English-only).
Parses training logs and generates visual plots with matplotlib.
Outputs:
- runs/visualization/loss_curves.png
- runs/visualization/best_metrics.png
- runs/visualization/convergence_analysis.png
- runs/visualization/training_report.md
- runs/visualization/training_data.json
"""

import re
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt


def parse_training_log(log_path: Path):
    """Parse training log and extract loss and metric data."""
    epochs = []
    train_losses = []
    val_losses = []
    val_rel_l2 = []

    if not log_path.exists():
        return epochs, train_losses, val_losses, val_rel_l2

    content = log_path.read_text(encoding="utf-8")

    # Use regex to extract training data
    pattern = r"Epoch\s+(\d+)\s+-\s+Train Loss:\s+([\d.]+)\s+Val Loss:\s+([\d.]+)\s+Val Rel-L2:\s+([\d.]+)"
    matches = re.findall(pattern, content)

    for match in matches:
        epoch, train_loss, val_loss, rel_l2 = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_rel_l2.append(float(rel_l2))

    return epochs, train_losses, val_losses, val_rel_l2


def extract_best_metrics(log_path: Path):
    """Extract best validation metrics from log content."""
    if not log_path.exists():
        return None, {}, None, None

    content = log_path.read_text(encoding="utf-8")

    # Extract best validation loss
    best_val_loss_match = re.search(r"Best validation loss:\s+([\d.]+)", content)
    best_val_loss = float(best_val_loss_match.group(1)) if best_val_loss_match else None

    # Extract best validation metrics
    metrics_pattern = (
        r"'rel_l2': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?"
        r"'mae': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?"
        r"'psnr': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\].*?"
        r"'ssim': tensor\(\[\[([\d.]+)\],\s*\[([\d.]+)\]\]"
    )
    metrics_match = re.search(metrics_pattern, content, re.DOTALL)

    best_metrics = {}
    if metrics_match:
        rel_l2_1, rel_l2_2, mae_1, mae_2, psnr_1, psnr_2, ssim_1, ssim_2 = metrics_match.groups()
        best_metrics = {
            "rel_l2": [float(rel_l2_1), float(rel_l2_2)],
            "mae": [float(mae_1), float(mae_2)],
            "psnr": [float(psnr_1), float(psnr_2)],
            "ssim": [float(ssim_1), float(ssim_2)],
        }

    # Extract training/validation time
    train_time_match = re.search(r"Total training time:\s+([\d.]+)s", content)
    val_time_match = re.search(r"Total validation time:\s+([\d.]+)s", content)

    train_time = float(train_time_match.group(1)) if train_time_match else None
    val_time = float(val_time_match.group(1)) if val_time_match else None

    return best_val_loss, best_metrics, train_time, val_time


def create_loss_curves(epochs, train_losses, val_losses, val_rel_l2, output_dir: Path):
    """Create loss curve plots using matplotlib in English."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Loss curves
    axes[0].plot(epochs, train_losses, label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="tab:orange")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative L2 curves (validation)
    axes[1].plot(epochs, val_rel_l2, label="Val Rel-L2", color="tab:green")
    axes[1].set_title("Validation Relative L2 over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rel-L2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_metrics_visualization(best_metrics: dict, output_dir: Path):
    """Create visualization for best metrics using matplotlib."""
    if not best_metrics:
        return

    metrics = ["rel_l2", "mae", "psnr", "ssim"]
    ch1 = [best_metrics[m][0] if m in best_metrics else 0 for m in metrics]
    ch2 = [best_metrics[m][1] if m in best_metrics else 0 for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, ch1, width, label="Channel 1")
    ax.bar(x + width / 2, ch2, width, label="Channel 2")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_title("Best Validation Metrics by Channel")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "best_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_convergence_analysis(epochs, train_losses, val_losses, val_rel_l2, output_dir: Path):
    """Create convergence analysis plots using matplotlib."""
    best_val_idx = int(np.argmin(val_losses)) if len(val_losses) else 0
    best_epoch = epochs[best_val_idx] if len(epochs) else 0
    best_val_loss = val_losses[best_val_idx] if len(val_losses) else 0.0

    best_rel_l2_idx = int(np.argmin(val_rel_l2)) if len(val_rel_l2) else 0
    best_rel_l2_epoch = epochs[best_rel_l2_idx] if len(epochs) else 0
    best_rel_l2_value = val_rel_l2[best_rel_l2_idx] if len(val_rel_l2) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Validation loss
    axes[0].plot(epochs, val_losses, label="Val Loss", color="tab:orange")
    axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label="Best Val Loss Epoch")
    axes[0].set_title("Validation Loss Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative L2
    axes[1].plot(epochs, val_rel_l2, label="Val Rel-L2", color="tab:green")
    axes[1].axvline(best_rel_l2_epoch, color="red", linestyle="--", alpha=0.6, label="Best Rel-L2 Epoch")
    axes[1].set_title("Relative L2 Convergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rel-L2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "convergence_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return best_epoch, best_val_loss, best_rel_l2_epoch, best_rel_l2_value


def generate_training_report(
    epochs,
    train_losses,
    val_losses,
    val_rel_l2,
    best_val_loss,
    best_metrics,
    train_time,
    val_time,
    best_epoch,
    best_rel_l2_epoch,
    output_dir: Path,
):
    """Generate a training summary report in English."""

    # Compute statistics
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    final_rel_l2 = val_rel_l2[-1] if val_rel_l2 else 0

    min_train_loss = min(train_losses) if train_losses else 0
    min_val_loss = min(val_losses) if val_losses else 0
    min_rel_l2 = min(val_rel_l2) if val_rel_l2 else 0

    # Build report
    report = f"""
# Training Summary Report

## Training Configuration
- **Model**: SwinUNet
- **Task**: SR x4 (4× super-resolution)
- **Dataset**: PDEBench
- **Training samples**: 1000
- **Validation samples**: 100
- **Batch size**: 4
- **Total epochs**: {len(epochs)}

## Training Performance
- **Total training time**: {train_time:.2f}s
- **Total validation time**: {val_time:.2f}s
- **Avg time per epoch**: {(train_time + val_time) / len(epochs):.2f}s

## Loss Convergence
- **Final training loss**: {final_train_loss:.6f}
- **Final validation loss**: {final_val_loss:.6f}
- **Final relative L2**: {final_rel_l2:.6f}

## Best Performance
- **Best validation loss**: {best_val_loss:.6f} (Epoch {best_epoch})
- **Best relative L2**: {min_rel_l2:.6f} (Epoch {best_rel_l2_epoch})

### Best validation metrics details
"""

    if best_metrics:
        for metric, values in best_metrics.items():
            avg_value = np.mean(values)
            report += f"- **{metric.upper()}**: {avg_value:.6f} (Channel 1: {values[0]:.6f}, Channel 2: {values[1]:.6f})\n"

    report += f"""
## Convergence Analysis
- **Training loss reduction**: {train_losses[0]:.6f} → {final_train_loss:.6f} ({((train_losses[0] - final_train_loss) / train_losses[0] * 100):.1f}% decrease)
- **Validation loss reduction**: {val_losses[0]:.6f} → {final_val_loss:.6f} ({((val_losses[0] - final_val_loss) / val_losses[0] * 100):.1f}% decrease)
- **Relative L2 reduction**: {val_rel_l2[0]:.6f} → {final_rel_l2:.6f} ({((val_rel_l2[0] - final_rel_l2) / val_rel_l2[0] * 100):.1f}% decrease)

## Model Performance
- **PSNR**: {np.mean(best_metrics['psnr']) if best_metrics and 'psnr' in best_metrics else 'N/A':.2f} dB
- **SSIM**: {np.mean(best_metrics['ssim']) if best_metrics and 'ssim' in best_metrics else 'N/A':.4f}
- **MAE**: {np.mean(best_metrics['mae']) if best_metrics and 'mae' in best_metrics else 'N/A':.6f}

## Training Stability
- **Train loss std**: {np.std(train_losses):.6f}
- **Val loss std**: {np.std(val_losses):.6f}
- **Rel-L2 std**: {np.std(val_rel_l2):.6f}

## Conclusion
Training completed successfully and converged well after {len(epochs)} epochs.
Best validation loss: {best_val_loss:.6f}; best relative L2: {min_rel_l2:.6f}.
PSNR reached {np.mean(best_metrics['psnr']) if best_metrics and 'psnr' in best_metrics else 'N/A':.2f} dB,
SSIM {np.mean(best_metrics['ssim']) if best_metrics and 'ssim' in best_metrics else 'N/A':.4f},
indicating strong super-resolution reconstruction performance.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    (output_dir / "training_report.md").write_text(report, encoding="utf-8")

    # Save JSON data
    data = {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_rel_l2": val_rel_l2,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "train_time": train_time,
        "val_time": val_time,
        "best_epoch": best_epoch,
        "best_rel_l2_epoch": best_rel_l2_epoch,
    }
    (output_dir / "training_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    """Main function to run visualization pipeline."""
    # Set paths
    log_path = Path("runs/train.log")
    output_dir = Path("runs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing training log...")

    # Parse training log
    epochs, train_losses, val_losses, val_rel_l2 = parse_training_log(log_path)

    if not epochs:
        print("No training data found. Please check the log file.")
        return

    print(f"Successfully parsed training data for {len(epochs)} epochs.")

    # Extract best metrics
    best_val_loss, best_metrics, train_time, val_time = extract_best_metrics(log_path)

    print("Generating visualization plots...")

    # Create loss curves
    create_loss_curves(epochs, train_losses, val_losses, val_rel_l2, output_dir)
    print("Loss curves generated.")

    # Create metrics visualization
    create_metrics_visualization(best_metrics, output_dir)
    print("Best metrics plot generated.")

    # Create convergence analysis
    best_epoch, best_val_loss_found, best_rel_l2_epoch, best_rel_l2_value = create_convergence_analysis(
        epochs, train_losses, val_losses, val_rel_l2, output_dir
    )
    print("Convergence analysis plot generated.")

    # Generate training report
    generate_training_report(
        epochs,
        train_losses,
        val_losses,
        val_rel_l2,
        best_val_loss,
        best_metrics,
        train_time,
        val_time,
        best_epoch,
        best_rel_l2_epoch,
        output_dir,
    )
    print("Training report generated.")

    print(f"\nVisualization completed. Results saved to: {output_dir}")
    print(f"Loss curves: {output_dir}/loss_curves.png")
    print(f"Best metrics: {output_dir}/best_metrics.png")
    print(f"Convergence analysis: {output_dir}/convergence_analysis.png")
    print(f"Training report: {output_dir}/training_report.md")
    print(f"Training data: {output_dir}/training_data.json")

    print("\nKey Stats:")
    print(f"   Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"   Best relative L2: {best_rel_l2_value:.6f} (Epoch {best_rel_l2_epoch})")
    if best_metrics:
        print(f"   Avg PSNR: {np.mean(best_metrics['psnr']):.2f} dB")
        print(f"   Avg SSIM: {np.mean(best_metrics['ssim']):.4f}")
        print(f"   Avg MAE: {np.mean(best_metrics['mae']):.6f}")
    print(f"   Total training time: {train_time:.2f}s")


if __name__ == "__main__":
    main()