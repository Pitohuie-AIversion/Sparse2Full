#!/usr/bin/env python3
"""
Paper package generation for PDEBench sparse observation reconstruction.
Compatible with golden rules and reproducibility requirements.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from ..utils.logger import get_logger
from ..utils.metrics import compute_all_metrics
from ..ops.degradation import ObservationOperator

logger = get_logger(__name__)


class PaperPackageGenerator:
    """Generate paper package with all required materials for reproducibility."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.package_dir = self.output_dir / "paper_package"
        self.package_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            "data_cards": self.package_dir / "data_cards",
            "configs": self.package_dir / "configs", 
            "checkpoints": self.package_dir / "checkpoints",
            "metrics": self.package_dir / "metrics",
            "figs": self.package_dir / "figs",
            "scripts": self.package_dir / "scripts"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def generate_complete_package(
        self,
        trainer,
        validation_results: Dict[str, Any],
        checkpoints: List[Path],
        seed_results: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Path:
        """Generate complete paper package with all materials."""
        logger.info("Generating paper package...")
        
        # 1. Generate data cards
        self._generate_data_cards()
        
        # 2. Save configuration snapshots
        self._save_config_snapshots()
        
        # 3. Copy checkpoints
        self._copy_checkpoints(checkpoints)
        
        # 4. Generate metrics
        self._generate_metrics(validation_results, seed_results)
        
        # 5. Generate visualizations
        self._generate_visualizations(trainer, validation_results)
        
        # 6. Generate scripts
        self._generate_scripts()
        
        # 7. Generate README
        self._generate_readme()
        
        # 8. Validate package
        self._validate_package()
        
        logger.info(f"Paper package generated at: {self.package_dir}")
        return self.package_dir
    
    def _generate_data_cards(self):
        """Generate data cards with source and licensing information."""
        data_config = self.config.get("data", {})
        
        data_card = {
            "dataset": data_config.get("name", "unknown"),
            "source": "PDEBench",
            "version": "1.0",
            "description": "Sparse observation reconstruction dataset",
            "variables": data_config.get("keys", ["u"]),
            "image_size": data_config.get("image_size", 256),
            "observation_mode": data_config.get("observation_mode", "SR"),
            "observation_params": data_config.get("observation_params", {}),
            "splits": {
                "train": "splits/train.txt",
                "validation": "splits/val.txt", 
                "test": "splits/test.txt"
            },
            "normalization": "z-score per channel",
            "boundary_conditions": "periodic",
            "license": "CC BY 4.0",
            "citation": "PDEBench Consortium, 2023",
            "generated_at": datetime.now().isoformat()
        }
        
        with open(self.dirs["data_cards"] / "dataset_card.json", "w") as f:
            json.dump(data_card, f, indent=2)
        
        # Generate split information
        splits_info = {
            "train_samples": len(self._load_split_file("train")),
            "val_samples": len(self._load_split_file("val")),
            "test_samples": len(self._load_split_file("test")),
            "total_samples": sum(len(self._load_split_file(split)) for split in ["train", "val", "test"])
        }
        
        with open(self.dirs["data_cards"] / "splits_info.json", "w") as f:
            json.dump(splits_info, f, indent=2)
    
    def _save_config_snapshots(self):
        """Save configuration snapshots for reproducibility."""
        # Save merged configuration
        with open(self.dirs["configs"] / "config_merged.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save git information
        try:
            git_info = {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip(),
                "status": subprocess.check_output(["git", "status", "--porcelain"]).decode().strip(),
                "remote_url": subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
            }
            
            with open(self.dirs["configs"] / "git_info.json", "w") as f:
                json.dump(git_info, f, indent=2)
        except subprocess.CalledProcessError:
            logger.warning("Could not retrieve git information")
        
        # Save environment information
        import sys
        import torch
        
        env_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(self.dirs["configs"] / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)
    
    def _copy_checkpoints(self, checkpoints: List[Path]):
        """Copy key checkpoints to paper package."""
        for checkpoint in checkpoints:
            if checkpoint.exists():
                dst_path = self.dirs["checkpoints"] / checkpoint.name
                shutil.copy2(checkpoint, dst_path)
        
        # Create checkpoint manifest
        manifest = {
            "checkpoints": [
                {
                    "name": cp.name,
                    "path": str(cp),
                    "size_bytes": cp.stat().st_size if cp.exists() else 0,
                    "created_at": datetime.fromtimestamp(cp.stat().st_mtime).isoformat() if cp.exists() else None
                }
                for cp in checkpoints
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(self.dirs["checkpoints"] / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _generate_metrics(
        self, 
        validation_results: Dict[str, Any],
        seed_results: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        """Generate comprehensive metrics and statistical analysis."""
        
        # Case-level metrics
        case_metrics = []
        for case_idx, case_result in enumerate(validation_results.get("cases", [])):
            case_metric = {
                "case_id": case_idx,
                "rel_l2": case_result.get("rel_l2", 0.0),
                "mae": case_result.get("mae", 0.0),
                "psnr": case_result.get("psnr", 0.0),
                "ssim": case_result.get("ssim", 0.0),
                "frmse_low": case_result.get("frmse_low", 0.0),
                "frmse_mid": case_result.get("frmse_mid", 0.0),
                "frmse_high": case_result.get("frmse_high", 0.0),
                "brmse": case_result.get("brmse", 0.0),
                "data_consistency_error": case_result.get("data_consistency_error", 0.0)
            }
            case_metrics.append(case_metric)
        
        # Save case-level metrics
        with open(self.dirs["metrics"] / "case_metrics.jsonl", "w") as f:
            for case_metric in case_metrics:
                f.write(json.dumps(case_metric) + "\n")
        
        # Aggregate metrics
        aggregate_metrics = {
            "mean": {
                metric: np.mean([cm[metric] for cm in case_metrics if metric in cm])
                for metric in ["rel_l2", "mae", "psnr", "ssim", "frmse_low", "frmse_mid", "frmse_high", "brmse", "data_consistency_error"]
            },
            "std": {
                metric: np.std([cm[metric] for cm in case_metrics if metric in cm])
                for metric in ["rel_l2", "mae", "psnr", "ssim", "frmse_low", "frmse_mid", "frmse_high", "brmse", "data_consistency_error"]
            },
            "min": {
                metric: np.min([cm[metric] for cm in case_metrics if metric in cm])
                for metric in ["rel_l2", "mae", "psnr", "ssim", "frmse_low", "frmse_mid", "frmse_high", "brmse", "data_consistency_error"]
            },
            "max": {
                metric: np.max([cm[metric] for cm in case_metrics if metric in cm])
                for metric in ["rel_l2", "mae", "psnr", "ssim", "frmse_low", "frmse_mid", "frmse_high", "brmse", "data_consistency_error"]
            },
            "n_cases": len(case_metrics),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(self.dirs["metrics"] / "aggregate_metrics.json", "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        # Generate statistical analysis if multiple seeds available
        if seed_results and len(seed_results) >= 3:
            self._generate_statistical_analysis(seed_results)
        
        # Generate LaTeX table
        self._generate_latex_table(aggregate_metrics)
    
    def _generate_statistical_analysis(self, seed_results: Dict[int, Dict[str, Any]]):
        """Generate statistical analysis for multiple seeds."""
        from scipy import stats
        
        # Collect metrics across seeds
        seed_metrics = {}
        for seed, results in seed_results.items():
            if "aggregate" in results:
                for metric, value in results["aggregate"].items():
                    if metric not in seed_metrics:
                        seed_metrics[metric] = []
                    seed_metrics[metric].append(value)
        
        # Statistical analysis
        analysis = {
            "sample_size": len(seed_results),
            "metrics": {}
        }
        
        for metric, values in seed_metrics.items():
            if len(values) >= 3:
                values_array = np.array(values)
                analysis["metrics"][metric] = {
                    "mean": np.mean(values_array),
                    "std": np.std(values_array),
                    "min": np.min(values_array),
                    "max": np.max(values_array),
                    "confidence_interval_95": stats.t.interval(
                        0.95, len(values_array)-1, 
                        loc=np.mean(values_array), 
                        scale=stats.sem(values_array)
                    ),
                    "coefficient_of_variation": np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else 0
                }
        
        with open(self.dirs["metrics"] / "statistical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
    
    def _generate_latex_table(self, aggregate_metrics: Dict[str, Any]):
        """Generate LaTeX table for paper."""
        latex_content = """
\\begin{table}[ht]
\\centering
\\caption{Performance metrics for sparse observation reconstruction.}
\\label{tab:results}
\\begin{tabular}{lccc}
\\toprule
Metric & Mean & Std & Range \\\\
\\midrule
"""
        
        metrics_order = ["rel_l2", "mae", "psnr", "ssim", "frmse_low", "frmse_mid", "frmse_high", "brmse", "data_consistency_error"]
        
        for metric in metrics_order:
            if metric in aggregate_metrics["mean"]:
                mean_val = aggregate_metrics["mean"][metric]
                std_val = aggregate_metrics["std"][metric]
                min_val = aggregate_metrics["min"][metric]
                max_val = aggregate_metrics["max"][metric]
                
                metric_name = metric.replace("_", " ").title()
                latex_content += f"{metric_name} & {mean_val:.4f} & {std_val:.4f} & [{min_val:.4f}, {max_val:.4f}] \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open(self.dirs["metrics"] / "results_table.tex", "w") as f:
            f.write(latex_content)
    
    def _generate_visualizations(self, trainer, validation_results: Dict[str, Any]):
        """Generate standard visualizations."""
        
        # Training curves
        if hasattr(trainer, "monitor") and trainer.monitor.metrics_history:
            self._plot_training_curves(trainer.monitor.metrics_history)
        
        # Validation visualizations
        if "cases" in validation_results:
            self._plot_validation_cases(validation_results["cases"])
        
        # Power spectrum analysis
        if "cases" in validation_results:
            self._plot_power_spectrum(validation_results["cases"])
        
        # Resource usage
        if hasattr(trainer, "resource_monitor"):
            self._plot_resource_usage(trainer.resource_monitor)
    
    def _plot_training_curves(self, metrics_history: Dict[str, List[float]]):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Curves", fontsize=16)
        
        # Loss curves
        if "train_loss" in metrics_history:
            axes[0, 0].plot(metrics_history["train_loss"], label="Train Loss")
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if "val_loss" in metrics_history:
            axes[0, 1].plot(metrics_history["val_loss"], label="Val Loss", color="orange")
            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        if "learning_rate" in metrics_history:
            axes[1, 0].plot(metrics_history["learning_rate"], label="LR", color="green")
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Validation metrics
        if "val_rel_l2" in metrics_history:
            axes[1, 1].plot(metrics_history["val_rel_l2"], label="Rel-L2", color="red")
            axes[1, 1].set_title("Validation Rel-L2")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Rel-L2 Error")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.dirs["figs"] / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_validation_cases(self, cases: List[Dict[str, Any]]):
        """Plot validation case comparisons."""
        n_cases = min(3, len(cases))  # Show up to 3 cases
        
        for i in range(n_cases):
            case = cases[i]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Validation Case {i+1}", fontsize=16)
            
            # Ground truth
            if "ground_truth" in case:
                gt = case["ground_truth"]
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                
                im1 = axes[0].imshow(gt[0] if gt.ndim > 2 else gt, cmap="viridis")
                axes[0].set_title("Ground Truth")
                axes[0].set_xlabel("x")
                axes[0].set_ylabel("y")
                plt.colorbar(im1, ax=axes[0])
            
            # Prediction
            if "prediction" in case:
                pred = case["prediction"]
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                
                im2 = axes[1].imshow(pred[0] if pred.ndim > 2 else pred, cmap="viridis")
                axes[1].set_title("Prediction")
                axes[1].set_xlabel("x")
                axes[1].set_ylabel("y")
                plt.colorbar(im2, ax=axes[1])
            
            # Error
            if "ground_truth" in case and "prediction" in case:
                error = np.abs(gt - pred)
                
                im3 = axes[2].imshow(error[0] if error.ndim > 2 else error, cmap="Reds")
                axes[2].set_title("Absolute Error")
                axes[2].set_xlabel("x")
                axes[2].set_ylabel("y")
                plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(self.dirs["figs"] / f"validation_case_{i+1}.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    def _plot_power_spectrum(self, cases: List[Dict[str, Any]]):
        """Plot power spectrum analysis."""
        n_cases = min(3, len(cases))
        
        fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 5))
        if n_cases == 1:
            axes = [axes]
        
        fig.suptitle("Power Spectrum Analysis", fontsize=16)
        
        for i in range(n_cases):
            case = cases[i]
            
            if "ground_truth" in case and "prediction" in case:
                gt = case["ground_truth"]
                pred = case["prediction"]
                
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                
                # Compute 2D FFT
                gt_fft = np.fft.fft2(gt[0] if gt.ndim > 2 else gt)
                pred_fft = np.fft.fft2(pred[0] if pred.ndim > 2 else pred)
                
                gt_power = np.log(np.abs(gt_fft) + 1e-10)
                pred_power = np.log(np.abs(pred_fft) + 1e-10)
                
                # Plot
                im = axes[i].imshow(np.abs(gt_power - pred_power), cmap="RdBu", vmin=-5, vmax=5)
                axes[i].set_title(f"Case {i+1}: Power Spectrum Difference")
                axes[i].set_xlabel("Frequency (kx)")
                axes[i].set_ylabel("Frequency (ky)")
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.dirs["figs"] / "power_spectrum_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_resource_usage(self, resource_monitor):
        """Plot resource usage over time."""
        if not hasattr(resource_monitor, "history") or not resource_monitor.history:
            return
        
        history = resource_monitor.history
        times = [h["timestamp"] for h in history]
        gpu_memory = [h.get("gpu_memory_mb", 0) for h in history]
        gpu_utilization = [h.get("gpu_utilization", 0) for h in history]
        cpu_utilization = [h.get("cpu_utilization", 0) for h in history]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("Resource Usage", fontsize=16)
        
        # GPU usage
        axes[0].plot(times, gpu_memory, label="GPU Memory (MB)", color="blue")
        axes[0].set_ylabel("GPU Memory (MB)", color="blue")
        axes[0].tick_params(axis="y", labelcolor="blue")
        
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(times, gpu_utilization, label="GPU Utilization (%)", color="red")
        ax0_twin.set_ylabel("GPU Utilization (%)", color="red")
        ax0_twin.tick_params(axis="y", labelcolor="red")
        
        axes[0].set_title("GPU Resources")
        axes[0].grid(True)
        
        # CPU usage
        axes[1].plot(times, cpu_utilization, label="CPU Utilization (%)", color="green")
        axes[1].set_title("CPU Utilization")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("CPU Utilization (%)")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.dirs["figs"] / "resource_usage.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _generate_scripts(self):
        """Generate reproduction scripts."""
        
        # Training reproduction script
        train_script = f"""#!/bin/bash
# Reproduction script for {self.config.get('experiment', {}).get('name', 'experiment')}
# Generated on {datetime.now().isoformat()}

# Set environment
export PYTHONPATH=/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full:$PYTHONPATH

# Set random seed for reproducibility
export PYTHONHASHSEED=0

# Run training with saved configuration
python tools/train.py \\
    --config-path={self.dirs['configs']} \\
    --config-name=config_merged \\
    experiment.seed=42

# Run evaluation
python tools/evaluate.py \\
    --checkpoint={self.dirs['checkpoints']}/best_model.pt \\
    --config={self.dirs['configs']}/config_merged.yaml

# Generate paper package (if not already done)
python tools/generate_paper_package.py \\
    --run-dir={self.output_dir} \\
    --output-dir={self.package_dir}
"""
        
        with open(self.dirs["scripts"] / "reproduce.sh", "w") as f:
            f.write(train_script)
        
        # Make executable
        (self.dirs["scripts"] / "reproduce.sh").chmod(0o755)
        
        # Summary generation script
        summary_script = f"""#!/usr/bin/env python3
"""Generate summary report from paper package."""

import json
import pandas as pd
from pathlib import Path

package_dir = Path("{self.package_dir}")

# Load metrics
with open(package_dir / "metrics" / "aggregate_metrics.json", "r") as f:
    metrics = json.load(f)

# Create summary table
df = pd.DataFrame({
    'Metric': list(metrics['mean'].keys()),
    'Mean': list(metrics['mean'].values()),
    'Std': list(metrics['std'].values()),
    'Min': list(metrics['min'].values()),
    'Max': list(metrics['max'].values())
})

print("Performance Summary:")
print(df.to_string(index=False))

# Save summary
df.to_csv(package_dir / "metrics" / "summary.csv", index=False)
print(f"\\nSummary saved to {package_dir / 'metrics' / 'summary.csv'}")
"""
        
        with open(self.dirs["scripts"] / "generate_summary.py", "w") as f:
            f.write(summary_script)
        
        # Make executable
        (self.dirs["scripts"] / "generate_summary.py").chmod(0o755)
    
    def _generate_readme(self):
        """Generate README for paper package."""
        
        readme_content = f"""# Paper Package: {self.config.get('experiment', {}).get('name', 'PDEBench Experiment')}

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This paper package contains all materials required for reproducibility and peer review of the sparse observation reconstruction experiment.

## Contents

### Data Cards (`data_cards/`)
- `dataset_card.json`: Dataset metadata and licensing information
- `splits_info.json`: Training/validation/test split statistics

### Configurations (`configs/`)
- `config_merged.yaml`: Complete experiment configuration
- `git_info.json`: Git repository information (commit, branch, status)
- `environment.json`: Python/PyTorch environment details

### Checkpoints (`checkpoints/`)
- Model checkpoints for reproduction
- `manifest.json`: Checkpoint metadata

### Metrics (`metrics/`)
- `case_metrics.jsonl`: Per-case performance metrics
- `aggregate_metrics.json`: Aggregate statistics (mean, std, min, max)
- `statistical_analysis.json`: Statistical analysis (if multiple seeds)
- `results_table.tex`: LaTeX table for papers
- `summary.csv`: Summary table in CSV format

### Figures (`figs/`)
- `training_curves.png`: Training loss and metric curves
- `validation_case_*.png`: Validation case comparisons (GT, Pred, Error)
- `power_spectrum_analysis.png`: Frequency domain analysis
- `resource_usage.png`: GPU/CPU resource usage (if available)

### Scripts (`scripts/`)
- `reproduce.sh`: Shell script to reproduce the experiment
- `generate_summary.py`: Python script to generate summary reports

## Reproduction

To reproduce this experiment:

```bash
# Navigate to package directory
cd {self.package_dir}

# Run reproduction script
./scripts/reproduce.sh
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA (recommended)
- See `configs/environment.json` for exact versions

## License

Code: MIT/Apache-2.0
Data: CC BY 4.0 (PDEBench)
Checkpoints: Follow original model licenses

## Citation

If you use this work, please cite:

```bibtex
@article{{sparsepdebench2024,
  title={{Sparse Observation Reconstruction for PDEBench}},
  author={{Your Name}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```

## Contact

For questions about this paper package, please open an issue in the repository or contact the authors.
"""
        
        with open(self.package_dir / "README.md", "w") as f:
            f.write(readme_content)
    
    def _validate_package(self):
        """Validate paper package completeness."""
        
        required_files = [
            self.dirs["data_cards"] / "dataset_card.json",
            self.dirs["configs"] / "config_merged.yaml",
            self.dirs["metrics"] / "aggregate_metrics.json",
            self.dirs["figs"] / "training_curves.png",
            self.package_dir / "README.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.warning(f"Missing required files in paper package: {missing_files}")
        else:
            logger.info("Paper package validation passed")
    
    def _load_split_file(self, split: str) -> List[str]:
        """Load split file."""
        split_file = Path(self.config.get("data", {}).get("path", "")) / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []