import os
import subprocess
import sys
from pathlib import Path

# Configuration
ROOT_DIR = Path("runs_drd_paper/sr_scan_batch")
CONFIG_PATH = "thesis_paper/configs/ar_paper_crop_edsr_spatial_only_refined.yaml"
SCRIPT_PATH = "tools/training/train_real_data_ar.py"

# Define experiments and their scales
experiments = [
    {"input": 32, "scale": 4},
    {"input": 16, "scale": 8},
    {"input": 8, "scale": 16},
    {"input": 4, "scale": 32},
    {"input": 2, "scale": 64},
    {"input": 1, "scale": 128},
]

def run_visualization(exp_config):
    input_size = exp_config["input"]
    scale = exp_config["scale"]
    exp_name = f"AR-DR2D-SR-Scan-Input{input_size}"
    exp_dir = ROOT_DIR / exp_name
    ckpt_path = exp_dir / "best.ckpt"
    
    if not ckpt_path.exists():
        print(f"⚠️ Skipping {exp_name}: Checkpoint not found at {ckpt_path}")
        return

    print(f"\n🎨 Generating visualizations for {exp_name} (Scale x{scale})...")
    
    # Construct command
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--config", CONFIG_PATH,
        "--resume", str(ckpt_path),
        f"experiment.name={exp_name}",
        "--test-only",
        "data.observation.mode=sr",
        f"data.observation.scale={scale}",
        "training.degradation.mode=sr",
        f"training.degradation.scale_factor={scale}",
        f"training.degradation.scale={scale}",
        f"model.upscale={scale}",
        "data.dataloader.batch_size=8",  # Safe batch size for inference
        "training.batch_size=8",
        "training.epochs=100",
        "testing.save_visualizations=true",
        "testing.num_visualization_samples=5",
        "device.devices=1" # Use single GPU for inference
    ]
    
    log_file = exp_dir / "viz_gen.log"
    
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"✅ Success! Visualizations saved in {exp_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate visualizations for {exp_name}. Check logs at {log_file}")

def main():
    if not ROOT_DIR.exists():
        print(f"Error: Root directory {ROOT_DIR} does not exist.")
        return

    for exp in experiments:
        run_visualization(exp)

if __name__ == "__main__":
    main()
