import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer

run_dir = project_root / "runs_drd_paper" / "AR-DR2D-E2E-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260116"
config_path = run_dir / "config_merged.yaml"
ckpt_path = run_dir / "best.ckpt"

print(f"Loading config from {config_path}")
trainer = RealDataARTrainer(str(config_path))
trainer.output_dir = str(run_dir)
print(f"Loading checkpoint from {ckpt_path}")
trainer.load_checkpoint(str(ckpt_path))

trainer.get_model().eval()
print("Starting test...")
# Modify trainer to only run sample 10
try:
    trainer.test()
except Exception as e:
    print(f"Test failed: {e}")
