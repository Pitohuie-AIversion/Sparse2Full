import sys
from pathlib import Path
import torch

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer

def evaluate_spatial(ckpt_path, config_path):
    trainer = RealDataARTrainer(
        config_path=config_path,
        overrides=[
            "testing.test_only=true",
            "data.dataloader.test_batch_size=1",
            "data.T_out=1",
            "data.stride=1"
        ]
    )
    if not trainer.load_checkpoint(ckpt_path):
        print("Failed to load checkpoint")
        return None

    res = trainer.test_epoch()
    print(f"Test Results for {config_path}:", res)

if __name__ == '__main__':
    evaluate_spatial(
        './runs_drd_paper/AR-DR2D-UNet-SRx4-Ablation-NoSpec-model_UNet-s2025-20260115/best.ckpt',
        './runs_drd_paper/AR-DR2D-UNet-SRx4-Ablation-NoSpec-model_UNet-s2025-20260115/config_merged.yaml'
    )
