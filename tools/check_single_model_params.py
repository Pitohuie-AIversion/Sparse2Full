import sys
import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tools.training.train_real_data_ar import RealDataARTrainer

def mock_setup_data(self):
    """Mock data setup to avoid loading files"""
    print("⚡ Mocking setup_data - skipping I/O")
    self.train_loader = None
    self.val_loader = None
    self.test_loader = None
    # Set necessary attributes for model setup
    if not hasattr(self.config, 'data'):
        self.config.data = OmegaConf.create({})
    self.config.data.img_size = 128
    self.config.data.input_channels = 1
    self.config.data.target_channels = 1
    self.config.data.T_in = 1
    self.config.data.T_out = 1
    
    # Mock observation op if needed (for consistency checks inside trainer)
    self.observation_op = lambda x: x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Monkey patch
    RealDataARTrainer.setup_data = mock_setup_data

    # Overrides for budget
    overrides = [
        "model_budget.target_params_m=10.0",
        "model_budget.tolerance_m=2.0",
        "model_budget.auto_tune=True",
        "training.validation.enabled=False",
        "logging.log_model=False",
        "device.accelerator=cuda", # Use GPU for model creation to catch CUDA specific issues
        f"model.name={args.model}"
    ]

    try:
        trainer = RealDataARTrainer(
            config_path=args.config,
            model_name=args.model,
            overrides=overrides,
            skip_optimizer=True,
            skip_monitoring=True
        )
        
        # Manually trigger model setup if not done in init (it is done in init)
        # Check params
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module
            
        total_params = sum(p.numel() for p in model.parameters())
        print(f"RESULT:SUCCESS:{total_params/1e6:.4f}")
        
    except Exception as e:
        # Clean error message
        msg = str(e).replace("\n", " ")
        print(f"RESULT:FAILURE:{msg}")

if __name__ == "__main__":
    main()
