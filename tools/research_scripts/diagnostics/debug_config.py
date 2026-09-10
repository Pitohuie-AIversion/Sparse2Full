from omegaconf import OmegaConf
from pathlib import Path

config_path = "thesis_paper/configs/ar_paper_aligned_sr4_shallow_water.yaml"
base_cfg = OmegaConf.load(config_path)

print(f"experiment keys: {base_cfg.experiment.keys()}")
print(f"output_dir: {base_cfg.experiment.get('output_dir', 'Not Found')}")
