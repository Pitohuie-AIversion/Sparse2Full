import os
import yaml
from pathlib import Path

base_dir = Path("runs")

print(f"{'Experiment':<60} | {'Curriculum':<10} | {'Stages':<20}")
print("-" * 100)

for root, dirs, files in os.walk(base_dir):
    config_path = None
    if "config_merged.yaml" in files:
        config_path = Path(root) / "config_merged.yaml"
    elif "config.yaml" in files:
        config_path = Path(root) / "config.yaml"
    elif "hydra_config.yaml" in files: # Sometimes hydra saves here
        config_path = Path(root) / "hydra_config.yaml"
        
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for curriculum
            # Structure might be training.curriculum or just curriculum
            curriculum = False
            stages = "N/A"
            
            training_cfg = config.get('training', {})
            if isinstance(training_cfg, dict):
                curr_cfg = training_cfg.get('curriculum', {})
                if isinstance(curr_cfg, dict):
                    curriculum = curr_cfg.get('enabled', False)
                    stages = str(curr_cfg.get('stages', []))
            
            # Also check top level if not found
            if not curriculum and 'curriculum' in config:
                curr_cfg = config['curriculum']
                if isinstance(curr_cfg, dict):
                    curriculum = curr_cfg.get('enabled', False)
                    stages = str(curr_cfg.get('stages', []))

            if curriculum:
                print(f"{os.path.basename(root):<60} | {str(curriculum):<10} | {stages:<20}")
        except Exception as e:
            # print(f"Error reading {config_path}: {e}")
            pass
