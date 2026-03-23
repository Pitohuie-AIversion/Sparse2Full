import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import torch
from omegaconf import DictConfig, OmegaConf

class DeviceManager:
    """Device manager, sets up device and detects distributed environment."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cpu')
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_primary = True

    def setup_device(self) -> torch.device:
        want = self.config.get('experiment', {}).get('device', 'cpu')
        if want == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Simplified distributed setup: environment variable driven
        if os.environ.get('WORLD_SIZE') and os.environ.get('RANK'):
            try:
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.rank = int(os.environ['RANK'])
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.distributed = True
                self.is_primary = self.rank == 0
            except Exception:
                self.distributed = False
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                self.is_primary = True

        return self.device


class LogManager:
    """Log manager, creates logger and optionally saves config snapshot."""

    def __init__(self, config: DictConfig, output_dir: Path, is_primary: bool = True):
        self.config = config
        self.output_dir = Path(output_dir)
        self.is_primary = is_primary
        self.logger = None

    def setup_logging(self) -> logging.Logger:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_name = 'training.log' if self.is_primary else f'training_rank{os.environ.get("RANK", 0)}.log'
        log_file = self.output_dir / log_name
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('trainer')

        # Save config snapshot (merged YAML)
        try:
            merged = OmegaConf.to_yaml(self.config)
            (self.output_dir / 'config_merged.yaml').write_text(merged)
        except Exception:
            pass

        return self.logger

def convert_numpy_types(obj):
    """Recursively convert numpy types to JSON serializable Python native types."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    else:
        return obj

def seed_worker_fn(worker_id: int, base_seed: int = 2025):
    """Set random seed for DataLoader worker."""
    import random
    import numpy as np

    try:
        worker_seed = int(base_seed) + int(worker_id)
    except Exception:
        worker_seed = 2025 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass
