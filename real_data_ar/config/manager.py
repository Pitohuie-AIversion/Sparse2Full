from __future__ import annotations
import os
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import torch
from .defaults import DEFAULT_CONFIG

class SpatiotemporalConfigManager:
    """Configuration manager for Spatiotemporal AR training.
    
    Handles loading, validating, and managing configuration parameters for
    the three-stage training process (Spatial -> Temporal -> Joint).
    """

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> DictConfig:
        """Load configuration from a file or return defaults.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            DictConfig: The loaded configuration object.
        """
        if config_path and os.path.exists(config_path):
            user_config = OmegaConf.load(config_path)
            # Merge user config with defaults
            config = OmegaConf.merge(DEFAULT_CONFIG, user_config)
        else:
            config = DEFAULT_CONFIG.copy()
        
        return config

    @staticmethod
    def validate_config(config: DictConfig) -> DictConfig:
        """Validate and correct configuration parameters.

        Args:
            config: The configuration object to validate.

        Returns:
            DictConfig: The validated and corrected configuration.
        """
        # DataLoader parameter correction
        if 'data' in config and 'dataloader' in config.data:
            dl = config.data.dataloader
            num_workers = dl.get('num_workers', 0)
            if num_workers == 0:
                dl['prefetch_factor'] = None
                dl['persistent_workers'] = False

        # AMP precision settings
        exp = config.get('experiment', {})
        precision = exp.get('precision', '32')
        if precision == 'auto':
            exp['precision'] = '16-mixed' if torch.cuda.is_available() else '32'

        # Observation operator parameters
        obs = config.get('observation', {})
        k = int(obs.get('kernel_size', 1))
        if k % 2 == 0:
            obs['kernel_size'] = k + 1
        sigma = float(obs.get('blur_sigma', 0.0))
        if sigma < 0:
            obs['blur_sigma'] = 0.0
        if obs.get('downsample_interpolation') not in {'area', 'nearest', 'bilinear'}:
            obs['downsample_interpolation'] = 'area'

        # Early stopping and checkpointing
        tr = config.get('training', {})
        es = tr.get('early_stopping', {})
        if es:
            es['patience'] = max(20, int(es.get('patience', 20)))
        ck = tr.get('checkpoint', {})
        if ck:
            ck['max_keep'] = max(2, int(ck.get('max_keep', 2)))
            ck['save_every_n_epochs'] = max(0, int(ck.get('save_every_n_epochs', 0)))

        return config
