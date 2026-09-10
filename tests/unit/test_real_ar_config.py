import pytest
from omegaconf import DictConfig
from real_data_ar.config.manager import SpatiotemporalConfigManager

def test_load_config_defaults():
    config = SpatiotemporalConfigManager.load_config(None)
    assert isinstance(config, DictConfig)
    assert 'experiment' in config
    assert config.experiment.name == 'spatiotemporal_decomposition'

def test_validate_config():
    config = DictConfig({
        'experiment': {'precision': 'auto'},
        'data': {'dataloader': {'num_workers': 0}},
        'observation': {'kernel_size': 2, 'blur_sigma': -1.0}
    })
    
    validated = SpatiotemporalConfigManager.validate_config(config)
    
    # Check precision update
    assert validated.experiment.precision in ['16-mixed', '32']
    
    # Check dataloader update
    assert validated.data.dataloader.prefetch_factor is None
    assert validated.data.dataloader.persistent_workers is False
    
    # Check observation update
    assert validated.observation.kernel_size == 3  # 2 + 1
    assert validated.observation.blur_sigma == 0.0
