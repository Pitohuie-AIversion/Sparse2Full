import pytest
import torch
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from omegaconf import DictConfig
from real_data_ar.training.trainer import SpatiotemporalTrainer
from real_data_ar.config.defaults import DEFAULT_CONFIG

@pytest.fixture
def mock_config(tmp_path):
    config = DEFAULT_CONFIG.copy()
    config.experiment.output_dir = str(tmp_path / "runs")
    config.experiment.device = 'cpu'
    
    # Mock data settings
    config.data.data_path = str(tmp_path / "dummy.h5") 
    config.training.spatial_stage.epochs = 1
    config.training.temporal_stage.epochs = 1
    config.training.joint_stage.epochs = 1
    
    return config

def test_trainer_initialization(mock_config):
    # Initialization should succeed even if file doesn't exist (until setup is called)
    trainer = SpatiotemporalTrainer(mock_config)
    assert trainer is not None
    assert trainer.current_stage == 'spatial'

@pytest.fixture
def mock_h5_file(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "dummy.h5"
    with h5py.File(p, 'w') as f:
        for i in range(5):
            grp = f.create_group(f"{i:04d}")
            # [T=20, H=32, W=32, C=2]
            grp.create_dataset('data', data=np.random.randn(20, 32, 32, 2).astype(np.float32))
    return str(p)

def test_trainer_setup(mock_config, mock_h5_file):
    mock_config.data.data_path = mock_h5_file
    mock_config.data.img_size = 32
    mock_config.model.img_size = 32
    
    trainer = SpatiotemporalTrainer(mock_config)
    success = trainer.setup()
    assert success is True

def test_trainer_structure(mock_config, mock_h5_file):
    mock_config.data.data_path = mock_h5_file
    mock_config.data.img_size = 32
    mock_config.model.img_size = 32
    
    trainer = SpatiotemporalTrainer(mock_config)
    trainer.setup()
    
    # Mock model
    trainer.model = Mock(spec=torch.nn.Module)
    trainer.model.spatial_module = Mock(spec=torch.nn.Module)
    trainer.model.temporal_module = Mock(spec=torch.nn.Module)
    
    # Mock parameters
    trainer.model.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(1))])
    trainer.model.spatial_module.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(1))])
    trainer.model.temporal_module.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(1))])
    
    # Mock forward returns
    # Spatial: returns dict or tensor. Trainer expects pred.spatial_pred or pred
    spatial_out = Mock()
    spatial_out.spatial_pred = torch.randn(2, 1, 2, 32, 32, requires_grad=True)
    trainer.model.spatial_module.return_value = spatial_out
    
    # Temporal
    temp_out = Mock()
    temp_out.final_pred = torch.randn(2, 5, 2, 32, 32, requires_grad=True)
    trainer.model.temporal_module.return_value = temp_out
    
    # Joint
    trainer.model.return_value = {'final_pred': torch.randn(2, 5, 2, 32, 32, requires_grad=True)}
    
    # Verify methods exist
    assert hasattr(trainer, 'train_spatial_stage')
    assert hasattr(trainer, 'train_temporal_stage')
    assert hasattr(trainer, 'train_joint_stage')
