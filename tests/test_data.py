import pytest
import torch
import h5py
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from real_data_ar.data.dataset import RealDiffusionReactionDataset
from real_data_ar.data.module import RealDiffusionReactionDataModule

@pytest.fixture
def mock_h5_file(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_data.h5"
    
    with h5py.File(p, 'w') as f:
        # Create 10 samples
        for i in range(10):
            grp = f.create_group(f"{i:04d}")
            # [T, H, W, C]
            grp.create_dataset('data', data=np.random.randn(20, 32, 32, 2).astype(np.float32))
            
    return str(p)

def test_dataset_initialization(mock_h5_file):
    dataset = RealDiffusionReactionDataset(
        data_path=mock_h5_file,
        T_in=1,
        T_out=2,
        split='train',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    assert len(dataset) > 0
    item = dataset[0]
    assert 'input_sequence' in item
    assert 'target_sequence' in item
    assert item['input_sequence'].shape == (1, 2, 32, 32)
    assert item['target_sequence'].shape == (2, 2, 32, 32)

def test_data_module(mock_h5_file):
    config = DictConfig({
        'data': {
            'data_path': mock_h5_file,
            'T_in': 1,
            'T_out': 2,
            'dataloader': {'batch_size': 2, 'num_workers': 0}
        },
        'training': {'batch_size': 2}
    })
    
    dm = RealDiffusionReactionDataModule(config)
    dm.setup()
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch['input_sequence'].shape[0] == 2
    
    spatial_loader = dm.spatial_train_loader()
    assert spatial_loader is not None
    
    temporal_loader = dm.temporal_train_loader()
    assert temporal_loader is not None
