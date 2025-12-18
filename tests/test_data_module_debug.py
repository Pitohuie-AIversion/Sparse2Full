#!/usr/bin/env python3
"""测试数据模块调试脚本"""

import pytest
from omegaconf import DictConfig

try:
    from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
except Exception:
    RealDiffusionReactionDataModule = None


def test_data_module(data_path_resolver):
    preferred_paths = [
        "2D/diffusion-reaction/2D_diff-react_NA_NA.h5",
        "diffusion-reaction/2D_diff-react_NA_NA.h5",
        "DR2D/2D_diff-react_NA_NA.h5",
    ]
    data_path = data_path_resolver.resolve(preferred_paths)
    if not data_path:
        pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")

    if RealDiffusionReactionDataModule is None:
        pytest.skip("缺少 datasets.real_diffusion_reaction_dataset 模块")

    config = DictConfig(
        {
            "data": {
                "data_path": data_path,
                "T_in": 1,
                "T_out": 2,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "time_step_start": 0,
                "time_step_end": 50,
                "time_step_stride": 1,
                "normalize": True,
                "normalize_sample_size": 8,
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "shuffle": False,
                    "drop_last": False,
                },
            },
            "training": {"batch_size": 1},
            "testing": {"batch_size": 1},
            "hardware": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
            "seed": 2025,
        }
    )

    dm = RealDiffusionReactionDataModule(config)
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    assert "input_sequence" in batch
    assert "target_sequence" in batch
    assert "sample_idx" in batch
    assert "start_time" in batch
    assert batch["input_sequence"].ndim == 5
    assert batch["target_sequence"].ndim == 5
