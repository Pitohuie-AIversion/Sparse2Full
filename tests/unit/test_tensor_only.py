import h5py
import numpy as np
import pytest
import torch

from datasets.pdebench import PDEBenchSR


def test_tensor_only_dataset(tmp_path):
    path = tmp_path / "tensor_only.h5"
    rng = np.random.default_rng(123)
    tensor = rng.standard_normal((6, 1, 128, 128), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=tensor)

    dataset = PDEBenchSR(
        data_path=str(path),
        keys=["tensor"],
        scale=4,
        sigma=1.0,
        image_size=128,
        normalize=False,
        split="train",
    )
    sample = dataset[0]
    assert "target" in sample
    assert "baseline" in sample
    assert "lr_observation" in sample
    assert isinstance(sample["target"], torch.Tensor)
    assert sample["target"].shape == (1, 128, 128)

    lr = sample["lr_observation"]
    assert lr.ndim == 3
    assert lr.shape[-2:] == (32, 32)
