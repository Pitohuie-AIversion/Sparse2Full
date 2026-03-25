from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .pdebench import PDEBenchBase


class PDEBenchDataset(PDEBenchBase):
    pass


def create_dataloader(
    data_path: str | Path,
    keys: Sequence[str],
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    splits_dir: str | Path | None = None,
    normalize: bool = False,
    image_size: int | None = None,
    **kwargs: dict[str, Any],
) -> DataLoader:
    dataset = PDEBenchDataset(
        data_path=data_path,
        keys=keys,
        split=split,
        splits_dir=splits_dir,
        normalize=normalize,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
    )
