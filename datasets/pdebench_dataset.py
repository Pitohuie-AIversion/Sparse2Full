from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from torch.utils.data import DataLoader

from .pdebench import PDEBenchBase


class PDEBenchDataset(PDEBenchBase):
    pass


def create_dataloader(
    data_path: Union[str, Path, Any],
    keys: Optional[Sequence[str]] = None,
    split: str = "train",
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = False,
    splits_dir: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    image_size: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> DataLoader:
    # 支持直接传入 DictConfig 或 dict
    if not isinstance(data_path, (str, Path)) and hasattr(data_path, "get") or hasattr(data_path, "data"):
        cfg = getattr(data_path, "data", data_path)
        actual_path = str(cfg.get("data_path", "datasets/sample_data.hdf5"))
        keys = list(cfg.get("keys", ["tensor"])) if keys is None else keys
        splits_dir = cfg.get("splits_dir", splits_dir)
        image_size = cfg.get("image_size", image_size)
        if batch_size is None:
            dl_cfg = cfg.get("dataloader", {}) if hasattr(cfg, "get") else {}
            batch_size = int(getattr(dl_cfg, "batch_size", dl_cfg.get("batch_size", 2)) if hasattr(dl_cfg, "get") else 2)
        prep = cfg.get("preprocessing", {}) if hasattr(cfg, "get") else {}
        normalize = bool(getattr(prep, "normalize", prep.get("normalize", normalize)) if hasattr(prep, "get") else normalize)
        data_path = actual_path

    if keys is None:
        keys = ["tensor"]
    if batch_size is None:
        batch_size = 2

    # 如果指定路径不存在且为 sample_data.hdf5，回退到 safe check
    path_obj = Path(str(data_path))
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

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

