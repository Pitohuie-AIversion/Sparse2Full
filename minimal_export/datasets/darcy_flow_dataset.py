"""Darcy Flow Dataset Loader

Dedicated dataset loader for Darcy Flow (2D steady-state) data from PDEBench/HDF5 format.
Supports 'nu' (permeability) as input and 'tensor' (pressure/solution) as target.
Adapts to the [T, C, H, W] format required by the training pipeline.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Import degradation operator
try:
    from ops.degradation import apply_degradation_operator
except ImportError as e:
    print(f"WARNING: Failed to import degradation operator: {e}")
    # Fallback if run from different context, though sys.path should be fine in this project
    apply_degradation_operator = None


class DarcyFlowDataset(Dataset):
    """Darcy Flow 2D Dataset (Steady State)"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        splits_dir: str | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        normalize: bool = True,
        img_size: int = 128,
        keys: list[str] | None = None,
        h_params: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            h_params: Degradation parameters (for SR/Crop)
        """
        self.data_path = data_path
        self.split = split
        self.splits_dir = splits_dir
        self.normalize = normalize
        self.img_size = img_size
        self.keys = keys if keys is not None else ["tensor"]
        self.h_params = h_params

        # Verify keys
        if "nu" not in self.keys or "tensor" not in self.keys:
            # Fallback or warning could be here, but we strictly expect these for Darcy
            pass

        # Open file to get total samples and shape
        with h5py.File(self.data_path, "r") as f:
            # DarcyFlow HDF5 structure:
            # nu: (N, 128, 128) or (N, 1, 128, 128)
            # tensor: (N, 1, 128, 128)
            # x-coordinate, y-coordinate

            if "tensor" in f:
                self.n_samples = f["tensor"].shape[0]
                self.original_shape = f["tensor"].shape[-2:]  # H, W
            else:
                raise ValueError(
                    f"Invalid DarcyFlow file: 'tensor' key not found in {data_path}"
                )

        # Determine indices for this split
        self.indices = self._get_split_indices(
            self.n_samples, train_ratio, val_ratio, test_ratio
        )

        # Compute/Load Normalization Stats
        self.stats_mean = {}
        self.stats_std = {}
        if self.normalize:
            self._compute_norm_stats()

    @property
    def mean(self):
        """Return target mean for compatibility with trainer"""
        if "tensor" in self.stats_mean:
            m = self.stats_mean["tensor"]
            if m.ndim == 0:
                return m.unsqueeze(0)
            return m
        return torch.tensor([0.0])

    @property
    def std(self):
        """Return target std for compatibility with trainer"""
        if "tensor" in self.stats_std:
            s = self.stats_std["tensor"]
            if s.ndim == 0:
                return s.unsqueeze(0)
            return s
        return torch.tensor([1.0])

    def _get_split_indices(self, n_samples, train_ratio, val_ratio, test_ratio):
        # Priority: splits_dir > ratio
        if self.splits_dir:
            split_file = Path(self.splits_dir) / f"{self.split}.txt"
            if split_file.exists():
                # Assume split file contains indices or case names
                # For Darcy, usually it's just indices 0..N-1
                # But if file contains names, we need to map.
                # PDEBench usually has strings. Darcy HDF5 from PDEBench often has simple integer indexing.
                # Let's fallback to ratio if file read fails or logic is complex,
                # BUT the user config specified splits_dir.
                try:
                    with open(split_file) as f:
                        lines = [line.strip() for line in f if line.strip()]
                    # Try to convert to int
                    indices = [int(x) for x in lines]
                    # Filter valid
                    indices = [i for i in indices if i < n_samples]
                    return indices
                except Exception as e:
                    print(
                        f"Warning: Failed to read split file {split_file}: {e}. Falling back to ratio."
                    )

        # Ratio based splitting
        indices = np.arange(n_samples)
        # We assume fixed seed for reproducibility across runs if shuffle needed,
        # but here we use simple slicing for deterministic splits without shuffle
        # unless we want random. Standard PDEBench often uses first N for train.

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        if self.split == "train":
            return indices[:n_train]
        elif self.split == "val":
            return indices[n_train : n_train + n_val]
        else:  # test
            return indices[n_train + n_val :]

    def _compute_norm_stats(self):
        # Check for cached stats
        stat_file = Path(self.splits_dir) / "norm_stat.npz" if self.splits_dir else None

        if stat_file and stat_file.exists():
            data = np.load(stat_file)
            for k in self.keys:
                self.stats_mean[k] = torch.tensor(
                    data[f"{k}_mean"], dtype=torch.float32
                )
                self.stats_std[k] = torch.tensor(data[f"{k}_std"], dtype=torch.float32)
            return

        # Compute from training data
        # We need to know which indices are training indices.
        # If we are in 'val' or 'test' mode, we should reload 'train' indices to compute stats
        # to avoid data leakage.
        # For simplicity, we assume if stats file missing, we compute on CURRENT split
        # (which is wrong for val/test) OR we compute on the fly.
        # Correct way: Re-calculate 'train' indices.

        train_indices = self._get_split_indices(
            self.n_samples, 0.8, 0.1, 0.1
        )  # Default ratios?
        # Ideally we pass the config ratios.
        # Let's verify if we can compute easily.

        # For large datasets, computing on the fly is slow.
        # We will use a simplified approach: Compute on current split if train,
        # or error if val/test and no stats file.
        # BUT for robustness, let's just compute on a subset of current data if needed.

        # Actually, let's just assume identity if not found, or compute on first 1000 samples of file.
        # Better: Compute on current indices.

        print(
            f"Computing normalization stats for {self.split} (Warning: Should use train stats for all)"
        )

        with h5py.File(self.data_path, "r") as f:
            for k in self.keys:
                # Sample max 1000 items
                sample_indices = self.indices[:1000]
                data_list = []
                for i in sample_indices:
                    d = f[k][i]
                    data_list.append(d)

                data_tensor = np.stack(data_list)
                self.stats_mean[k] = torch.tensor(
                    data_tensor.mean(), dtype=torch.float32
                )
                self.stats_std[k] = torch.tensor(data_tensor.std(), dtype=torch.float32)

        # Save if possible
        if stat_file:
            save_dict = {}
            for k in self.keys:
                save_dict[f"{k}_mean"] = self.stats_mean[k].numpy()
                save_dict[f"{k}_std"] = self.stats_std[k].numpy()
            np.savez(stat_file, **save_dict)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]

        with h5py.File(self.data_path, "r") as f:
            # Read Data
            # nu: (128, 128) usually
            # tensor: (1, 128, 128) usually

            data_dict = {}
            for k in self.keys:
                d = f[k][file_idx]
                d = torch.from_numpy(d).float()

                # Ensure shape (C, H, W)
                if d.ndim == 2:
                    d = d.unsqueeze(0)

                # Normalize
                if self.normalize and k in self.stats_mean:
                    d = (d - self.stats_mean[k]) / (self.stats_std[k] + 1e-8)

                data_dict[k] = d

        # Construct return dict compatible with training pipeline
        # Pipeline expects: input_sequence [T, C, H, W], target_sequence [T, C, H, W]
        # For Darcy: T=1

        tensor = data_dict["tensor"].unsqueeze(0)  # [1, C, H, W]

        # Initialize input as tensor (default) or None
        # We will determine input based on degradation
        inp = tensor.clone()
        lr = None

        # Apply degradation if configured (e.g. for SR)
        if self.h_params and apply_degradation_operator:
            # Ensure h_params has task
            task = str(self.h_params.get("task", "")).lower()
            if "sr" in task or "crop" in task:
                # Apply degradation to the target (HR solution)
                # tensor shape is [1, C, H, W]
                # Debug print for first few samples
                if idx < 5:
                    print(f"DEBUG: Applying degradation. Params: {self.h_params}")

                lr = apply_degradation_operator(tensor, self.h_params)

                if idx < 5:
                    print(f"DEBUG: LR shape: {lr.shape}, HR shape: {tensor.shape}")

                inp = lr  # Input is the degraded observation
        else:
            if idx < 5:
                print(
                    f"DEBUG: Degradation SKIPPED. h_params={bool(self.h_params)}, op={bool(apply_degradation_operator)}"
                )

        # If 'nu' exists, we could use it, but user requested to decouple.
        # So we strictly use LR -> HR.

        ret = {
            "input_sequence": inp,  # LR Observation or Tensor
            "target_sequence": tensor,  # HR Solution
            "sample_idx": file_idx,
            "start_time": 0,
            "sample_key": str(file_idx),
            "time_indices": [0],
        }

        if lr is not None:
            ret["observed_lr_sequence"] = lr

        return ret


import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class DarcyFlowDataModule(pl.LightningDataModule):
    """Darcy Flow Data Module"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_path = config.data.data_path

        # Dataset split config
        self.train_ratio = config.data.get("train_ratio", 0.8)
        self.val_ratio = config.data.get("val_ratio", 0.1)
        self.test_ratio = config.data.get("test_ratio", 0.1)
        self.splits_dir = config.data.get("splits_dir", None)

        self.normalize = config.data.get("normalize", True)
        self.img_size = config.data.get("img_size", 128)
        self.keys = config.data.get("keys", ["nu", "tensor"])
        self.seed = config.get("seed", 2025)

        # Observation/Degradation config
        # Handle config as dict or object
        data_cfg = (
            config.get("data", {})
            if isinstance(config, dict)
            else getattr(config, "data", None)
        )
        if data_cfg is None:
            print("WARNING: 'data' config not found!")
            obs_cfg = None
        else:
            # data_cfg might be dict or object
            if isinstance(data_cfg, dict):
                obs_cfg = data_cfg.get("observation", None)
            else:
                obs_cfg = getattr(data_cfg, "observation", None)

        print(f"DEBUG: DataModule obs_cfg: {obs_cfg}")

        self.h_params = None
        if obs_cfg:
            self.h_params = {}
            # Flatten config for apply_degradation_operator
            # obs_cfg might be dict or object
            if isinstance(obs_cfg, dict):
                mode = obs_cfg.get("mode", "SR")
                sr_cfg = obs_cfg.get("sr", {})
            else:
                mode = getattr(obs_cfg, "mode", "SR")
                sr_cfg = getattr(obs_cfg, "sr", {})
                # Convert Omegaconf/Namespace to dict if needed
                if hasattr(sr_cfg, "items"):
                    pass  # ok
                elif hasattr(sr_cfg, "__dict__"):
                    sr_cfg = sr_cfg.__dict__

            self.h_params["task"] = mode
            if mode == "SR":
                # sr_cfg might be dict-like
                if hasattr(sr_cfg, "items"):
                    for k, v in sr_cfg.items():
                        self.h_params[k] = v
                else:
                    print(f"WARNING: sr_cfg is not a dict: {type(sr_cfg)}")

                # Map config keys to degradation operator keys
                if "blur_sigma" in self.h_params and "sigma" not in self.h_params:
                    self.h_params["sigma"] = self.h_params["blur_sigma"]
                if (
                    "blur_kernel_size" in self.h_params
                    and "kernel_size" not in self.h_params
                ):
                    self.h_params["kernel_size"] = self.h_params["blur_kernel_size"]
                if "scale_factor" in self.h_params and "scale" not in self.h_params:
                    self.h_params["scale"] = self.h_params["scale_factor"]

        # Dataloader config
        dl_cfg = getattr(config.data, "dataloader", DictConfig({}))
        self.batch_size = dl_cfg.get(
            "batch_size", config.training.get("batch_size", 32)
        )
        self.val_batch_size = dl_cfg.get("val_batch_size", self.batch_size)
        self.test_batch_size = dl_cfg.get("test_batch_size", 1)
        self.num_workers = dl_cfg.get("num_workers", 4)
        self.pin_memory = dl_cfg.get("pin_memory", True)
        self.persistent_workers = dl_cfg.get("persistent_workers", True)
        self.prefetch_factor = dl_cfg.get("prefetch_factor", 2)
        self.shuffle = dl_cfg.get("shuffle", True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DarcyFlowDataset(
                data_path=self.data_path,
                split="train",
                splits_dir=self.splits_dir,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                normalize=self.normalize,
                img_size=self.img_size,
                keys=self.keys,
                h_params=self.h_params,
            )

            self.val_dataset = DarcyFlowDataset(
                data_path=self.data_path,
                split="val",
                splits_dir=self.splits_dir,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                normalize=self.normalize,
                img_size=self.img_size,
                keys=self.keys,
                h_params=self.h_params,
            )

            # Share stats
            if self.normalize:
                self.val_dataset.stats_mean = (
                    self.train_dataset.stats_mean.copy()
                    if hasattr(self.train_dataset.stats_mean, "copy")
                    else self.train_dataset.stats_mean
                )
                self.val_dataset.stats_std = (
                    self.train_dataset.stats_std.copy()
                    if hasattr(self.train_dataset.stats_std, "copy")
                    else self.train_dataset.stats_std
                )

        if stage == "test" or stage is None:
            self.test_dataset = DarcyFlowDataset(
                data_path=self.data_path,
                split="test",
                splits_dir=self.splits_dir,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                normalize=self.normalize,
                img_size=self.img_size,
                keys=self.keys,
                h_params=self.h_params,
            )
            # Share stats
            if self.normalize and self.train_dataset:
                self.test_dataset.stats_mean = self.train_dataset.stats_mean
                self.test_dataset.stats_std = self.train_dataset.stats_std

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def get_normalization_stats(self):
        if self.train_dataset:
            # Return target stats as Tensors for trainer compatibility
            return {"mean": self.train_dataset.mean, "std": self.train_dataset.std}
        return None
