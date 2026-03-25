from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from ops.degradation import apply_degradation_operator


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _read_split_ids(splits_dir: str | Path | None, split: str) -> list[str] | None:
    if splits_dir is None:
        return None
    split_path = _as_path(splits_dir) / f"{split}.txt"
    if not split_path.exists():
        return None
    ids: list[str] = []
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def _infer_h5_case_ids(h5: Any) -> list[str]:
    if "tensor" in h5:
        n = int(h5["tensor"].shape[0])
        return [str(i) for i in range(n)]
    return sorted([k for k in h5.keys() if isinstance(k, str)])


def _load_case_tensor(h5: Any, case_id: str, keys: Sequence[str]) -> torch.Tensor:
    if "tensor" in h5:
        x = h5["tensor"][int(case_id)]
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            pass
        elif arr.ndim == 4:
            arr = arr[-1]
        else:
            raise ValueError(f"Unsupported tensor ndim: {arr.ndim}")
        return torch.from_numpy(arr.astype(np.float32))

    g = h5[case_id]
    chans: list[np.ndarray] = []
    for k in keys:
        if k not in g:
            raise KeyError(f"Missing key '{k}' in case '{case_id}'")
        arr = np.asarray(g[k]).astype(np.float32)
        if arr.ndim == 2:
            chans.append(arr)
        elif arr.ndim == 3:
            chans.append(arr[-1])
        elif arr.ndim == 4:
            # (T, H, W, C) -> take last time step and first channel
            # This is a heuristic for RealDiffusionReaction which has (T, H, W, 2)
            # We assume we want the first component (u)
            chans.append(arr[-1, ..., 0])
        else:
            raise ValueError(
                f"Unsupported dataset shape for {case_id}/{k}: {arr.shape}"
            )
    stacked = np.stack(chans, axis=0)
    return torch.from_numpy(stacked)


def _make_coords(h: int, w: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device)
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device)
    grid_x = xs[None, :].repeat(h, 1)
    grid_y = ys[:, None].repeat(1, w)
    return torch.stack([grid_x, grid_y], dim=0)


@dataclass(frozen=True)
class _ObsCfg:
    mode: str
    scale: int = 1
    sigma: float = 0.0
    kernel_size: int = 1
    noise_std: float = 0.0
    crop_size: tuple[int, int] = (0, 0)
    patch_align: int = 1
    center_sampler: str = "uniform"
    boundary: str = "mirror"


def _parse_obs_cfg(cfg: Any) -> _ObsCfg:
    if cfg is None:
        return _ObsCfg(mode="SR")

    mode = str(getattr(cfg, "mode", "SR"))
    sr = getattr(cfg, "sr", None)
    crop = getattr(cfg, "crop", None)
    if mode.lower() == "sr":
        scale = (
            int(getattr(sr, "scale_factor", getattr(cfg, "scale", 1)))
            if sr is not None
            else int(getattr(cfg, "scale", 1))
        )
        sigma = (
            float(getattr(sr, "blur_sigma", getattr(cfg, "sigma", 0.0)))
            if sr is not None
            else float(getattr(cfg, "sigma", 0.0))
        )
        kernel = (
            int(
                getattr(
                    sr,
                    "blur_kernel_size",
                    getattr(cfg, "blur_kernel", getattr(cfg, "kernel_size", 1)),
                )
            )
            if sr is not None
            else int(getattr(cfg, "blur_kernel", getattr(cfg, "kernel_size", 1)))
        )
        boundary = (
            str(getattr(sr, "boundary_mode", getattr(cfg, "boundary", "mirror")))
            if sr is not None
            else str(getattr(cfg, "boundary", "mirror"))
        )
        noise_std = (
            float(getattr(sr, "noise_std", getattr(cfg, "noise_std", 0.0)))
            if sr is not None
            else float(getattr(cfg, "noise_std", 0.0))
        )
        return _ObsCfg(
            mode="SR",
            scale=scale,
            sigma=sigma,
            kernel_size=kernel,
            noise_std=noise_std,
            boundary=boundary,
        )

    if mode.lower() == "crop":
        crop_size = getattr(crop, "crop_size", getattr(cfg, "crop_size", (0, 0)))
        if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            cs = (int(crop_size[0]), int(crop_size[1]))
        else:
            cs = (0, 0)
        patch_align = (
            int(getattr(crop, "patch_align", getattr(cfg, "patch_align", 1)))
            if crop is not None
            else int(getattr(cfg, "patch_align", 1))
        )
        center_sampler = (
            str(
                getattr(
                    crop, "crop_strategy", getattr(cfg, "center_sampler", "uniform")
                )
            )
            if crop is not None
            else str(getattr(cfg, "center_sampler", "uniform"))
        )
        boundary = (
            str(getattr(crop, "boundary_mode", getattr(cfg, "boundary", "mirror")))
            if crop is not None
            else str(getattr(cfg, "boundary", "mirror"))
        )
        return _ObsCfg(
            mode="Crop",
            crop_size=cs,
            patch_align=patch_align,
            center_sampler=center_sampler,
            boundary=boundary,
        )

    return _ObsCfg(mode=mode)


class PDEBenchBase(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        keys: Sequence[str],
        split: str = "train",
        splits_dir: str | Path | None = None,
        normalize: bool = False,
        image_size: int | None = None,
    ) -> None:
        self.data_path = str(data_path)
        self.keys = list(keys)
        self.split = split
        self.splits_dir = str(splits_dir) if splits_dir is not None else None
        self.normalize = bool(normalize)
        self.image_size = int(image_size) if image_size is not None else None

        import h5py

        with h5py.File(self.data_path, "r") as f:
            all_ids = _infer_h5_case_ids(f)

        split_ids = _read_split_ids(self.splits_dir, split)
        if split_ids is None:
            self.case_ids = all_ids
        else:
            valid = set(all_ids)
            self.case_ids = [cid for cid in split_ids if cid in valid]

        self.norm_stats: dict[str, float] | None = None
        if self.normalize:
            self.norm_stats = self._get_or_compute_norm_stats()

    def __len__(self) -> int:
        return len(self.case_ids)

    def _get_or_compute_norm_stats(self) -> dict[str, float]:
        if self.splits_dir is None:
            raise ValueError("splits_dir must be provided when normalize=True")
        stats_path = _as_path(self.splits_dir) / "norm_stat.npz"
        if stats_path.exists():
            data = np.load(stats_path)
            return {k: float(data[k]) for k in data.files}

        train_ids = _read_split_ids(self.splits_dir, "train")
        if train_ids is None:
            train_ids = self.case_ids

        import h5py

        sums = dict.fromkeys(self.keys, 0.0)
        sumsq = dict.fromkeys(self.keys, 0.0)
        counts = dict.fromkeys(self.keys, 0)
        with h5py.File(self.data_path, "r") as f:
            for cid in train_ids:
                if cid not in self.case_ids and "tensor" not in f:
                    continue
                x = _load_case_tensor(f, cid, self.keys)
                for i, k in enumerate(self.keys):
                    v = x[i].float()
                    sums[k] += float(v.sum().item())
                    sumsq[k] += float((v * v).sum().item())
                    counts[k] += int(v.numel())
        out: dict[str, float] = {}
        for k in self.keys:
            c = max(counts[k], 1)
            mean = sums[k] / float(c)
            var = max(sumsq[k] / float(c) - mean * mean, 1e-12)
            std = float(np.sqrt(var))
            out[f"{k}_mean"] = float(mean)
            out[f"{k}_std"] = float(std)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, **out)
        return out

    def _normalize_data(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if not self.normalize or self.norm_stats is None:
            return x
        m = float(self.norm_stats.get(f"{key}_mean", 0.0))
        s = float(self.norm_stats.get(f"{key}_std", 1.0))
        s = s if s > 0 else 1.0
        return (x - m) / s

    def _denormalize_data(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if not self.normalize or self.norm_stats is None:
            return x
        m = float(self.norm_stats.get(f"{key}_mean", 0.0))
        s = float(self.norm_stats.get(f"{key}_std", 1.0))
        s = s if s > 0 else 1.0
        return x * s + m

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cid = self.case_ids[idx]
        import h5py

        with h5py.File(self.data_path, "r") as f:
            x = _load_case_tensor(f, cid, self.keys)

        if self.image_size is not None:
            _, h, w = x.shape
            if h != self.image_size or w != self.image_size:
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        for i, k in enumerate(self.keys):
            x[i] = self._normalize_data(x[i], k)

        return {"target": x, "case_id": cid, "keys": list(self.keys)}


class PDEBenchSR(PDEBenchBase):
    def __init__(
        self,
        data_path: str | Path,
        keys: Sequence[str],
        scale: int = 4,
        sigma: float = 1.0,
        blur_kernel: int = 5,
        boundary: str = "mirror",
        noise_std: float = 0.0,
        split: str = "train",
        splits_dir: str | Path | None = None,
        normalize: bool = False,
        image_size: int | None = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            keys=keys,
            split=split,
            splits_dir=splits_dir,
            normalize=normalize,
            image_size=image_size,
        )
        self.scale = int(scale)
        self.sigma = float(sigma)
        self.blur_kernel = int(blur_kernel)
        self.boundary = str(boundary)
        self.noise_std = float(noise_std)
        self.h_params = {
            "task": "SR",
            "scale": self.scale,
            "sigma": self.sigma,
            "kernel_size": self.blur_kernel,
            "boundary": self.boundary,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        out = super().__getitem__(idx)
        target: torch.Tensor = out["target"]
        c, h, w = target.shape

        lr = apply_degradation_operator(target.unsqueeze(0), self.h_params).squeeze(0)
        if self.noise_std > 0:
            lr = lr + torch.randn_like(lr) * self.noise_std

        baseline = F.interpolate(
            lr.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        coords = _make_coords(h, w, device=target.device)
        mask = torch.ones(1, h, w, device=target.device, dtype=target.dtype)

        out.update(
            {
                "baseline": baseline,
                "coords": coords,
                "mask": mask,
                "h_params": dict(self.h_params),
                "lr_observation": lr,
            }
        )
        return out


class PDEBenchCrop(PDEBenchBase):
    def __init__(
        self,
        data_path: str | Path,
        keys: Sequence[str],
        crop_size: tuple[int, int] = (64, 64),
        patch_align: int = 8,
        center_sampler: str = "mixed",
        boundary: str = "mirror",
        split: str = "train",
        splits_dir: str | Path | None = None,
        normalize: bool = False,
        image_size: int | None = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            keys=keys,
            split=split,
            splits_dir=splits_dir,
            normalize=normalize,
            image_size=image_size,
        )
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.patch_align = int(patch_align)
        self.center_sampler = str(center_sampler)
        self.boundary = str(boundary)
        self.crop_h = (self.crop_size[0] // self.patch_align) * self.patch_align
        self.crop_w = (self.crop_size[1] // self.patch_align) * self.patch_align
        self.h_params = {
            "task": "Crop",
            "crop_size": (self.crop_h, self.crop_w),
            "patch_align": self.patch_align,
            "boundary": self.boundary,
        }

    def _sample_crop_box(self, h: int, w: int) -> tuple[int, int, int, int]:
        ch, cw = self.crop_h, self.crop_w
        if ch <= 0 or cw <= 0:
            return (0, 0, w, h)
        if ch >= h and cw >= w:
            return (0, 0, w, h)
        if self.center_sampler == "boundary":
            x1 = 0
            y1 = 0
        elif self.center_sampler == "gradient":
            x1 = max((w - cw) // 2, 0)
            y1 = max((h - ch) // 2, 0)
        else:
            x1 = int(torch.randint(low=0, high=max(w - cw + 1, 1), size=(1,)).item())
            y1 = int(torch.randint(low=0, high=max(h - ch + 1, 1), size=(1,)).item())
        x2 = min(x1 + cw, w)
        y2 = min(y1 + ch, h)
        return (x1, y1, x2, y2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        out = super().__getitem__(idx)
        target: torch.Tensor = out["target"]
        c, h, w = target.shape

        x1, y1, x2, y2 = self._sample_crop_box(h, w)
        mask = torch.zeros(1, h, w, device=target.device, dtype=target.dtype)
        mask[:, y1:y2, x1:x2] = 1.0

        baseline = torch.zeros_like(target)
        baseline[:, y1:y2, x1:x2] = target[:, y1:y2, x1:x2]

        coords = _make_coords(h, w, device=target.device)
        h_params = dict(self.h_params)
        h_params["crop_box"] = (x1, y1, x2, y2)

        out.update(
            {"baseline": baseline, "coords": coords, "mask": mask, "h_params": h_params}
        )
        return out


class PDEBenchDataModule:
    def __init__(self, config: DictConfig | dict[str, Any]):
        self.config = config

        self.data_path: str
        if isinstance(config, DictConfig):
            data_path = str(config.get("data_path"))
            dataset_name = config.get("dataset_name")
        else:
            data_path = str(config.get("data_path"))
            dataset_name = config.get("dataset_name")

        if dataset_name is not None and _as_path(data_path).is_dir():
            self.data_path = str(_as_path(data_path) / str(dataset_name))
        else:
            self.data_path = data_path

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config
        # Use .get() or item access to avoid conflict with DictConfig methods like .keys()
        if isinstance(cfg, DictConfig):
            keys = list(cfg.get("keys", []))
            normalize = bool(cfg.get("normalize", False))
            image_size = cfg.get("image_size", None)
            splits_dir = cfg.get("splits_dir", None)
            obs_cfg_raw = cfg.get("observation")
        else:
            keys = list(cfg.get("keys", []))
            normalize = bool(cfg.get("normalize", False))
            image_size = cfg.get("image_size", None)
            splits_dir = cfg.get("splits_dir", None)
            obs_cfg_raw = cfg.get("observation")

        obs_cfg = _parse_obs_cfg(obs_cfg_raw)
        mode = obs_cfg.mode

        if str(mode).lower() == "sr":
            ds_ctor = lambda split: PDEBenchSR(
                data_path=self.data_path,
                keys=keys,
                split=split,
                splits_dir=splits_dir,
                normalize=normalize,
                image_size=image_size,
                scale=obs_cfg.scale,
                sigma=obs_cfg.sigma,
                blur_kernel=obs_cfg.kernel_size,
                boundary=obs_cfg.boundary,
                noise_std=obs_cfg.noise_std,
            )
        elif str(mode).lower() == "crop":
            ds_ctor = lambda split: PDEBenchCrop(
                data_path=self.data_path,
                keys=keys,
                split=split,
                splits_dir=splits_dir,
                normalize=normalize,
                image_size=image_size,
                crop_size=obs_cfg.crop_size,
                patch_align=obs_cfg.patch_align,
                center_sampler=obs_cfg.center_sampler,
                boundary=obs_cfg.boundary,
            )
        else:
            ds_ctor = lambda split: PDEBenchBase(
                data_path=self.data_path,
                keys=keys,
                split=split,
                splits_dir=splits_dir,
                normalize=normalize,
                image_size=image_size,
            )

        train_split = (
            getattr(cfg, "train_split", "train")
            if isinstance(cfg, DictConfig)
            else cfg.get("train_split", "train")
        )
        val_split = (
            getattr(cfg, "val_split", "val")
            if isinstance(cfg, DictConfig)
            else cfg.get("val_split", "val")
        )
        test_split = (
            getattr(cfg, "test_split", "test")
            if isinstance(cfg, DictConfig)
            else cfg.get("test_split", "test")
        )

        self.train_dataset = ds_ctor(str(train_split))
        self.val_dataset = ds_ctor(str(val_split))
        self.test_dataset = ds_ctor(str(test_split))

    def _dl_cfg(self) -> dict[str, Any]:
        cfg = self.config
        if isinstance(cfg, DictConfig):
            dl = cfg.get("dataloader", {})
            return {
                "batch_size": int(dl.get("batch_size", 1)),
                "num_workers": int(dl.get("num_workers", 0)),
                "pin_memory": bool(dl.get("pin_memory", False)),
                "persistent_workers": bool(dl.get("persistent_workers", False)),
            }
        dl = cfg.get("dataloader", {})
        return {
            "batch_size": int(dl.get("batch_size", 1)),
            "num_workers": int(dl.get("num_workers", 0)),
            "pin_memory": bool(dl.get("pin_memory", False)),
            "persistent_workers": bool(dl.get("persistent_workers", False)),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
        cfg = self._dl_cfg()
        return DataLoader(
            self.train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["persistent_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None
        cfg = self._dl_cfg()
        return DataLoader(
            self.val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["persistent_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        assert self.test_dataset is not None
        cfg = self._dl_cfg()
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            persistent_workers=cfg["persistent_workers"],
        )

    def get_normalization_stats(self) -> dict[str, float] | None:
        if self.train_dataset is None:
            self.setup()
        if hasattr(self.train_dataset, "norm_stats"):
            return self.train_dataset.norm_stats
        return None
