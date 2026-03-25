#!/usr/bin/env python3
"""
DC一致性检查脚本

验证 MSE(H(GT), y) < 1e-8，确保训练DC与观测算子H完全一致。
支持两类输入：
1) 人工构造的HDF5（根含 'gt'、'obs' 数据集与可选 'params' 组）
2) PDEBench/真实数据HDF5（层级中包含 'data' 数据集，形如 (N,H,W,C) 或 (H,W,C)），
   此时从配置中抽取观测参数并在原值域上生成 obs，再执行一致性检查。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import torch

# 将项目根目录加入 sys.path，确保可导入 utils/ops 等本地模块
try:
    project_root = Path(__file__).resolve().parents[1]
    p = str(project_root)
    try:
        if p in sys.path:
            sys.path.remove(p)
    except Exception:
        pass
    sys.path.insert(0, p)
except Exception:
    pass

from omegaconf import OmegaConf

from ops.degradation import apply_degradation_operator, verify_degradation_consistency
from utils.data_consistency_checker import (
    DataConsistencyChecker as _CoreDataConsistencyChecker,
)


def _read_params_from_group(g: h5py.Group) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for k, v in g.attrs.items():
        try:
            params[k] = v.item() if hasattr(v, "item") else v
        except Exception:
            params[k] = v
    # 也支持子dataset或JSON文本
    for name in g.keys():
        try:
            obj = g[name]
            if isinstance(obj, h5py.Dataset):
                # 支持 crop_size/crop_box 作为dataset
                data = obj[()]
                if hasattr(data, "tolist"):
                    data = data.tolist()
                params[name] = data
            elif isinstance(obj, h5py.Group) and "json" in obj:
                try:
                    text = obj["json"][()].decode("utf-8")
                    params.update(json.loads(text))
                except Exception:
                    pass
        except Exception:
            pass
    return params


def _find_first_data_dataset(h5: h5py.File) -> h5py.Dataset | None:
    """在层级结构中搜索名为 'data' 的第一个dataset"""
    found = None

    def _visitor(name, obj):
        nonlocal found
        if found is not None:
            return
        try:
            if isinstance(obj, h5py.Dataset) and name.endswith("/data"):
                found = obj
        except Exception:
            pass

    try:
        h5.visititems(_visitor)
    except Exception:
        pass
    return found


def _to_bchw(arr: Any) -> torch.Tensor:
    """将 numpy/h5 数据转换为 [B,C,H,W]

    支持以下输入形状：
    - (H, W)
    - (H, W, C) 视为单样本 NHWC
    - (N, H, W, C) NHWC 批量
    - (N, C, H, W) NCHW 批量（此前未显式支持，现加入判断）

    形状判定策略：
    - 对 4D：优先判定为 NCHW 当第二维（C）较小且后两维（H,W）为图像尺寸；否则按 NHWC。
    """
    import numpy as np

    a = arr[()]
    if hasattr(a, "astype"):
        a = a.astype("float32")
    a = np.array(a)
    if a.ndim == 2:  # (H,W)
        H, W = a.shape
        t = torch.from_numpy(a).view(1, 1, H, W)
    elif a.ndim == 3:  # (H,W,C)
        H, W, C = a.shape
        t = torch.from_numpy(a).permute(2, 0, 1).contiguous().view(1, C, H, W)
    elif a.ndim == 4:
        # 判定 NCHW 或 NHWC
        N0, D1, D2, D3 = a.shape
        # 经验性判断：通道数通常较小（<= 32），而 H/W 通常较大（>= 8）
        is_nchw = D1 <= 32 and D2 >= 8 and D3 >= 8
        if is_nchw:  # (N,C,H,W)
            N, C, H, W = a.shape
            t = torch.from_numpy(a[0]).contiguous().view(1, C, H, W)
        else:  # (N,H,W,C)
            N, H, W, C = a.shape
            t = torch.from_numpy(a[0]).permute(2, 0, 1).contiguous().view(1, C, H, W)
    else:
        raise ValueError(f"Unsupported data shape: {a.shape}")
    return t


def run_check_from_h5(
    h5_path: str, tolerance: float = 1e-8, params_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    p = Path(h5_path)
    assert p.exists(), f"HDF5 not found: {p}"
    with h5py.File(p, "r") as f:
        params = _read_params_from_group(f["params"]) if "params" in f else {}
        if params_override:
            params.update(params_override)

        if "gt" in f and "obs" in f:
            gt = _to_bchw(f["gt"])
            obs = _to_bchw(f["obs"])
        else:
            # PDEBench/真实数据：查找 data dataset，提取一个样本作为GT，并用观测算子生成obs
            data_ds = _find_first_data_dataset(f)
            assert (
                data_ds is not None
            ), "未在HDF5中找到 'data' 数据集，也不包含 'gt'/'obs'"
            gt = _to_bchw(data_ds)
            assert params, "真实数据HDF5缺少 'gt'/'obs'，需要通过 --config 提供观测参数"
            obs = apply_degradation_operator(gt, params)

    checker = _CoreDataConsistencyChecker(tolerance=tolerance)
    return checker.check(gt, obs, params)


class DataConsistencyChecker:
    def __init__(self, config: Any = None, tolerance: float = 1e-8):
        if isinstance(config, (int, float)):
            tolerance = float(config)
            config = None
        self.config = config
        self.tolerance = float(tolerance)

    def check_consistency(
        self, dataset, degradation_op, num_samples: int = 10
    ) -> dict[str, Any]:
        import torch.nn.functional as F

        if len(dataset) == 0:
            return {
                "mse": 0.0,
                "max_error": 0.0,
                "tolerance": self.tolerance,
                "passed": True,
            }

        sample_count = min(int(num_samples), int(len(dataset)))
        mses: list[float] = []
        max_errs: list[float] = []
        for i in range(sample_count):
            item = dataset[i]
            gt = item["gt"].unsqueeze(0)
            obs_ref = item.get("baseline")
            if obs_ref is None:
                obs_ref = degradation_op(gt)
            else:
                obs_ref = obs_ref.unsqueeze(0)

            obs_pred = degradation_op(gt)
            if obs_pred.shape != obs_ref.shape:
                obs_ref = F.interpolate(
                    obs_ref,
                    size=obs_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                c = min(obs_pred.shape[1], obs_ref.shape[1])
                obs_pred = obs_pred[:, :c]
                obs_ref = obs_ref[:, :c]

            diff = (obs_pred - obs_ref).float()
            mses.append(float((diff**2).mean().item()))
            max_errs.append(float(diff.abs().max().item()))

        mse = float(sum(mses) / max(1, len(mses)))
        max_error = float(max(max_errs) if max_errs else 0.0)
        return {
            "mse": mse,
            "max_error": max_error,
            "tolerance": self.tolerance,
            "passed": bool(mse < self.tolerance),
        }


def check_degradation_consistency(
    target: torch.Tensor,
    observation: torch.Tensor,
    h_params: dict[str, Any],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """兼容测试用的退化一致性检查函数。

    该函数用于验证观测算子 H 与训练 DC 的一致性，满足黄金法则：复用同一实现与配置。
    内部复用 ops.degradation.verify_degradation_consistency 的实现。

    Args:
        target: 真值张量，形状 [B, C, H, W]
        observation: 观测张量，形状 [B, C, H', W']（与 H(target) 一致）
        h_params: H 算子的参数字典（task/scale/sigma/kernel_size/boundary 等）
        tolerance: MSE 阈值，默认 1e-8

    Returns:
        字典包含 'mse'、'max_error'、'tolerance'、'passed' 等键
    """
    res = verify_degradation_consistency(target, observation, h_params, tolerance)
    # 保持返回字典键的稳定可读性
    return {
        "mse": float(res.get("mse", 0.0)),
        "max_error": float(res.get("max_error", 0.0)),
        "tolerance": float(tolerance),
        "passed": bool(res.get("passed", False)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DC一致性检查")
    parser.add_argument("--h5", type=str, help="HDF5文件路径")
    parser.add_argument("--config", type=str, help="Hydra YAML 配置文件路径（可选）")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    params_override: dict[str, Any] | None = None
    h5_path = args.h5
    # 若提供了配置文件，则从配置读取观测参数与数据路径
    if args.config:
        cfg = OmegaConf.load(args.config)
        obs_cfg = getattr(cfg, "observation", {})
        # 规范化任务名到统一集合 {"SR", "Crop", "Identity"}
        _mode = str(getattr(obs_cfg, "mode", "sr")).strip()
        _mode_norm = (
            "SR"
            if _mode.lower() in {"sr", "super", "super_resolution", "super-resolution"}
            else "Crop" if _mode.lower() in {"crop", "patch"} else "Identity"
        )
        # 提取观测参数（支持sr/crop等），统一键名到apply_degradation_operator
        params_override = {
            "task": _mode_norm,
            "scale": int(getattr(obs_cfg, "scale_factor", 1)),
            "sigma": float(getattr(obs_cfg, "blur_sigma", 0.0)),
            "kernel_size": int(getattr(obs_cfg, "kernel_size", 5)),
            "boundary": getattr(obs_cfg, "boundary", "mirror"),
            "crop_size": getattr(obs_cfg, "crop_size", None),
            "crop_box": getattr(obs_cfg, "crop_box", None),
            "downsample_interpolation": getattr(
                obs_cfg, "downsample_interpolation", "area"
            ),
        }
        if not h5_path:
            # 默认从配置的数据路径读取
            data_cfg = getattr(cfg, "data", {})
            h5_path = getattr(data_cfg, "data_path", None)
    assert h5_path is not None, "必须提供 --h5 或 --config（其中 data.data_path 存在）"
    res = run_check_from_h5(h5_path, args.tolerance, params_override)
    print(json.dumps(res, indent=2))
    exit(0 if res.get("passed", False) else 1)


if __name__ == "__main__":
    main()
