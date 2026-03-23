#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dump model architecture summary by running a dummy forward + forward hooks.

Outputs:
- JSON: modules in execution order with output shapes + param counts
- MD : a markdown table of the same info (good for paper diagram drafting)

Usage (run from project root):
python tools/analysis/dump_arch_summary.py \
  --config thesis_paper/configs/ar_paper_aligned_sr4_2D_diff_react_NA_NA.yaml \
  --model UformerLite \
  --device cpu \
  --out_json runs/arch_uformerlite.json \
  --out_md runs/arch_uformerlite.md \
  --max_depth 2
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf


def _to_plain(obj: Any) -> Any:
    """OmegaConf/DictConfig -> python dict/list."""
    try:
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj


def _extract_shape(x: Any) -> Any:
    """Robust shape extractor for tensor / (list|tuple|dict) of tensors."""
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.append(_extract_shape(item))
        return out
    if isinstance(x, dict):
        return {k: _extract_shape(v) for k, v in x.items()}
    return str(type(x))


def _count_params_unique(module: torch.nn.Module) -> Tuple[int, int]:
    """Return (params_total, params_trainable) counting unique parameter tensors."""
    seen = set()
    total = 0
    trainable = 0
    for p in module.parameters(recurse=True):
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def _looks_useful(name: str, module: torch.nn.Module, keywords: List[str]) -> bool:
    """Heuristic: keep stage-like modules and important blocks."""
    lname = name.lower()
    if any(k in lname for k in keywords):
        return True
    # also keep if module itself has direct params (some top-level blocks)
    direct_params = sum(p.numel() for p in module.parameters(recurse=False))
    if direct_params > 0:
        return True
    return False


@dataclass
class HookRecord:
    order: int
    name: str
    module_type: str
    depth: int
    params: int
    params_trainable: int
    out_shape: Any


def build_model_from_project(cfg, model_name_override: Optional[str] = None):
    """
    Reuse your project's existing loaders (same as train_real_data_ar.py).
    Falls back layer-by-layer to improve robustness.
    """
    model_cfg = _to_plain(getattr(cfg, "model", {})) or {}
    model_name = model_name_override or model_cfg.get("name", None) or getattr(cfg.model, "name", None)
    if model_name is None:
        raise RuntimeError("Cannot find model name in config. Expect cfg.model.name")

    # kwargs: prefer cfg.model keys except 'name'
    kwargs = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    kwargs.pop("name", None)

    # ensure common keys exist
    data_cfg = _to_plain(getattr(cfg, "data", {})) or {}
    img_size = kwargs.get("img_size", None) or data_cfg.get("img_size", 128)
    in_ch = kwargs.get("in_channels", None) or data_cfg.get("input_channels", 1)
    out_ch = kwargs.get("out_channels", None) or data_cfg.get("target_channels", 1)
    kwargs.setdefault("img_size", int(img_size))
    kwargs.setdefault("in_channels", int(in_ch))
    kwargs.setdefault("out_channels", int(out_ch))

    errors = []

    # Try improved -> enhanced -> original (matches your training script import stack)
    try:
        from tools.training.model_loader_improved import create_improved_model
        m = create_improved_model(model_name, cfg, **kwargs)
        return m, "improved_loader", kwargs
    except Exception as e:
        errors.append(f"improved_loader: {e}")

    try:
        from tools.training.model_loader_enhanced import create_enhanced_model
        m = create_enhanced_model(model_name, cfg, **kwargs)
        return m, "enhanced_loader", kwargs
    except Exception as e:
        errors.append(f"enhanced_loader: {e}")

    try:
        from tools.training.model_loader import create_model_with_loader
        m = create_model_with_loader(model_name, cfg, **kwargs)
        return m, "original_loader", kwargs
    except Exception as e:
        errors.append(f"original_loader: {e}")

    raise RuntimeError("All model loaders failed:\n" + "\n".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--model", default=None, type=str, help="Override cfg.model.name (e.g., UformerLite)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Run dummy forward on cpu/cuda")
    ap.add_argument("--input_shape", nargs=4, type=int, default=None, metavar=("B", "C", "H", "W"))
    ap.add_argument("--max_depth", type=int, default=2, help="Max module depth to keep in summary")
    ap.add_argument("--keywords", type=str, default="enc,dec,down,up,stage,block,attn,fft,fno,patch,head,embed,bottleneck,skip",
                    help="Comma separated keywords to keep")
    ap.add_argument("--out_json", required=True, type=str)
    ap.add_argument("--out_md", required=True, type=str)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    cfg_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    # override model name in cfg for consistency (optional)
    if args.model:
        try:
            cfg.model.name = args.model
        except Exception:
            pass

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model, creation_method, used_kwargs = build_model_from_project(cfg, model_name_override=args.model)
    model = model.to(device)
    model.eval()

    # infer input shape
    if args.input_shape is not None:
        B, C, H, W = args.input_shape
    else:
        data_cfg = _to_plain(getattr(cfg, "data", {})) or {}
        model_cfg = _to_plain(getattr(cfg, "model", {})) or {}
        H = int(model_cfg.get("img_size", data_cfg.get("img_size", 128)))
        W = H
        C = int(model_cfg.get("in_channels", data_cfg.get("input_channels", 1)))
        B = 1

    x = torch.zeros((B, C, H, W), device=device)

    # hook selection
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    records: List[HookRecord] = []
    order = 0
    seen_names = set()

    def hook_fn(name: str):
        def _fn(mod, inp, out):
            nonlocal order
            if name in seen_names:
                return
            depth = _module_depth(name)
            if depth > args.max_depth:
                return
            if not _looks_useful(name, mod, keywords) and depth > 1:
                return
            p_all, p_tr = _count_params_unique(mod)
            rec = HookRecord(
                order=order,
                name=name,
                module_type=type(mod).__name__,
                depth=depth,
                params=p_all,
                params_trainable=p_tr,
                out_shape=_extract_shape(out),
            )
            records.append(rec)
            seen_names.add(name)
            order += 1
        return _fn

    hooks = []
    # Always keep top-level children (depth==1), and keyword-matching modules up to max_depth
    for name, mod in model.named_modules():
        if name == "":
            continue
        depth = _module_depth(name)
        if depth > args.max_depth:
            continue
        if depth == 1 or _looks_useful(name, mod, keywords):
            try:
                hooks.append(mod.register_forward_hook(hook_fn(name)))
            except Exception:
                pass

    # run dummy forward
    t0 = time.time()
    with torch.no_grad():
        _ = model(x)
    dt = time.time() - t0

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_json = (project_root / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_md = (project_root / args.out_md).resolve() if not Path(args.out_md).is_absolute() else Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "config": str(cfg_path),
            "model_name": getattr(cfg.model, "name", None),
            "creation_method": creation_method,
            "used_kwargs": used_kwargs,
            "device": str(device),
            "dummy_input_shape": [B, C, H, W],
            "dummy_forward_sec": dt,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        },
        "modules": [rec.__dict__ for rec in sorted(records, key=lambda r: r.order)],
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown table
    lines = []
    lines.append(f"# Model Arch Summary: {payload['meta'].get('model_name')}\n")
    lines.append(f"- config: `{payload['meta']['config']}`")
    lines.append(f"- creation_method: `{payload['meta']['creation_method']}`")
    lines.append(f"- device: `{payload['meta']['device']}`")
    lines.append(f"- dummy_input_shape: `{payload['meta']['dummy_input_shape']}`")
    lines.append(f"- dummy_forward_sec: `{payload['meta']['dummy_forward_sec']:.6f}`")
    lines.append(f"- total_params: `{payload['meta']['total_params']}`")
    lines.append(f"- trainable_params: `{payload['meta']['trainable_params']}`\n")

    lines.append("| order | name | type | depth | params | trainable | out_shape |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
    for rec in sorted(records, key=lambda r: r.order):
        lines.append(
            f"| {rec.order} | `{rec.name}` | `{rec.module_type}` | {rec.depth} | {rec.params} | {rec.params_trainable} | `{rec.out_shape}` |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()
