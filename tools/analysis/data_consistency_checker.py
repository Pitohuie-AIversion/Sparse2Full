"""
DataConsistencyChecker: 验证H与DC一致性，复用ops.degradation.apply_degradation_operator

遵循黄金法则：观测算子 H 与训练 DC 必须复用同一实现与配置。
本工具类用于在训练/评估期间进行快速一致性抽检，以及CI脚本的复用。
"""

from typing import Dict, Optional
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from ops.degradation import apply_degradation_operator


class DataConsistencyChecker:
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = float(tolerance)

    @torch.no_grad()
    def check(self, gt_orig: torch.Tensor, observation: torch.Tensor, h_params: Dict) -> Dict:
        """检查单个样本的一致性：MSE(H(GT), y) < tol

        Args:
            gt_orig: 原值域GT [B,C,H,W]
            observation: 观测 y [B,C,h,w]
            h_params: H算子参数
        Returns:
            结果字典，包含mse, max_err, passed
        """
        # 应用H到GT
        h_gt = apply_degradation_operator(gt_orig, h_params)

        # 尺寸对齐
        if h_gt.shape[-2:] != observation.shape[-2:]:
            observation = F.interpolate(observation, size=h_gt.shape[-2:], mode='area')

        mse = torch.mean((h_gt - observation) ** 2).item()
        max_err = torch.max(torch.abs(h_gt - observation)).item()
        passed = bool(mse < self.tolerance)
        return {"mse": float(mse), "max_err": float(max_err), "passed": passed}

    def summarize(self, results: Dict[str, Dict]) -> Dict:
        """汇总多案例结果"""
        values = [r["mse"] for r in results.values()] if results else []
        maxes = [r["max_err"] for r in results.values()] if results else []
        passed = sum(1 for r in results.values() if r["passed"]) if results else 0
        failed = (len(results) - passed) if results else 0
        import numpy as np
        return {
            "total": int(len(results)),
            "passed": int(passed),
            "failed": int(failed),
            "mse_mean": float(np.mean(values)) if values else float("inf"),
            "mse_max": float(np.max(values)) if values else float("inf"),
            "max_error_mean": float(np.mean(maxes)) if maxes else float("inf"),
            "threshold": float(self.tolerance),
        }

    def save_report(self, summary: Dict, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)