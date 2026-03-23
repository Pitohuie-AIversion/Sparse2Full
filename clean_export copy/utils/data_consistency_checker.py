"""
DataConsistencyChecker

确保训练中的数据一致性损失 DC 与观测算子 H 完全一致。
复用 ops.degradation.apply_degradation_operator 的实现，不重复逻辑。
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from ops.degradation import apply_degradation_operator


class DataConsistencyChecker:
    """数据一致性检查器

    使用统一的 H 算子实现，验证 MSE(H(GT), y) < tol。
    """

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = float(tolerance)

    @torch.no_grad()
    def check(self, target: torch.Tensor, observation: torch.Tensor, h_params: Dict[str, Any]) -> Dict[str, Any]:
        """对单个样本进行一致性检查

        Args:
            target: 真值张量 [B, C, H, W]
            observation: 观测张量 [B, C, H', W']
            h_params: H 算子参数字典

        Returns:
            结果字典，包含 mse、max_error、passed 等键
        """
        # 应用 H 到 GT
        h_target = apply_degradation_operator(target, h_params)

        # 对齐尺寸
        if h_target.shape != observation.shape:
            # 通道对齐
            if observation.shape[1] != h_target.shape[1]:
                c = min(observation.shape[1], h_target.shape[1])
                observation = observation[:, :c]
                h_target = h_target[:, :c]
            # 空间尺寸对齐
            if observation.shape[-2:] != h_target.shape[-2:]:
                observation = F.interpolate(observation, size=h_target.shape[-2:], mode='bilinear', align_corners=False)

        mse = F.mse_loss(h_target, observation).item()
        max_error = torch.max(torch.abs(h_target - observation)).item()
        passed = bool(mse < self.tolerance)

        return {
            'mse': mse,
            'max_error': max_error,
            'tolerance': self.tolerance,
            'passed': passed,
            'h_target_shape': list(h_target.shape),
            'observation_shape': list(observation.shape),
        }