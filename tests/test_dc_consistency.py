#!/usr/bin/env python3
"""
数据一致性检查单测

验证 DataConsistencyChecker 与 ops.degradation.H 一致，符合黄金法则。
"""

import torch
import pytest

from utils.data_consistency_checker import DataConsistencyChecker
from ops.degradation import apply_degradation_operator


@pytest.mark.parametrize("task", ["sr", "crop"]) 
def test_dc_checker_basic(task):
    torch.manual_seed(0)
    target = torch.randn(2, 1, 64, 64)

    if task == "sr":
        h_params = {
            'task': 'sr',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror',
        }
    else:
        h_params = {
            'task': 'crop',
            'crop_size': (32, 32),
            'crop_box': None,
            'boundary': 'mirror',
        }

    # 生成观测
    observation = apply_degradation_operator(target, h_params)

    checker = DataConsistencyChecker(tolerance=1e-8)
    res = checker.check(target, observation, h_params)

    assert res['passed'], f"DC一致性失败: mse={res['mse']} tol={res['tolerance']}"
    assert res['mse'] < 1e-8