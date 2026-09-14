"""
测试 DataConsistencyChecker：使用随机张量与退化算子参数，验证MSE(H(GT), y) < tol
"""

import torch
from tools.data_consistency_checker import DataConsistencyChecker
from ops.degradation import apply_degradation_operator


def test_dc_checker_sr_mode():
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 64, 64)
    h_params = {"task": "sr", "scale": 2, "sigma": 1.0, "kernel_size": 5, "boundary": "mirror"}
    y = apply_degradation_operator(gt, h_params)

    checker = DataConsistencyChecker(tolerance=1e-8)
    res = checker.check(gt, y, h_params)
    assert res["passed"]


def test_dc_checker_crop_mode():
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 64, 64)
    h_params = {"task": "crop", "crop_size": (32, 32), "crop_box": None, "boundary": "mirror"}
    y = apply_degradation_operator(gt, h_params)

    checker = DataConsistencyChecker(tolerance=1e-8)
    res = checker.check(gt, y, h_params)
    assert res["passed"]


def test_dc_checker_channel_and_dtype_alignment():
    torch.manual_seed(0)
    gt = torch.rand(1, 2, 64, 64)
    h_params = {"task": "sr", "scale": 2, "sigma": 1.0, "kernel_size": 5, "boundary": "mirror"}
    y = apply_degradation_operator(gt, h_params)
    # 模拟 observation 额外拼接了 mask/坐标通道 (例如 2 -> 3 通道)
    extra_channel = torch.ones(1, 1, y.shape[-2], y.shape[-1])
    y_with_mask = torch.cat([y, extra_channel], dim=1).to(torch.float64)

    checker = DataConsistencyChecker(tolerance=1e-8)
    res = checker.check(gt, y_with_mask, h_params)
    assert res["passed"]
    assert res["mse"] < 1e-8