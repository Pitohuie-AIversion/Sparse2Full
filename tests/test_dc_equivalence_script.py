#!/usr/bin/env python3
"""
脚本级别DC一致性检查单测

生成临时HDF5，写入GT/OBS和参数组，调用脚本的run_check_from_h5验证通过。
"""

import tempfile
from pathlib import Path
import h5py
import torch
import numpy as np

from tools.check_dc_equivalence import run_check_from_h5
from ops.degradation import apply_degradation_operator


def test_dc_equivalence_script_sr():
    H, W = 64, 64
    gt = torch.randn(1, 1, H, W)
    params = {
        'task': 'sr',
        'scale': 2,
        'sigma': 1.0,
        'kernel_size': 5,
        'boundary': 'mirror',
    }
    obs = apply_degradation_operator(gt, params)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case_sr.h5"
        with h5py.File(p, 'w') as f:
            f.create_dataset('gt', data=gt.numpy())
            f.create_dataset('obs', data=obs.numpy())
            g = f.create_group('params')
            for k, v in params.items():
                g.attrs[k] = v

        res = run_check_from_h5(str(p))
        assert res['passed'], f"脚本SR一致性失败: {res}"
        assert res['mse'] < 1e-8


def test_dc_equivalence_script_crop():
    H, W = 64, 64
    gt = torch.randn(1, 1, H, W)
    params = {
        'task': 'crop',
        'crop_size': (32, 32),
        'crop_box': None,
        'boundary': 'mirror',
    }
    obs = apply_degradation_operator(gt, params)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case_crop.h5"
        with h5py.File(p, 'w') as f:
            f.create_dataset('gt', data=gt.numpy())
            f.create_dataset('obs', data=obs.numpy())
            g = f.create_group('params')
            # 将复杂类型转为基本类型属性
            g.attrs['task'] = params['task']
            g.attrs['boundary'] = params['boundary']
            # 以dataset形式写入 crop_size
            f.create_dataset('params/crop_size', data=np.array(params['crop_size'], dtype=np.int32))

        res = run_check_from_h5(str(p))
        assert res['passed'], f"脚本Crop一致性失败: {res}"
        assert res['mse'] < 1e-8