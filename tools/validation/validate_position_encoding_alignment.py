#!/usr/bin/env python3
import json
import numpy as np
import torch
from pathlib import Path

def pixel_center_grid(h: int, w: int):
    ys = (2.0 * (torch.arange(h, dtype=torch.float32) + 0.5) / float(h)) - 1.0
    xs = (2.0 * (torch.arange(w, dtype=torch.float32) + 0.5) / float(w)) - 1.0
    yy = ys.view(-1, 1).expand(h, w)
    xx = xs.view(1, -1).expand(h, w)
    return torch.stack([xx, yy], dim=0)  # [2,H,W]

def downsample_block_center(hr_coords: torch.Tensor, scale: int):
    # 使用面积权重的块中心（等价 INTER_AREA），块中心像素索引 = floor((j+0.5)/s) 的反映射
    h, w = hr_coords.shape[-2:]
    assert h % scale == 0 and w % scale == 0
    H_lr, W_lr = h // scale, w // scale
    grid_y = torch.arange(H_lr).view(-1, 1).expand(H_lr, W_lr)
    grid_x = torch.arange(W_lr).view(1, -1).expand(H_lr, W_lr)
    # 对应 HR 子块中心连续坐标（像素中心公式），取子块所有像素的坐标均值作为面积中心
    lr_coords = torch.empty(2, H_lr, W_lr, dtype=hr_coords.dtype)
    for i in range(H_lr):
        for j in range(W_lr):
            y0, y1 = i * scale, (i + 1) * scale
            x0, x1 = j * scale, (j + 1) * scale
            block_x = hr_coords[0, y0:y1, x0:x1]
            block_y = hr_coords[1, y0:y1, x0:x1]
            lr_coords[0, i, j] = block_x.mean()
            lr_coords[1, i, j] = block_y.mean()
    return lr_coords

def validate_alignment(h: int, w: int, scale: int, tol: float = 1e-6):
    hr = pixel_center_grid(h, w)
    lr = pixel_center_grid(h // scale, w // scale)
    lr_from_hr = downsample_block_center(hr, scale)
    err = (lr - lr_from_hr).abs().max().item()
    return err <= tol, err

def main():
    H, W, S = 128, 128, 4
    ok, err = validate_alignment(H, W, S)
    out = {
        'hr_size': [H, W],
        'scale': S,
        'aligned': ok,
        'max_abs_error': err
    }
    out_path = Path('runs') / 'position_encoding_alignment.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Alignment: {ok}, max_abs_error={err:.2e}, written {out_path}")

if __name__ == '__main__':
    main()
