import torch
from utils.metrics import MetricsCalculator

# 创建测试数据
pred = torch.randn(2, 3, 2, 128, 128)  # B, T, C, H, W
target = torch.randn(2, 3, 2, 128, 128)

# 创建metrics计算器
metrics_calc = MetricsCalculator(
    image_size=(128, 128),
    boundary_width=16,
    freq_bands={'low': (0, 16), 'mid': (16, 32), 'high': (32, 64)}
)

# 计算指标
rel_l2 = metrics_calc.compute_rel_l2(pred, target)
print(f'rel_l2 type: {type(rel_l2)}, shape: {rel_l2.shape if torch.is_tensor(rel_l2) else "scalar"}, value: {rel_l2}')

psnr = metrics_calc.compute_psnr(pred, target)
print(f'psnr type: {type(psnr)}, shape: {psnr.shape if torch.is_tensor(psnr) else "scalar"}, value: {psnr}')

ssim = metrics_calc.compute_ssim(pred, target)
print(f'ssim type: {type(ssim)}, shape: {ssim.shape if torch.is_tensor(ssim) else "scalar"}, value: {ssim}')