"""
独立测试集评估与流体能谱分析脚本
比较 Baseline (Bicubic), 原架构 (SwinTWithEncoder), 全新架构 (SwinFluidSR) 在未知测试集上的定量与物理表现。
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from datasets.pdebench import PDEBenchSR
from models.registry import create_model
from utils.metrics import compute_all_metrics


def compute_radial_energy_spectrum(field: torch.Tensor):
    """
    计算 2D 标量场的径向平均能量谱 (Radially Averaged Power Spectrum)
    field: [H, W] numpy array
    """
    H, W = field.shape
    # 2D FFT
    F_field = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(F_field) ** 2 / (H * W)

    # 径向坐标网格
    y, x = np.indices((H, W))
    center = (int(H / 2), int(W / 2))
    r = np.hypot(x - center[1], y - center[0]).astype(int)

    # 沿半径统计平均功率
    radial_mean = np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel())
    k = np.arange(len(radial_mean))
    # 截取有效波数范围
    max_k = min(H, W) // 2
    return k[1:max_k], radial_mean[1:max_k]


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running evaluation on {device}...')
    os.makedirs('runs_swin_fluid_sr', exist_ok=True)

    # 1. 准备测试集
    test_dataset = PDEBenchSR(
        data_path='/root/autodl-tmp/datasets/2D_rdb_NA_NA.h5',
        keys=['data'],
        split='test',
        splits_dir='splits_shallow',
        scale=4
    )
    print(f'Test dataset loaded with {len(test_dataset)} samples.')

    # 2. 加载旧模型 (SwinTWithEncoder)
    ckpt_old_path = 'runs/checkpoints/best_model.pth'
    model_old = create_model(
        'SwinTWithEncoder',
        in_channels=4,
        out_channels=1,
        img_size=128,
        window_size=8,
        post_conv3x3=True
    ).to(device)
    if os.path.exists(ckpt_old_path):
        ckpt_old = torch.load(ckpt_old_path, map_location=device, weights_only=False)
        state_dict = ckpt_old.get('model_state_dict', ckpt_old)
        # 兼容 DataParallel 键名前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_old.load_state_dict(state_dict, strict=False)
        print('Loaded SwinTWithEncoder checkpoint.')
    model_old.eval()

    # 3. 加载新模型 (SwinFluidSR)
    ckpt_new_path = 'runs_swin_fluid_sr/checkpoints/best_model.pth'
    model_new = create_model(
        'SwinFluidSR',
        in_channels=1,
        out_channels=1,
        img_size=32,
        upscale_factor=4,
        embed_dim=96,
        depths=[4, 4, 4, 4],
        num_heads=[4, 4, 4, 4],
        window_size=8,
        mlp_ratio=2.0,
        global_residual=True,
    ).to(device)
    if os.path.exists(ckpt_new_path):
        ckpt_new = torch.load(ckpt_new_path, map_location=device, weights_only=False)
        state_dict = ckpt_new.get('model_state_dict', ckpt_new)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_new.load_state_dict(state_dict, strict=False)
        print('Loaded SwinFluidSR checkpoint.')
    model_new.eval()

    # 4. 执行盲测统计
    metrics_bicubic = []
    metrics_old = []
    metrics_new = []

    samples_to_plot = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            target = item['target'].unsqueeze(0).to(device)            # [1, 1, 128, 128]
            lr = item['original_observation'].unsqueeze(0).to(device)   # [1, 1, 32, 32]
            baseline = item['baseline'].unsqueeze(0).to(device)        # [1, 1, 128, 128]
            coords = item['coords'].unsqueeze(0).to(device)            # [1, 2, 128, 128]
            mask = item['mask'].unsqueeze(0).to(device)                # [1, 1, 128, 128]

            # 4.1 Bicubic 插值评估
            pred_bic = baseline
            m_bic = compute_all_metrics(pred_bic, target)
            metrics_bicubic.append(m_bic)

            # 4.2 旧模型 SwinTWithEncoder
            in_old = torch.cat([baseline, coords, mask], dim=1)         # [1, 4, 128, 128]
            pred_old = model_old(in_old)
            m_old = compute_all_metrics(pred_old, target)
            metrics_old.append(m_old)

            # 4.3 新模型 SwinFluidSR
            pred_new = model_new(lr)                                    # [1, 1, 128, 128]
            m_new = compute_all_metrics(pred_new, target)
            metrics_new.append(m_new)

            # 挑选有代表性的样本做可视化
            if i in [0, 2, 4]:
                samples_to_plot.append({
                    'index': i,
                    'lr': lr[0, 0].cpu().numpy(),
                    'gt': target[0, 0].cpu().numpy(),
                    'bicubic': pred_bic[0, 0].cpu().numpy(),
                    'pred_old': pred_old[0, 0].cpu().numpy(),
                    'pred_new': pred_new[0, 0].cpu().numpy(),
                })

    # 5. 汇总平均指标
    def average_metrics(m_list):
        keys = ['rel_l2', 'mae', 'psnr', 'ssim', 'frmse_high', 'frmse_mid', 'frmse_low']
        return {k: np.mean([m[k] for m in m_list if k in m]) for k in keys}

    avg_bic = average_metrics(metrics_bicubic)
    avg_old = average_metrics(metrics_old)
    avg_new = average_metrics(metrics_new)

    print("\n" + "=" * 65)
    print("=== 独立测试集 (Test Split) 定量盲测结果 ===")
    print("=" * 65)
    print(f"{'Metric':<14} | {'Bicubic':<14} | {'SwinTWithEncoder':<18} | {'SwinFluidSR':<14}")
    print("-" * 65)
    for k in avg_bic.keys():
        print(f"{k:<14} | {avg_bic[k]:<14.5f} | {avg_old[k]:<18.5f} | {avg_new[k]:<14.5f}")
    print("=" * 65)

    # 6. 绘制多样本重构画廊图
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    titles = ['Input (32x32 LR)', 'Bicubic (128x128)', 'SwinTWithEncoder', 'SwinFluidSR (New)', 'Ground Truth (HR)']

    for row_idx, sample in enumerate(samples_to_plot):
        imgs = [
            sample['lr'],
            sample['bicubic'],
            sample['pred_old'],
            sample['pred_new'],
            sample['gt']
        ]
        vmin, vmax = sample['gt'].min(), sample['gt'].max()

        for col_idx, img in enumerate(imgs):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=13, fontweight='bold', pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"Test Case #{sample['index']}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_gallery = os.path.join('runs_swin_fluid_sr', 'test_set_reconstruction_gallery.png')
    plt.savefig(out_gallery, dpi=200)
    plt.close()
    print(f'Saved test gallery to {out_gallery}')

    # 7. 计算并绘制径向能量谱对比 (Energy Spectrum Analysis)
    plt.figure(figsize=(10, 6.5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    sample0 = samples_to_plot[0]
    k_gt, spec_gt = compute_radial_energy_spectrum(sample0['gt'])
    _, spec_bic = compute_radial_energy_spectrum(sample0['bicubic'])
    _, spec_old = compute_radial_energy_spectrum(sample0['pred_old'])
    _, spec_new = compute_radial_energy_spectrum(sample0['pred_new'])

    plt.loglog(k_gt, spec_gt, label='Ground Truth (Physics Real)', color='black', lw=2.5)
    plt.loglog(k_gt, spec_bic, label='Bicubic Interpolation', color='gray', lw=1.8, linestyle=':')
    plt.loglog(k_gt, spec_old, label='SwinTWithEncoder (Old)', color='#e74c3c', lw=2, linestyle='--')
    plt.loglog(k_gt, spec_new, label='SwinFluidSR (New Architecture)', color='#2ecc71', lw=2.5)

    plt.title('Turbulence Energy Spectrum Comparison (Radially Averaged)', fontsize=14, fontweight='bold')
    plt.xlabel('Wavenumber k', fontsize=12)
    plt.ylabel('Energy Spectrum E(k)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    out_spec = os.path.join('runs_swin_fluid_sr', 'turbulence_energy_spectrum_comparison.png')
    plt.savefig(out_spec, dpi=200)
    plt.close()
    print(f'Saved energy spectrum plot to {out_spec}')


if __name__ == '__main__':
    main()
