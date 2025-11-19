#!/usr/bin/env python3
"""
AR Training Visualizer
Provides training curves, prediction visualization, and error analysis.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-GUI backend for headless/HPC environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
import sys
warnings.filterwarnings('ignore')

# 添加tools/visualization路径以导入可视化工具
project_root = Path(__file__).resolve().parents[1]
viz_tools_path = project_root / "tools" / "visualization"
if str(viz_tools_path) not in sys.path:
    sys.path.append(str(viz_tools_path))

from create_input_output_prediction_error_viz import InputOutputPredictionErrorVisualizer

# Safe English font configuration to avoid missing glyph boxes
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # High-quality output

class ARTrainingVisualizer:
    """AR training visualizer"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.vis_dir / "training_curves").mkdir(exist_ok=True)
        (self.vis_dir / "predictions").mkdir(exist_ok=True)
        (self.vis_dir / "error_analysis").mkdir(exist_ok=True)
        (self.vis_dir / "temporal_analysis").mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def plot_training_curves(self, history: Dict[str, List], save_name: str = "training_curves"):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AR Training Monitoring', fontsize=16, fontweight='bold')
        
        epochs = history.get('epochs', [])
        # 如果没有epochs，但有损失数据，使用长度生成1..N的索引
        max_series_len = max(
            len(history.get('train_losses', [])),
            len(history.get('val_losses', [])),
            len(history.get('learning_rates', []))
        )
        if not epochs and max_series_len > 0:
            epochs = list(range(1, max_series_len + 1))
        if not epochs:
            print("Warning: no training history found")
            return
        
        def x_for(series_len: int) -> List[int]:
            """根据序列长度生成与y匹配的x轴，优先使用epochs的前缀"""
            if series_len <= 0:
                return []
            if len(epochs) >= series_len:
                return epochs[:series_len]
            return list(range(1, series_len + 1))
        
        # 训练和验证损失
        ax1 = axes[0, 0]
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        if train_losses:
            ax1.plot(x_for(len(train_losses)), train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            ax1.plot(x_for(len(val_losses)), val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 学习率曲线
        ax2 = axes[0, 1]
        learning_rates = history.get('learning_rates', [])
        if learning_rates:
            ax2.plot(x_for(len(learning_rates)), learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Curriculum learning stages
        ax3 = axes[1, 0]
        if 'curriculum_stages' in history:
            stages = history['curriculum_stages']
            if stages:
                stage_epochs = []
                stage_T_outs = []
                for stage in stages:
                    # 某些记录可能缺少epoch，跳过以避免与x轴不对齐
                    if 'epoch' in stage:
                        stage_epochs.append(stage['epoch'])
                        stage_T_outs.append(stage.get('T_out', 0))
                if stage_epochs and stage_T_outs:
                    ax3.step(stage_epochs, stage_T_outs, 'o-', linewidth=2, markersize=8)
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Prediction steps (T_out)')
                    ax3.set_title('Curriculum Progress')
                    ax3.grid(True, alpha=0.3)
        
        # Validation metrics
        ax4 = axes[1, 1]
        if 'val_metrics' in history and history['val_metrics']:
            # 提取指标数据
            metrics_data = {}
            for epoch_metrics in history['val_metrics']:
                for metric_name, value in epoch_metrics.items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(value)
            
            # 绘制主要指标，确保x轴长度与y一致
            for metric_name, values in metrics_data.items():
                if metric_name in ['rel_l2', 'mae', 'mse'] and values:
                    ax4.plot(x_for(len(values)), values, 'o-', label=metric_name.upper(), linewidth=2)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Metric Value')
            ax4.set_title('Validation Metrics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        
        plt.tight_layout()
        save_path = self.vis_dir / "training_curves" / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved: {save_path}")
        
    def visualize_ar_predictions(self, input_seq: torch.Tensor, target_seq: torch.Tensor, 
                                pred_seq: torch.Tensor, timestep_idx: int = 0, 
                                save_name: str = "ar_predictions", norm_stats: dict = None):
        """Visualize AR predictions with unified colorbar"""
        # 转换为numpy
        if isinstance(input_seq, torch.Tensor):
            input_seq = input_seq.detach().cpu().numpy()
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()
        
        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and 'mean' in norm_stats and 'std' in norm_stats:
            mean = norm_stats['mean']
            std = norm_stats['std']
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0
                
            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，AR可视化使用z-score域数据")
        
        # 选择第一个样本
        if len(input_seq.shape) == 4:  # [B, C, H, W]
            input_frame = input_seq[0]  # [C, H, W]
            target_frames = target_seq[0]  # [T, C, H, W]
            pred_frames = pred_seq[0]  # [T, C, H, W]
        else:
            input_frame = input_seq
            target_frames = target_seq
            pred_frames = pred_seq
        
        # 确保维度正确
        if len(input_frame.shape) == 3:
            input_frame = input_frame[0]  # 取第一个通道
        if len(target_frames.shape) == 4:
            target_frames = target_frames[:, 0]  # 取第一个通道 [T, H, W]
        if len(pred_frames.shape) == 4:
            pred_frames = pred_frames[:, 0]  # 取第一个通道 [T, H, W]
        
        # 反归一化到真实数据尺度
        input_frame = input_frame * std_val + mean_val
        target_frames = target_frames * std_val + mean_val
        pred_frames = pred_frames * std_val + mean_val
        
        T_out = min(target_frames.shape[0], pred_frames.shape[0])
        
        # 计算统一的数值范围（基于百分位数避免异常值影响）
        all_values = np.concatenate([
            input_frame.flatten(),
            target_frames[:min(T_out, 6)].flatten(),
            pred_frames[:min(T_out, 6)].flatten()
        ])
        vmin = np.percentile(all_values, 2)  # 2%分位数
        vmax = np.percentile(all_values, 98)  # 98%分位数
        
        # 创建更合理的布局：时间序列可视化
        n_cols = min(T_out, 6)
        fig = plt.figure(figsize=(4 * n_cols + 1, 10))
        
        # 创建网格布局，为colorbar预留右侧空间
        gs = fig.add_gridspec(3, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.05, hspace=0.15)
        
        fig.suptitle(f'AR Predictions - Timestep {timestep_idx}', fontsize=16, fontweight='bold', y=0.95)
        
        for t in range(n_cols):
            # 输入帧（只在第一列显示）
            if t == 0:
                input_2d = self._ensure_2d_for_imshow(input_frame)
                ax_input = fig.add_subplot(gs[0, t])
                im1 = ax_input.imshow(input_2d, cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
                ax_input.set_title(f'Input Frame (t=0)', fontsize=10, fontweight='bold')
                ax_input.set_xticks([])
                ax_input.set_yticks([])
            else:
                # 对于其他列，创建空的输入帧位置
                ax_input = fig.add_subplot(gs[0, t])
                ax_input.axis('off')
            
            # 真实值
            target_2d = self._ensure_2d_for_imshow(target_frames[t])
            ax_target = fig.add_subplot(gs[1, t])
            im2 = ax_target.imshow(target_2d, cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
            ax_target.set_title(f'Ground Truth (t={t+1})', fontsize=10, fontweight='bold')
            ax_target.set_xticks([])
            ax_target.set_yticks([])
            
            # 预测值
            pred_2d = self._ensure_2d_for_imshow(pred_frames[t])
            ax_pred = fig.add_subplot(gs[2, t])
            im3 = ax_pred.imshow(pred_2d, cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Prediction (t={t+1})', fontsize=10, fontweight='bold')
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
        
        # 添加统一的颜色条 - 在最右侧
        cbar = fig.colorbar(im2, cax=fig.add_subplot(gs[:, -1]), orientation='vertical')
        cbar.set_label('Value', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        save_path = self.vis_dir / "predictions" / f"{save_name}_t{timestep_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"✅ AR prediction visualization saved: {save_path}")
        
    def create_error_analysis(self, target_seq: torch.Tensor, pred_seq: torch.Tensor, 
                             save_name: str = "error_analysis", norm_stats: dict = None):
        """Create error analysis visualization"""
        # 转换为numpy
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()
        
        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and 'mean' in norm_stats and 'std' in norm_stats:
            mean = norm_stats['mean']
            std = norm_stats['std']
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0
                
            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，误差分析使用z-score域数据")
        
        # 选择第一个样本
        if len(target_seq.shape) == 4:  # [B, T, C, H, W]
            target_seq = target_seq[0]  # [T, C, H, W]
            pred_seq = pred_seq[0]  # [T, C, H, W]
        
        # 反归一化到真实数据尺度
        target_seq = target_seq * std_val + mean_val
        pred_seq = pred_seq * std_val + mean_val
        
        T_out = min(target_seq.shape[0], pred_seq.shape[0])
        
        # 计算误差（在真实数据尺度上）
        errors = np.abs(target_seq - pred_seq)
        
        # 创建误差分析图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AR Prediction Error Analysis', fontsize=16, fontweight='bold')
        
        # 时间步误差演化
        ax1 = axes[0, 0]
        mse_per_step = []
        mae_per_step = []
        for t in range(T_out):
            mse = np.mean((target_seq[t] - pred_seq[t])**2)
            mae = np.mean(np.abs(target_seq[t] - pred_seq[t]))
            mse_per_step.append(mse)
            mae_per_step.append(mae)
        
        ax1.plot(range(1, T_out+1), mse_per_step, 'ro-', label='MSE', linewidth=2)
        ax1.plot(range(1, T_out+1), mae_per_step, 'bo-', label='MAE', linewidth=2)
        ax1.set_xlabel('Prediction Steps')
        ax1.set_ylabel('Error')
        ax1.set_title('Error Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 误差分布直方图
        ax2 = axes[0, 1]
        all_errors = errors.flatten()
        ax2.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution Histogram')
        ax2.grid(True, alpha=0.3)
        
        # 误差热图（最后一个时间步）
        ax3 = axes[0, 2]
        if T_out > 0:
            # errors 形状可能是 [T, C, H, W] 或 [C, H, W]
            if errors.ndim == 4:  # [T, C, H, W]
                error_map = errors[-1, 0]  # 获取最后时间步的第一个通道 [H, W]
            elif errors.ndim == 3:  # [C, H, W]
                error_map = errors[0]  # 获取第一个通道 [H, W]
            else:  # [H, W]
                error_map = errors
            
            # 确保是2D数组
            while error_map.ndim > 2:
                error_map = error_map[0]  # 继续取第一个元素直到2D
            
            if error_map.ndim == 1:
                # 如果是1D，尝试重塑为2D
                size = int(np.sqrt(error_map.shape[0]))
                if size * size == error_map.shape[0]:
                    error_map = error_map.reshape(size, size)
            
            im = ax3.imshow(error_map, cmap='Reds', aspect='equal')
            ax3.set_title(f'Error Heatmap (t={T_out})')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 空间误差分析
        ax4 = axes[1, 0]
        if T_out > 0:
            # 计算空间平均误差
            if errors.ndim == 4:  # [T, C, H, W]
                spatial_error = np.mean(errors, axis=0)  # 平均时间步 [C, H, W]
            else:  # [C, H, W] 或其他
                spatial_error = errors
            
            # 使用helper函数确保2D
            spatial_error_2d = self._ensure_2d_for_imshow(spatial_error)
            
            im = ax4.imshow(spatial_error_2d, cmap='Reds', aspect='equal')
            ax4.set_title('Spatial Mean Error')
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # 通道误差对比
        ax5 = axes[1, 1]
        if target_seq.shape[1] > 1:  # 多通道
            channel_errors = []
            for c in range(target_seq.shape[1]):
                channel_error = np.mean(errors[:, c])
                channel_errors.append(channel_error)
            
            ax5.bar(range(len(channel_errors)), channel_errors, color=['red', 'blue'])
            ax5.set_xlabel('Channel')
            ax5.set_ylabel('Mean Absolute Error')
            ax5.set_title('Channel Error Comparison')
            ax5.set_xticks(range(len(channel_errors)))
            ax5.set_xticklabels([f'Ch{i}' for i in range(len(channel_errors))])
        
        # 累积误差
        ax6 = axes[1, 2]
        cumulative_error = np.cumsum([np.mean(errors[t]) for t in range(T_out)])
        ax6.plot(range(1, T_out+1), cumulative_error, 'go-', linewidth=2)
        ax6.set_xlabel('Prediction Steps')
        ax6.set_ylabel('Cumulative Error')
        ax6.set_title('Cumulative Error Growth')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.vis_dir / "error_analysis" / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Error analysis visualization saved: {save_path}")
        
    def create_temporal_analysis(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                               save_name: str = "temporal_analysis", norm_stats: dict = None):
        """Create temporal analysis visualization"""
        # 转换为numpy
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.detach().cpu().numpy()
        if isinstance(target_seq, torch.Tensor):
            target_seq = target_seq.detach().cpu().numpy()
        
        # 获取归一化统计信息进行反归一化
        if norm_stats is not None and 'mean' in norm_stats and 'std' in norm_stats:
            mean = norm_stats['mean']
            std = norm_stats['std']
            # 确保mean和std是标量或可以广播到正确形状
            if isinstance(mean, torch.Tensor):
                mean_val = float(mean[0]) if mean.numel() > 0 else 0.0
            else:
                mean_val = float(mean) if np.isscalar(mean) else 0.0
                
            if isinstance(std, torch.Tensor):
                std_val = float(std[0]) if std.numel() > 0 else 1.0
            else:
                std_val = float(std) if np.isscalar(std) else 1.0
        else:
            # 如果没有归一化统计信息，使用默认值
            mean_val = 0.0
            std_val = 1.0
            print("⚠️ 未找到归一化统计信息，时间分析使用z-score域数据")
        
        # 选择第一个样本
        if len(pred_seq.shape) == 4:  # [B, T, C, H, W]
            pred_seq = pred_seq[0]  # [T, C, H, W]
            target_seq = target_seq[0]  # [T, C, H, W]
        
        # 反归一化到真实数据尺度
        pred_seq = pred_seq * std_val + mean_val
        target_seq = target_seq * std_val + mean_val
        
        T_out = min(pred_seq.shape[0], target_seq.shape[0])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Evolution Analysis', fontsize=16, fontweight='bold')
        
        # 能量演化
        ax1 = axes[0, 0]
        pred_energy = []
        target_energy = []
        for t in range(T_out):
            p = pred_seq[t]
            g = target_seq[t]
            if p.ndim == 3:  # [C,H,W]
                p_energy = np.sum(p**2)
            else:  # [H,W]
                p_energy = np.sum(p**2)
            if g.ndim == 3:
                g_energy = np.sum(g**2)
            else:
                g_energy = np.sum(g**2)
            pred_energy.append(p_energy)
            target_energy.append(g_energy)
        
        ax1.plot(range(1, T_out+1), target_energy, 'r-', label='Ground Truth Energy', linewidth=2)
        ax1.plot(range(1, T_out+1), pred_energy, 'b--', label='Prediction Energy', linewidth=2)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Conservation Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 相关性演化
        ax2 = axes[0, 1]
        correlations = []
        for t in range(T_out):
            p = pred_seq[t]
            g = target_seq[t]
            # 对齐通道与空间尺寸：通道取均值为单通道，空间取交集
            if p.ndim == 3:  # [C,H,W]
                p = p.mean(axis=0)
            if g.ndim == 3:
                g = g.mean(axis=0)
            h = min(p.shape[-2], g.shape[-2])
            w = min(p.shape[-1], g.shape[-1])
            p = p[:h, :w]
            g = g[:h, :w]
            try:
                corr = float(np.corrcoef(p.flatten(), g.flatten())[0, 1])
            except Exception:
                corr = 0.0
            correlations.append(corr)
        
        ax2.plot(range(1, T_out+1), correlations, 'go-', linewidth=2)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_title('Prediction–Ground Truth Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # 频谱分析
        ax3 = axes[1, 0]
        if T_out > 1:
            # 计算最后时间步的频谱
            p_last = pred_seq[-1]
            g_last = target_seq[-1]
            if p_last.ndim == 3:
                p_last = p_last.mean(axis=0)
            if g_last.ndim == 3:
                g_last = g_last.mean(axis=0)
            h = min(p_last.shape[-2], g_last.shape[-2])
            w = min(p_last.shape[-1], g_last.shape[-1])
            p_last = p_last[:h, :w]
            g_last = g_last[:h, :w]
            pred_fft = np.fft.fft2(p_last)
            target_fft = np.fft.fft2(g_last)
            
            pred_power = np.abs(pred_fft)**2
            target_power = np.abs(target_fft)**2
            
            # 确保是2D数组
            if pred_power.ndim > 2:
                pred_power = pred_power.squeeze()
            if target_power.ndim > 2:
                target_power = target_power.squeeze()
            
            # 如果仍然不是2D，取第一个通道
            if pred_power.ndim > 2:
                pred_power = pred_power[0]
            if target_power.ndim > 2:
                target_power = target_power[0]
            
            # 径向平均
            h, w = pred_power.shape
            center = (h//2, w//2)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            r_int = r.astype(int)
            max_r = min(h//2, w//2)
            
            pred_radial = []
            target_radial = []
            for i in range(max_r):
                mask = (r_int == i)
                if np.any(mask):
                    pred_radial.append(np.mean(pred_power[mask]))
                    target_radial.append(np.mean(target_power[mask]))
            
            ax3.loglog(pred_radial, 'b-', label='Prediction Power Spectrum', linewidth=2)
            ax3.loglog(target_radial, 'r-', label='Ground Truth Power Spectrum', linewidth=2)
            ax3.set_xlabel('Wavenumber')
            ax3.set_ylabel('Power')
            ax3.set_title('Power Spectrum Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 稳定性分析
        ax4 = axes[1, 1]
        if T_out > 1:
            stability = []
            for t in range(1, T_out):
                diff = np.mean(np.abs(pred_seq[t] - pred_seq[t-1]))
                stability.append(diff)
            
            ax4.plot(range(2, T_out+1), stability, 'mo-', linewidth=2)
            ax4.set_xlabel('Timestep')
            ax4.set_ylabel('Frame Difference')
            ax4.set_title('Prediction Stability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.vis_dir / "temporal_analysis" / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Temporal analysis visualization saved: {save_path}")
        
    def create_comprehensive_report(self, history: Dict, sample_data: Optional[Dict] = None):
        """Create comprehensive report"""
        report_path = self.vis_dir / "comprehensive_report.html"
        # 安全地计算统计值，避免空列表导致的索引错误
        epochs_list = history.get('epochs', []) or []
        train_losses = history.get('train_losses', []) or []
        val_losses = history.get('val_losses', []) or []
        total_epochs = len(epochs_list)
        final_train_loss = train_losses[-1] if len(train_losses) > 0 else 0.0
        final_val_loss = val_losses[-1] if len(val_losses) > 0 else 0.0
        best_val_loss = min(val_losses) if len(val_losses) > 0 else 0.0
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AR Training Comprehensive Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
                .metric-label {{ color: #7f8c8d; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AR Training Comprehensive Report</h1>
                <p>Generated at: {Path().cwd().name} - {np.datetime64('now')}</p>
            </div>
            
            <div class="section">
                <h2>Training Overview</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{total_epochs}</div>
                        <div class="metric-label">Epochs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{final_train_loss:.6f}</div>
                        <div class="metric-label">Final Training Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{final_val_loss:.6f}</div>
                        <div class="metric-label">Final Validation Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{best_val_loss:.6f}</div>
                        <div class="metric-label">Best Validation Loss</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image-grid">
        """
        
        # 添加图片
        for img_dir in ["training_curves", "predictions", "error_analysis", "temporal_analysis"]:
            img_path = self.vis_dir / img_dir
            if img_path.exists():
                for img_file in img_path.glob("*.png"):
                    rel_path = img_file.relative_to(self.vis_dir)
                    html_content += f'<div><h3>{img_file.stem}</h3><img src="{rel_path}" alt="{img_file.stem}"></div>\n'
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Comprehensive report saved: {report_path}")
        
        return str(report_path)
        
    def _ensure_2d_for_imshow(self, arr):
        """确保数组是2D的，适合imshow显示"""
        # 如果是tensor，转换为numpy
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()
        
        # 如果维度大于2，取第一个通道/元素
        while arr.ndim > 2:
            arr = arr[0]
        
        # 如果是1D，尝试重塑为2D
        if arr.ndim == 1:
            size = int(np.sqrt(arr.shape[0]))
            if size * size == arr.shape[0]:
                arr = arr.reshape(size, size)
            else:
                # 如果不是完全平方数，创建一个合理的2D形状
                h = int(np.sqrt(arr.shape[0]))
                w = arr.shape[0] // h
                if h * w <= arr.shape[0]:
                    arr = arr[:h*w].reshape(h, w)
                else:
                    # 最后的备选方案：创建一个小的2D数组
                    arr = np.zeros((8, 8))
        
        # 确保是numpy数组且是2D
        arr = np.asarray(arr)
        if arr.ndim != 2:
            # 如果仍然不是2D，强制创建一个2D数组
            arr = np.zeros((8, 8))
        
        return arr