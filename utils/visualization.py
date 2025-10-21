"""可视化工具模块

实现训练过程和结果的可视化功能，包括时序动画、误差分析等
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2
from tqdm import tqdm

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TemporalVisualizer:
    """时序可视化器"""
    
    def __init__(self, save_dir: Path, config: Dict):
        self.save_dir = Path(save_dir)
        self.config = config
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "training").mkdir(exist_ok=True)
        (self.save_dir / "results").mkdir(exist_ok=True)
        (self.save_dir / "animations").mkdir(exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics_history: Dict[str, List], epoch: int):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'训练进度 - Epoch {epoch}', fontsize=16)
        
        # 损失曲线
        if metrics_history['train_loss'] and metrics_history['val_loss']:
            axes[0, 0].plot(metrics_history['train_loss'], label='训练损失', alpha=0.8)
            axes[0, 0].plot(metrics_history['val_loss'], label='验证损失', alpha=0.8)
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Rel-L2曲线
        if metrics_history['train_rel_l2'] and metrics_history['val_rel_l2']:
            axes[0, 1].plot(metrics_history['train_rel_l2'], label='训练Rel-L2', alpha=0.8)
            axes[0, 1].plot(metrics_history['val_rel_l2'], label='验证Rel-L2', alpha=0.8)
            axes[0, 1].set_title('Rel-L2曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Rel-L2')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # MAE曲线
        if metrics_history['train_mae'] and metrics_history['val_mae']:
            axes[1, 0].plot(metrics_history['train_mae'], label='训练MAE', alpha=0.8)
            axes[1, 0].plot(metrics_history['val_mae'], label='验证MAE', alpha=0.8)
            axes[1, 0].set_title('MAE曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if metrics_history['learning_rate']:
            axes[1, 1].plot(metrics_history['learning_rate'], alpha=0.8, color='orange')
            axes[1, 1].set_title('学习率曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training" / f"curves_epoch_{epoch:04d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_training_predictions(self, input_seq: torch.Tensor, 
                                target_seq: torch.Tensor, 
                                pred_seq: torch.Tensor,
                                step: int, epoch: int):
        """保存训练过程中的预测结果"""
        # 转换为numpy
        input_seq = input_seq.detach().cpu().numpy()  # [T_in, C, H, W] or [C, H, W]
        target_seq = target_seq.detach().cpu().numpy()  # [T_out, C, H, W] or [C, H, W]
        pred_seq = pred_seq.detach().cpu().numpy()  # [T_out, C, H, W] or [C, H, W]
        
        print(f"可视化输入形状: input={input_seq.shape}, target={target_seq.shape}, pred={pred_seq.shape}")
        
        # 处理不同的输入形状
        if len(input_seq.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            input_seq = input_seq[np.newaxis, ...]
        if len(target_seq.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            target_seq = target_seq[np.newaxis, ...]
        if len(pred_seq.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            pred_seq = pred_seq[np.newaxis, ...]
        
        # 选择第一个通道进行可视化
        input_seq = input_seq[:, 0]  # [T_in, H, W]
        target_seq = target_seq[:, 0]  # [T_out, H, W]
        pred_seq = pred_seq[:, 0]  # [T_out, H, W]
        
        print(f"可视化处理后形状: input={input_seq.shape}, target={target_seq.shape}, pred={pred_seq.shape}")
        
        # 创建对比图
        T_out = target_seq.shape[0]
        T_in = input_seq.shape[0]
        max_cols = max(3, T_out)
        fig, axes = plt.subplots(3, max_cols, figsize=(4*max_cols, 12))
        
        # 确保axes是2D数组
        if max_cols == 1:
            axes = axes.reshape(3, 1)
        
        # 输入序列
        for t in range(max_cols):
            if t < T_in:
                im = axes[0, t].imshow(input_seq[t], cmap='viridis')
                axes[0, t].set_title(f'输入 t={t}')
                axes[0, t].axis('off')
                plt.colorbar(im, ax=axes[0, t], fraction=0.046)
            else:
                axes[0, t].axis('off')
        
        # 目标序列
        for t in range(max_cols):
            if t < T_out:
                im = axes[1, t].imshow(target_seq[t], cmap='viridis')
                axes[1, t].set_title(f'目标 t={t+1}')
                axes[1, t].axis('off')
                plt.colorbar(im, ax=axes[1, t], fraction=0.046)
            else:
                axes[1, t].axis('off')
        
        # 预测序列
        T_pred = pred_seq.shape[0]  # 实际预测序列长度
        for t in range(max_cols):
            if t < T_pred:
                im = axes[2, t].imshow(pred_seq[t], cmap='viridis')
                axes[2, t].set_title(f'预测 t={t+1}')
                axes[2, t].axis('off')
                plt.colorbar(im, ax=axes[2, t], fraction=0.046)
            else:
                axes[2, t].axis('off')
        
        plt.suptitle(f'训练预测结果 - Step {step}, Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / "training" / f"pred_step_{step:06d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_temporal_animation(self, sequence: np.ndarray, 
                                title: str, filename: str,
                                fps: int = 5, interval: int = 200):
        """创建时序动画"""
        # sequence: [T, H, W]
        T, H, W = sequence.shape
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 设置颜色范围
        vmin, vmax = sequence.min(), sequence.max()
        
        # 初始化图像
        im = ax.imshow(sequence[0], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title} - t=0')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        
        def animate(frame):
            im.set_array(sequence[frame])
            ax.set_title(f'{title} - t={frame}')
            return [im]
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=T, interval=interval, blit=True, repeat=True
        )
        
        # 保存动画
        anim.save(self.save_dir / "animations" / f"{filename}.gif", 
                 writer='pillow', fps=fps)
        plt.close()
    
    def plot_error_analysis(self, target_seq: np.ndarray, 
                          pred_seq: np.ndarray, 
                          case_id: str):
        """绘制误差分析图"""
        # target_seq, pred_seq: [T, C, H, W]
        T, C, H, W = target_seq.shape
        
        # 计算误差
        error_seq = np.abs(pred_seq - target_seq)
        rel_error_seq = error_seq / (np.abs(target_seq) + 1e-8)
        
        # 选择第一个通道
        target_seq = target_seq[:, 0]  # [T, H, W]
        pred_seq = pred_seq[:, 0]
        error_seq = error_seq[:, 0]
        rel_error_seq = rel_error_seq[:, 0]
        
        # 创建对比图
        fig, axes = plt.subplots(4, T, figsize=(4*T, 16))
        
        for t in range(T):
            # 目标
            im1 = axes[0, t].imshow(target_seq[t], cmap='viridis')
            axes[0, t].set_title(f'目标 t={t+1}')
            axes[0, t].axis('off')
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046)
            
            # 预测
            im2 = axes[1, t].imshow(pred_seq[t], cmap='viridis')
            axes[1, t].set_title(f'预测 t={t+1}')
            axes[1, t].axis('off')
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046)
            
            # 绝对误差
            im3 = axes[2, t].imshow(error_seq[t], cmap='Reds')
            axes[2, t].set_title(f'绝对误差 t={t+1}')
            axes[2, t].axis('off')
            plt.colorbar(im3, ax=axes[2, t], fraction=0.046)
            
            # 相对误差
            im4 = axes[3, t].imshow(rel_error_seq[t], cmap='Reds')
            axes[3, t].set_title(f'相对误差 t={t+1}')
            axes[3, t].axis('off')
            plt.colorbar(im4, ax=axes[3, t], fraction=0.046)
        
        plt.suptitle(f'误差分析 - Case {case_id}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / "results" / f"error_analysis_{case_id}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_profiles(self, target_seq: np.ndarray, 
                             pred_seq: np.ndarray, 
                             case_id: str,
                             sample_points: List[Tuple[int, int]] = None):
        """绘制时序剖面图"""
        # target_seq, pred_seq: [T, C, H, W]
        T, C, H, W = target_seq.shape
        
        # 选择采样点
        if sample_points is None:
            sample_points = [
                (H//4, W//4),      # 左上
                (H//2, W//2),      # 中心
                (3*H//4, 3*W//4),  # 右下
            ]
        
        # 选择第一个通道
        target_seq = target_seq[:, 0]  # [T, H, W]
        pred_seq = pred_seq[:, 0]
        
        fig, axes = plt.subplots(len(sample_points), 1, figsize=(12, 4*len(sample_points)))
        if len(sample_points) == 1:
            axes = [axes]
        
        time_steps = np.arange(T)
        
        for i, (y, x) in enumerate(sample_points):
            # 提取时序数据
            target_profile = target_seq[:, y, x]
            pred_profile = pred_seq[:, y, x]
            
            # 绘制时序曲线
            axes[i].plot(time_steps, target_profile, 'o-', label='目标', alpha=0.8)
            axes[i].plot(time_steps, pred_profile, 's-', label='预测', alpha=0.8)
            axes[i].set_title(f'时序剖面 - 位置 ({y}, {x})')
            axes[i].set_xlabel('时间步')
            axes[i].set_ylabel('数值')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'时序剖面分析 - Case {case_id}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / "results" / f"temporal_profiles_{case_id}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_spectral_analysis(self, target_seq: np.ndarray, 
                             pred_seq: np.ndarray, 
                             case_id: str):
        """绘制频谱分析图"""
        # target_seq, pred_seq: [T, C, H, W]
        T, C, H, W = target_seq.shape
        
        # 选择第一个通道
        target_seq = target_seq[:, 0]  # [T, H, W]
        pred_seq = pred_seq[:, 0]
        
        fig, axes = plt.subplots(2, T, figsize=(4*T, 8))
        
        for t in range(T):
            # 计算2D FFT
            target_fft = np.fft.fft2(target_seq[t])
            pred_fft = np.fft.fft2(pred_seq[t])
            
            # 计算功率谱
            target_power = np.log10(np.abs(np.fft.fftshift(target_fft)) + 1e-8)
            pred_power = np.log10(np.abs(np.fft.fftshift(pred_fft)) + 1e-8)
            
            # 绘制功率谱
            im1 = axes[0, t].imshow(target_power, cmap='viridis')
            axes[0, t].set_title(f'目标功率谱 t={t+1}')
            axes[0, t].axis('off')
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046)
            
            im2 = axes[1, t].imshow(pred_power, cmap='viridis')
            axes[1, t].set_title(f'预测功率谱 t={t+1}')
            axes[1, t].axis('off')
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046)
        
        plt.suptitle(f'频谱分析 - Case {case_id}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / "results" / f"spectral_analysis_{case_id}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_comparison_animation(self, input_seq: np.ndarray,
                                  target_seq: np.ndarray,
                                  pred_seq: np.ndarray,
                                  case_id: str,
                                  fps: int = 5):
        """创建对比动画"""
        # input_seq: [T_in, C, H, W]
        # target_seq, pred_seq: [T_out, C, H, W]
        
        # 选择第一个通道
        input_seq = input_seq[:, 0] if input_seq.ndim == 4 else input_seq
        target_seq = target_seq[:, 0] if target_seq.ndim == 4 else target_seq
        pred_seq = pred_seq[:, 0] if pred_seq.ndim == 4 else pred_seq
        
        T_in, T_out = input_seq.shape[0], target_seq.shape[0]
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 设置颜色范围
        all_data = np.concatenate([input_seq.flatten(), target_seq.flatten(), pred_seq.flatten()])
        vmin, vmax = all_data.min(), all_data.max()
        
        # 初始化图像
        im1 = axes[0].imshow(input_seq[0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('输入序列')
        axes[0].axis('off')
        
        im2 = axes[1].imshow(target_seq[0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('目标序列')
        axes[1].axis('off')
        
        im3 = axes[2].imshow(pred_seq[0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('预测序列')
        axes[2].axis('off')
        
        # 添加颜色条
        for ax, im in zip(axes, [im1, im2, im3]):
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        def animate(frame):
            # 输入序列
            if frame < T_in:
                im1.set_array(input_seq[frame])
                axes[0].set_title(f'输入序列 t={frame}')
            else:
                # 保持最后一帧
                axes[0].set_title(f'输入序列 t={T_in-1}')
            
            # 目标和预测序列
            if frame >= T_in:
                out_frame = frame - T_in
                if out_frame < T_out:
                    im2.set_array(target_seq[out_frame])
                    im3.set_array(pred_seq[out_frame])
                    axes[1].set_title(f'目标序列 t={frame}')
                    axes[2].set_title(f'预测序列 t={frame}')
            
            return [im1, im2, im3]
        
        # 创建动画
        total_frames = T_in + T_out
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames, interval=1000//fps, blit=True, repeat=True
        )
        
        # 保存动画
        anim.save(self.save_dir / "animations" / f"comparison_{case_id}.gif", 
                 writer='pillow', fps=fps)
        plt.close()
    
    def create_final_visualizations(self, model: torch.nn.Module, 
                                  test_loader, device: torch.device):
        """创建最终的可视化结果"""
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="生成可视化")):
                if i >= 3:  # 只处理前3个样本
                    break
                
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(device)
                target_seq = batch['target_sequence'].to(device)
                case_id = batch['case_id'][0]
                
                if 'observation_sequence' in batch:
                    obs_seq = batch['observation_sequence'].to(device)
                else:
                    obs_seq = input_seq
                
                # 模型推理
                outputs = model(obs_seq, target_seq, mode='inference')
                pred_seq = outputs['predictions']
                
                # 转换为numpy
                input_np = input_seq[0].detach().cpu().numpy()
                target_np = target_seq[0].detach().cpu().numpy()
                pred_np = pred_seq[0].detach().cpu().numpy()
                obs_np = obs_seq[0].detach().cpu().numpy()
                
                # 生成各种可视化
                if self.config.results.plot_error_maps:
                    self.plot_error_analysis(target_np, pred_np, case_id)
                
                if self.config.results.plot_temporal_profiles:
                    self.plot_temporal_profiles(target_np, pred_np, case_id)
                
                if self.config.results.plot_spectral_analysis:
                    self.plot_spectral_analysis(target_np, pred_np, case_id)
                
                if self.config.results.create_animations:
                    # 创建目标序列动画
                    self.create_temporal_animation(
                        target_np[:, 0], f"目标序列 - Case {case_id}", 
                        f"target_{case_id}"
                    )
                    
                    # 创建预测序列动画
                    self.create_temporal_animation(
                        pred_np[:, 0], f"预测序列 - Case {case_id}", 
                        f"prediction_{case_id}"
                    )
                    
                    # 创建对比动画
                    self.create_comparison_animation(
                        obs_np, target_np, pred_np, case_id
                    )
        
        print(f"✅ 可视化结果已保存到: {self.save_dir}")


class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(self, results: Dict[str, Dict], 
                              save_name: str = "metrics_comparison"):
        """绘制多个实验的指标对比"""
        metrics = ['rel_l2', 'mae', 'psnr', 'ssim']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            exp_names = []
            values = []
            
            for exp_name, exp_results in results.items():
                if metric in exp_results:
                    exp_names.append(exp_name)
                    values.append(exp_results[metric])
            
            if values:
                axes[i].bar(exp_names, values, alpha=0.7)
                axes[i].set_title(f'{metric.upper()} 对比')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_metrics_table(self, results: Dict[str, Dict], 
                           save_name: str = "metrics_table"):
        """创建指标对比表格"""
        import pandas as pd
        
        # 转换为DataFrame
        df = pd.DataFrame(results).T
        
        # 保存为CSV
        df.to_csv(self.save_dir / f"{save_name}.csv")
        
        # 创建可视化表格
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.round(4).values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('实验结果对比表', fontsize=16, pad=20)
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return df