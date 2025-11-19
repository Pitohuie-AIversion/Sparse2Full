#!/usr/bin/env python3
"""时序预测结果可视化工具

提供多种可视化方法来分析时序AR模型的预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import warnings

# 设置matplotlib后端
plt.switch_backend('Agg')
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """可视化配置"""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    cmap: str = 'viridis'
    error_cmap: str = 'Reds'
    save_format: str = 'png'
    animation_fps: int = 10
    font_size: int = 12
    title_size: int = 14
    label_size: int = 10


class TemporalVisualizer:
    """时序预测结果可视化器"""
    
    def __init__(self, config: VisualizationConfig = None):
        """初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or VisualizationConfig()
        
        # 设置matplotlib样式
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'xtick.labelsize': self.config.label_size,
            'ytick.labelsize': self.config.label_size,
            'legend.fontsize': self.config.label_size,
            'figure.titlesize': self.config.title_size
        })
        
        logger.info("时序可视化器初始化完成")
    
    def plot_sequence_comparison(self, 
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               save_path: Union[str, Path],
                               sample_idx: int = 0,
                               channel_idx: int = 0,
                               time_steps: Optional[List[int]] = None,
                               title: str = "时序预测对比") -> None:
        """绘制时序预测对比图
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            sample_idx: 样本索引
            channel_idx: 通道索引
            time_steps: 要显示的时间步列表
            title: 图标题
        """
        # 转换为numpy
        pred_np = self._to_numpy(predictions[sample_idx, :, channel_idx])
        target_np = self._to_numpy(targets[sample_idx, :, channel_idx])
        
        seq_len = pred_np.shape[0]
        if time_steps is None:
            # 选择几个关键时间步
            time_steps = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
            time_steps = [t for t in time_steps if t < seq_len]
        
        n_steps = len(time_steps)
        fig, axes = plt.subplots(3, n_steps, figsize=(4*n_steps, 12))
        
        if n_steps == 1:
            axes = axes.reshape(-1, 1)
        
        # 计算全局颜色范围
        vmin = min(pred_np.min(), target_np.min())
        vmax = max(pred_np.max(), target_np.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        for i, t in enumerate(time_steps):
            # 预测结果
            im1 = axes[0, i].imshow(pred_np[t], cmap=self.config.cmap, norm=norm)
            axes[0, i].set_title(f'预测 t={t}')
            axes[0, i].axis('off')
            
            # 真实值
            im2 = axes[1, i].imshow(target_np[t], cmap=self.config.cmap, norm=norm)
            axes[1, i].set_title(f'真实 t={t}')
            axes[1, i].axis('off')
            
            # 误差
            error = np.abs(pred_np[t] - target_np[t])
            im3 = axes[2, i].imshow(error, cmap=self.config.error_cmap)
            axes[2, i].set_title(f'误差 t={t}')
            axes[2, i].axis('off')
            
            # 添加颜色条
            if i == n_steps - 1:
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
                plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=self.config.title_size)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时序对比图已保存至: {save_path}")
    
    def plot_error_evolution(self,
                           predictions: torch.Tensor,
                           targets: torch.Tensor,
                           save_path: Union[str, Path],
                           metrics: List[str] = ['mse', 'mae', 'rel_l2'],
                           title: str = "误差演化曲线") -> None:
        """绘制误差随时间演化的曲线
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            metrics: 要计算的指标列表
            title: 图标题
        """
        pred_np = self._to_numpy(predictions)
        target_np = self._to_numpy(targets)
        
        batch_size, seq_len = pred_np.shape[:2]
        
        # 计算各种误差指标
        errors = {metric: [] for metric in metrics}
        
        for t in range(seq_len):
            pred_t = pred_np[:, t]
            target_t = target_np[:, t]
            
            if 'mse' in metrics:
                mse = np.mean((pred_t - target_t) ** 2)
                errors['mse'].append(mse)
            
            if 'mae' in metrics:
                mae = np.mean(np.abs(pred_t - target_t))
                errors['mae'].append(mae)
            
            if 'rel_l2' in metrics:
                rel_l2 = np.sqrt(np.mean((pred_t - target_t) ** 2)) / np.sqrt(np.mean(target_t ** 2))
                errors['rel_l2'].append(rel_l2)
        
        # 绘制曲线
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        time_steps = range(seq_len)
        
        for i, metric in enumerate(metrics):
            axes[i].plot(time_steps, errors[metric], 'b-', linewidth=2, marker='o', markersize=4)
            axes[i].set_xlabel('时间步')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} 演化')
            axes[i].grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(errors[metric]) > 2:
                z = np.polyfit(time_steps, errors[metric], 1)
                p = np.poly1d(z)
                axes[i].plot(time_steps, p(time_steps), 'r--', alpha=0.7, label='趋势线')
                axes[i].legend()
        
        plt.suptitle(title, fontsize=self.config.title_size)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"误差演化图已保存至: {save_path}")
    
    def plot_spatial_error_heatmap(self,
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 save_path: Union[str, Path],
                                 sample_idx: int = 0,
                                 channel_idx: int = 0,
                                 title: str = "空间误差热力图") -> None:
        """绘制空间误差热力图
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            sample_idx: 样本索引
            channel_idx: 通道索引
            title: 图标题
        """
        pred_np = self._to_numpy(predictions[sample_idx, :, channel_idx])
        target_np = self._to_numpy(targets[sample_idx, :, channel_idx])
        
        # 计算时间平均误差
        mean_error = np.mean(np.abs(pred_np - target_np), axis=0)
        
        # 计算最大误差
        max_error = np.max(np.abs(pred_np - target_np), axis=0)
        
        # 计算相对误差
        rel_error = np.mean(np.abs(pred_np - target_np), axis=0) / (np.mean(np.abs(target_np), axis=0) + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 平均绝对误差
        im1 = axes[0].imshow(mean_error, cmap=self.config.error_cmap)
        axes[0].set_title('平均绝对误差')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 最大绝对误差
        im2 = axes[1].imshow(max_error, cmap=self.config.error_cmap)
        axes[1].set_title('最大绝对误差')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 相对误差
        im3 = axes[2].imshow(rel_error, cmap=self.config.error_cmap)
        axes[2].set_title('相对误差')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=self.config.title_size)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"空间误差热力图已保存至: {save_path}")
    
    def create_prediction_animation(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  save_path: Union[str, Path],
                                  sample_idx: int = 0,
                                  channel_idx: int = 0,
                                  title: str = "时序预测动画") -> None:
        """创建时序预测动画
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            sample_idx: 样本索引
            channel_idx: 通道索引
            title: 动画标题
        """
        pred_np = self._to_numpy(predictions[sample_idx, :, channel_idx])
        target_np = self._to_numpy(targets[sample_idx, :, channel_idx])
        
        seq_len = pred_np.shape[0]
        
        # 计算全局颜色范围
        vmin = min(pred_np.min(), target_np.min())
        vmax = max(pred_np.max(), target_np.max())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 初始化图像
        im1 = axes[0].imshow(pred_np[0], cmap=self.config.cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('预测')
        axes[0].axis('off')
        
        im2 = axes[1].imshow(target_np[0], cmap=self.config.cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('真实')
        axes[1].axis('off')
        
        error = np.abs(pred_np[0] - target_np[0])
        im3 = axes[2].imshow(error, cmap=self.config.error_cmap)
        axes[2].set_title('误差')
        axes[2].axis('off')
        
        # 添加颜色条
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 添加时间步显示
        time_text = fig.suptitle(f'{title} - t=0', fontsize=self.config.title_size)
        
        def animate(frame):
            """动画更新函数"""
            # 更新预测图像
            im1.set_array(pred_np[frame])
            
            # 更新真实图像
            im2.set_array(target_np[frame])
            
            # 更新误差图像
            error = np.abs(pred_np[frame] - target_np[frame])
            im3.set_array(error)
            im3.set_clim(vmin=0, vmax=error.max())
            
            # 更新标题
            time_text.set_text(f'{title} - t={frame}')
            
            return [im1, im2, im3, time_text]
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=seq_len, 
            interval=1000//self.config.animation_fps, 
            blit=False, repeat=True
        )
        
        # 保存动画
        save_path = Path(save_path)
        if save_path.suffix.lower() == '.gif':
            anim.save(save_path, writer='pillow', fps=self.config.animation_fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=self.config.animation_fps)
        
        plt.close()
        
        logger.info(f"预测动画已保存至: {save_path}")
    
    def plot_frequency_analysis(self,
                              predictions: torch.Tensor,
                              targets: torch.Tensor,
                              save_path: Union[str, Path],
                              sample_idx: int = 0,
                              channel_idx: int = 0,
                              title: str = "频域分析") -> None:
        """绘制频域分析图
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            sample_idx: 样本索引
            channel_idx: 通道索引
            title: 图标题
        """
        pred_np = self._to_numpy(predictions[sample_idx, :, channel_idx])
        target_np = self._to_numpy(targets[sample_idx, :, channel_idx])
        
        # 计算时间序列的FFT
        pred_fft = np.fft.fft(pred_np, axis=0)
        target_fft = np.fft.fft(target_np, axis=0)
        
        # 计算功率谱
        pred_power = np.mean(np.abs(pred_fft) ** 2, axis=(1, 2))
        target_power = np.mean(np.abs(target_fft) ** 2, axis=(1, 2))
        
        # 频率轴
        freqs = np.fft.fftfreq(pred_np.shape[0])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 功率谱对比
        axes[0, 0].semilogy(freqs[:len(freqs)//2], pred_power[:len(freqs)//2], 'b-', label='预测', linewidth=2)
        axes[0, 0].semilogy(freqs[:len(freqs)//2], target_power[:len(freqs)//2], 'r-', label='真实', linewidth=2)
        axes[0, 0].set_xlabel('频率')
        axes[0, 0].set_ylabel('功率谱密度')
        axes[0, 0].set_title('功率谱对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 功率谱误差
        power_error = np.abs(pred_power - target_power)
        axes[0, 1].semilogy(freqs[:len(freqs)//2], power_error[:len(freqs)//2], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('频率')
        axes[0, 1].set_ylabel('功率谱误差')
        axes[0, 1].set_title('功率谱误差')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 相位对比（选择一个空间点）
        h_mid, w_mid = pred_np.shape[1]//2, pred_np.shape[2]//2
        pred_phase = np.angle(pred_fft[:, h_mid, w_mid])
        target_phase = np.angle(target_fft[:, h_mid, w_mid])
        
        axes[1, 0].plot(freqs[:len(freqs)//2], pred_phase[:len(freqs)//2], 'b-', label='预测', linewidth=2)
        axes[1, 0].plot(freqs[:len(freqs)//2], target_phase[:len(freqs)//2], 'r-', label='真实', linewidth=2)
        axes[1, 0].set_xlabel('频率')
        axes[1, 0].set_ylabel('相位')
        axes[1, 0].set_title(f'相位对比 (位置: {h_mid}, {w_mid})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 相位误差
        phase_error = np.abs(pred_phase - target_phase)
        axes[1, 1].plot(freqs[:len(freqs)//2], phase_error[:len(freqs)//2], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('频率')
        axes[1, 1].set_ylabel('相位误差')
        axes[1, 1].set_title('相位误差')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=self.config.title_size)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"频域分析图已保存至: {save_path}")
    
    def plot_multi_step_comparison(self,
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 save_path: Union[str, Path],
                                 step_intervals: List[int] = None,
                                 sample_idx: int = 0,
                                 channel_idx: int = 0,
                                 title: str = "多步预测对比") -> None:
        """绘制多步预测对比图
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_path: 保存路径
            step_intervals: 步长间隔列表
            sample_idx: 样本索引
            channel_idx: 通道索引
            title: 图标题
        """
        pred_np = self._to_numpy(predictions[sample_idx, :, channel_idx])
        target_np = self._to_numpy(targets[sample_idx, :, channel_idx])
        
        seq_len = pred_np.shape[0]
        
        if step_intervals is None:
            step_intervals = [1, seq_len//4, seq_len//2, seq_len-1]
            step_intervals = [s for s in step_intervals if s < seq_len]
        
        n_steps = len(step_intervals)
        fig, axes = plt.subplots(n_steps, 3, figsize=(12, 4*n_steps))
        
        if n_steps == 1:
            axes = axes.reshape(1, -1)
        
        for i, step in enumerate(step_intervals):
            # 计算累积误差
            cumulative_error = np.mean(np.abs(pred_np[:step+1] - target_np[:step+1]), axis=0)
            
            # 当前时刻预测
            axes[i, 0].imshow(pred_np[step], cmap=self.config.cmap)
            axes[i, 0].set_title(f'预测 t={step}')
            axes[i, 0].axis('off')
            
            # 当前时刻真实值
            axes[i, 1].imshow(target_np[step], cmap=self.config.cmap)
            axes[i, 1].set_title(f'真实 t={step}')
            axes[i, 1].axis('off')
            
            # 累积误差
            im = axes[i, 2].imshow(cumulative_error, cmap=self.config.error_cmap)
            axes[i, 2].set_title(f'累积误差 t=0~{step}')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=self.config.title_size)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"多步预测对比图已保存至: {save_path}")
    
    def create_comprehensive_report(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  save_dir: Union[str, Path],
                                  sample_idx: int = 0,
                                  channel_idx: int = 0,
                                  prefix: str = "temporal_analysis") -> Dict[str, str]:
        """创建综合分析报告
        
        Args:
            predictions: 预测结果 [B, T, C, H, W]
            targets: 真实值 [B, T, C, H, W]
            save_dir: 保存目录
            sample_idx: 样本索引
            channel_idx: 通道索引
            prefix: 文件名前缀
            
        Returns:
            生成的文件路径字典
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # 1. 时序对比图
        comparison_path = save_dir / f"{prefix}_comparison.{self.config.save_format}"
        self.plot_sequence_comparison(predictions, targets, comparison_path, sample_idx, channel_idx)
        generated_files['comparison'] = str(comparison_path)
        
        # 2. 误差演化曲线
        error_evolution_path = save_dir / f"{prefix}_error_evolution.{self.config.save_format}"
        self.plot_error_evolution(predictions, targets, error_evolution_path)
        generated_files['error_evolution'] = str(error_evolution_path)
        
        # 3. 空间误差热力图
        spatial_error_path = save_dir / f"{prefix}_spatial_error.{self.config.save_format}"
        self.plot_spatial_error_heatmap(predictions, targets, spatial_error_path, sample_idx, channel_idx)
        generated_files['spatial_error'] = str(spatial_error_path)
        
        # 4. 频域分析
        frequency_path = save_dir / f"{prefix}_frequency_analysis.{self.config.save_format}"
        self.plot_frequency_analysis(predictions, targets, frequency_path, sample_idx, channel_idx)
        generated_files['frequency_analysis'] = str(frequency_path)
        
        # 5. 多步预测对比
        multi_step_path = save_dir / f"{prefix}_multi_step.{self.config.save_format}"
        self.plot_multi_step_comparison(predictions, targets, multi_step_path, None, sample_idx, channel_idx)
        generated_files['multi_step'] = str(multi_step_path)
        
        # 6. 预测动画（可选）
        try:
            animation_path = save_dir / f"{prefix}_animation.gif"
            self.create_prediction_animation(predictions, targets, animation_path, sample_idx, channel_idx)
            generated_files['animation'] = str(animation_path)
        except Exception as e:
            logger.warning(f"动画生成失败: {e}")
        
        logger.info(f"综合分析报告已生成至: {save_dir}")
        logger.info(f"生成的文件: {list(generated_files.keys())}")
        
        return generated_files
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将张量转换为numpy数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor


def create_visualization_summary(file_paths: Dict[str, str], 
                               save_path: Union[str, Path]) -> None:
    """创建可视化结果汇总HTML页面
    
    Args:
        file_paths: 生成的文件路径字典
        save_path: HTML文件保存路径
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>时序预测可视化报告</title>
        <style>
            body { font-family: system-ui, -apple-system, 'Noto Sans', 'Noto Sans CJK SC', 'Source Han Sans SC', 'DejaVu Sans', Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; }
            .image-container { text-align: center; margin: 10px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>时序预测可视化分析报告</h1>
    """
    
    sections = {
        'comparison': '时序预测对比',
        'error_evolution': '误差演化分析',
        'spatial_error': '空间误差分布',
        'frequency_analysis': '频域分析',
        'multi_step': '多步预测对比',
        'animation': '时序预测动画'
    }
    
    for key, title in sections.items():
        if key in file_paths:
            file_path = Path(file_paths[key])
            if file_path.exists():
                html_content += f"""
                <div class="section">
                    <h2>{title}</h2>
                    <div class="image-container">
                """
                
                if key == 'animation':
                    html_content += f'<img src="{file_path.name}" alt="{title}">'
                else:
                    html_content += f'<img src="{file_path.name}" alt="{title}">'
                
                html_content += """
                    </div>
                </div>
                """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"可视化汇总页面已保存至: {save_path}")


# 便捷函数
def quick_visualize(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   save_dir: Union[str, Path],
                   sample_idx: int = 0,
                   channel_idx: int = 0,
                   create_html: bool = True) -> Dict[str, str]:
    """快速可视化函数
    
    Args:
        predictions: 预测结果 [B, T, C, H, W]
        targets: 真实值 [B, T, C, H, W]
        save_dir: 保存目录
        sample_idx: 样本索引
        channel_idx: 通道索引
        create_html: 是否创建HTML汇总页面
        
    Returns:
        生成的文件路径字典
    """
    visualizer = TemporalVisualizer()
    
    file_paths = visualizer.create_comprehensive_report(
        predictions, targets, save_dir, sample_idx, channel_idx
    )
    
    if create_html:
        html_path = Path(save_dir) / "visualization_report.html"
        create_visualization_summary(file_paths, html_path)
        file_paths['html_report'] = str(html_path)
    
    return file_paths