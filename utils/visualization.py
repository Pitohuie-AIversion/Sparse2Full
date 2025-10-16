"""可视化工具模块

提供PDEBench稀疏观测重建系统的完整可视化功能
支持热图、功率谱、误差分析、横向对比等多种可视化需求

按照技术架构文档要求，生成标准的GT/Pred/Err热图、功率谱对数显示、边界带误差分析等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import seaborn as sns
from pathlib import Path
import json
from matplotlib.colors import LogNorm
import warnings

# 设置matplotlib参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class PDEBenchVisualizer:
    """PDEBench可视化器
    
    提供完整的可视化功能，包括：
    - GT/Pred/Error热图对比
    - 四联图可视化（Observation + GT + Pred + Error）
    - 功率谱分析（对数显示）
    - 边界带误差分析
    - 频域分段误差可视化
    - 横向模型对比图表
    - SVG格式输出支持
    """
    
    def __init__(self, save_dir: str, dpi: int = 300, output_format: str = 'png', logger=None):
        """
        Args:
            save_dir: 保存目录
            dpi: 图像分辨率
            output_format: 输出格式 ('png', 'svg', 'pdf')
            logger: 日志记录器
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.output_format = output_format.lower()
        self.logger = logger
        
        # 设置颜色映射
        self.field_cmap = 'RdBu_r'
        self.error_cmap = 'Reds'
        self.spectrum_cmap = 'hot'
        
        # 创建子目录
        (self.save_dir / 'fields').mkdir(exist_ok=True)
        (self.save_dir / 'spectra').mkdir(exist_ok=True)
        (self.save_dir / 'analysis').mkdir(exist_ok=True)
        (self.save_dir / 'comparisons').mkdir(exist_ok=True)

    def _save_figure(self, save_name: str, subdir: str = '') -> str:
        """保存图像的统一接口"""
        if subdir:
            save_path = self.save_dir / subdir / f"{save_name}.{self.output_format}"
        else:
            save_path = self.save_dir / f"{save_name}.{self.output_format}"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return str(save_path)

    def create_quadruplet_visualization(self,
                                      observed: torch.Tensor,
                                      gt: torch.Tensor,
                                      pred: torch.Tensor,
                                      save_name: str = "quadruplet",
                                      figsize: Tuple[int, int] = (20, 5),
                                      channel_idx: int = 0,
                                      title: str = None) -> str:
        """创建四联图可视化：Observation + GT + Prediction + Error
        
        Args:
            observed: 观测数据张量
            gt: 真值张量
            pred: 预测张量
            save_name: 保存文件名
            figsize: 图像尺寸
            channel_idx: 通道索引
            title: 图像标题
            
        Returns:
            保存路径
        """
        # 处理输入张量
        if len(observed.shape) == 4:
            observed = observed[0]
            gt = gt[0]
            pred = pred[0]
        
        if len(observed.shape) == 3 and observed.shape[0] > 1:
            observed = observed[channel_idx]
            gt = gt[channel_idx]
            pred = pred[channel_idx]
        elif len(observed.shape) == 3:
            observed = observed.squeeze(0)
            gt = gt.squeeze(0)
            pred = pred.squeeze(0)
        
        # 转换为numpy
        observed_np = observed.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # 确保是2D数组
        if observed_np.ndim == 1:
            # 如果是1D，尝试重塑为2D
            size = int(np.sqrt(observed_np.shape[0]))
            if size * size == observed_np.shape[0]:
                observed_np = observed_np.reshape(size, size)
                gt_np = gt_np.reshape(size, size)
                pred_np = pred_np.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D array of size {observed_np.shape[0]} to 2D")
        
        # 计算误差
        error_np = np.abs(pred_np - gt_np)
        
        # 计算全局值域
        vmin = min(observed_np.min(), gt_np.min(), pred_np.min())
        vmax = max(observed_np.max(), gt_np.max(), pred_np.max())
        
        # 创建子图
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 观测数据
        im1 = axes[0].imshow(observed_np, cmap=self.field_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Observation', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # 真值
        im2 = axes[1].imshow(gt_np, cmap=self.field_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 预测
        im3 = axes[2].imshow(pred_np, cmap=self.field_cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Prediction', fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # 误差
        im4 = axes[3].imshow(error_np, cmap=self.error_cmap, vmin=0, vmax=error_np.max())
        axes[3].set_title('Absolute Error', fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name, 'fields')

    def plot_field_comparison(self,
                            gt: torch.Tensor,
                            pred: torch.Tensor,
                            save_name: str = "field_comparison",
                            figsize: Tuple[int, int] = (15, 5),
                            channel_idx: int = 0,
                            title: str = None) -> str:
        """绘制场对比图：GT + Prediction + Error
        
        Args:
            gt: 真值张量
            pred: 预测张量
            save_name: 保存文件名
            figsize: 图像尺寸
            channel_idx: 通道索引
            title: 图像标题
            
        Returns:
            保存路径
        """
        # 处理输入张量
        if isinstance(gt, torch.Tensor):
            if len(gt.shape) == 4:
                gt = gt[0]
                pred = pred[0]
            
            if gt.shape[0] > 1:
                gt = gt[channel_idx]
                pred = pred[channel_idx]
            else:
                gt = gt.squeeze(0)
                pred = pred.squeeze(0)
        else:
            # 处理numpy数组
            gt = np.array(gt)
            pred = np.array(pred)
            if len(gt.shape) > 2:
                gt = gt.squeeze()
                pred = pred.squeeze()
        
        # 转换为numpy
        if isinstance(gt, torch.Tensor):
            gt_np = gt.detach().cpu().numpy()
        else:
            gt_np = np.array(gt)
            
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = np.array(pred)
        
        # 计算误差
        error_np = np.abs(pred_np - gt_np)
        
        # 计算全局值域
        vmin = min(gt_np.min(), pred_np.min())
        vmax = max(gt_np.max(), pred_np.max())
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 真值
        im1 = axes[0].imshow(gt_np, cmap=self.field_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # 预测
        im2 = axes[1].imshow(pred_np, cmap=self.field_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Prediction', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 误差
        im3 = axes[2].imshow(error_np, cmap=self.error_cmap, vmin=0, vmax=error_np.max())
        axes[2].set_title('Absolute Error', fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name, 'fields')

    def create_correlation_heatmap(self, 
                                 corr_matrix: np.ndarray,
                                 save_name: str = "correlation_heatmap",
                                 figsize: Tuple[int, int] = (12, 10)) -> str:
        """创建相关性热图
        
        Args:
            corr_matrix: 相关性矩阵
            save_name: 保存文件名
            figsize: 图像尺寸
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建掩码以隐藏上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 创建热图
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Metrics Correlation Matrix')
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name)

    def create_metrics_summary_plot(self, 
                                  metrics_data: Dict[str, Dict[str, float]],
                                  save_name: str = "metrics_summary",
                                  figsize: Tuple[int, int] = (12, 8)) -> str:
        """创建指标汇总图
        
        Args:
            metrics_data: {'method_name': {'metric_name': value}}
            save_name: 保存文件名
            figsize: 图像尺寸
            
        Returns:
            保存路径
        """
        # 提取数据
        methods = list(metrics_data.keys())
        metrics = list(next(iter(metrics_data.values())).keys())
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(methods), len(metrics)))
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = metrics_data[method].get(metric, 0)
        
        # 创建热图
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        
        # 添加数值标注
        for i in range(len(methods)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title("Metrics Summary")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name)

    def plot_training_curves(self,
                           train_losses: List[float],
                           val_losses: List[float],
                           save_name: str = "training_curves",
                           figsize: Tuple[int, int] = (12, 8),
                           additional_metrics: Dict[str, List[float]] = None) -> str:
        """绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_name: 保存文件名
            figsize: 图像尺寸
            additional_metrics: 额外指标字典
            
        Returns:
            保存路径
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 确定子图数量
        n_plots = 2 if additional_metrics is None else 2 + len(additional_metrics)
        n_cols = min(n_plots, 2)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # 损失曲线
        axes[0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 学习率曲线（如果有的话）
        if additional_metrics and 'learning_rate' in additional_metrics:
            axes[1].plot(epochs, additional_metrics['learning_rate'], color='green', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            plot_idx = 2
        else:
            plot_idx = 1
        
        # 其他指标
        if additional_metrics:
            for metric_name, values in additional_metrics.items():
                if metric_name == 'learning_rate':
                    continue
                if plot_idx < len(axes):
                    axes[plot_idx].plot(epochs, values, linewidth=2)
                    axes[plot_idx].set_xlabel('Epoch')
                    axes[plot_idx].set_ylabel(metric_name.upper())
                    axes[plot_idx].set_title(f'{metric_name.upper()} over Training')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
        
        # 隐藏多余的子图
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name, 'analysis')

    def create_power_spectrum_plot(self,
                                 field: torch.Tensor,
                                 save_name: str = "power_spectrum",
                                 figsize: Tuple[int, int] = (10, 6),
                                 channel_idx: int = 0) -> str:
        """创建功率谱图
        
        Args:
            field: 场数据张量
            save_name: 保存文件名
            figsize: 图像尺寸
            channel_idx: 通道索引
            
        Returns:
            保存路径
        """
        # 处理输入张量
        if isinstance(field, torch.Tensor):
            if len(field.shape) == 4:
                field = field[0]
            if len(field.shape) == 3 and field.shape[0] > 1:
                field = field[channel_idx]
            elif len(field.shape) == 3:
                field = field.squeeze(0)
            field_np = field.detach().cpu().numpy()
        else:
            field_np = np.array(field)
            
        # 确保是2D数组
        if field_np.ndim > 2:
            field_np = field_np.squeeze()
        if field_np.ndim == 1:
            # 如果是1D，尝试重塑为2D
            size = int(np.sqrt(field_np.shape[0]))
            if size * size == field_np.shape[0]:
                field_np = field_np.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D array of size {field_np.shape[0]} to 2D")
        
        # 确保是2D数组
        if field_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got {field_np.ndim}D array with shape {field_np.shape}")
        
        # 计算2D FFT
        fft_field = np.fft.fft2(field_np)
        fft_shifted = np.fft.fftshift(fft_field)
        power_spectrum = np.abs(fft_shifted) ** 2
        
        # 计算径向平均功率谱
        h, w = power_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # 创建径向坐标
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # 计算径向平均
        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 2D功率谱
        im1 = ax1.imshow(np.log10(power_spectrum + 1e-10), cmap=self.spectrum_cmap)
        ax1.set_title('2D Power Spectrum (log scale)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 径向平均功率谱
        k_values = np.arange(len(radial_profile))
        ax2.loglog(k_values[1:], radial_profile[1:], 'b-', linewidth=2)
        ax2.set_xlabel('Wavenumber k')
        ax2.set_ylabel('Power')
        ax2.set_title('Radial Power Spectrum')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name, 'spectra')

    def create_boundary_analysis(self,
                               gt: torch.Tensor,
                               pred: torch.Tensor,
                               boundary_width: int = 16,
                               save_name: str = "boundary_analysis",
                               figsize: Tuple[int, int] = (15, 5)) -> str:
        """创建边界分析图
        
        Args:
            gt: 真值张量
            pred: 预测张量
            boundary_width: 边界宽度
            save_name: 保存文件名
            figsize: 图像尺寸
            
        Returns:
            保存路径
        """
        # 处理输入张量
        if isinstance(gt, torch.Tensor):
            if len(gt.shape) == 4:
                gt = gt[0, 0]
                pred = pred[0, 0]
            elif len(gt.shape) == 3:
                gt = gt[0]
                pred = pred[0]
            gt_np = gt.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
        else:
            gt_np = np.array(gt)
            pred_np = np.array(pred)
        
        h, w = gt_np.shape
        
        # 创建边界掩码
        boundary_mask = np.zeros((h, w), dtype=bool)
        boundary_mask[:boundary_width, :] = True  # 上边界
        boundary_mask[-boundary_width:, :] = True  # 下边界
        boundary_mask[:, :boundary_width] = True  # 左边界
        boundary_mask[:, -boundary_width:] = True  # 右边界
        
        # 计算误差
        error = np.abs(pred_np - gt_np)
        
        # 边界和内部误差
        boundary_error = error[boundary_mask]
        interior_error = error[~boundary_mask]
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 边界掩码可视化
        axes[0].imshow(boundary_mask, cmap='RdYlBu', alpha=0.7)
        axes[0].contour(gt_np, levels=10, colors='black', alpha=0.5, linewidths=0.5)
        axes[0].set_title(f'Boundary Region (width={boundary_width})')
        axes[0].axis('off')
        
        # 误差分布
        im2 = axes[1].imshow(error, cmap=self.error_cmap)
        axes[1].set_title('Error Distribution')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 边界vs内部误差统计
        axes[2].hist(boundary_error.flatten(), bins=50, alpha=0.7, label=f'Boundary (μ={boundary_error.mean():.4f})', density=True)
        axes[2].hist(interior_error.flatten(), bins=50, alpha=0.7, label=f'Interior (μ={interior_error.mean():.4f})', density=True)
        axes[2].set_xlabel('Absolute Error')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Error Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        return self._save_figure(save_name, 'analysis')

    def set_output_format(self, format: str) -> None:
        """设置输出格式
        
        Args:
            format: 输出格式 ('png', 'svg', 'pdf')
        """
        self.output_format = format.lower()
        
    def get_supported_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return ['png', 'svg', 'pdf']


# 统一的可视化API接口
def create_comparison_plot(observed: np.ndarray, 
                          gt: np.ndarray, 
                          pred: np.ndarray,
                          titles: List[str] = None,
                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """创建对比图的便捷接口
    
    Args:
        observed: 观测数据
        gt: 真值数据
        pred: 预测数据
        titles: 子图标题列表
        figsize: 图像尺寸
        
    Returns:
        matplotlib图像对象
    """
    if titles is None:
        titles = ['Observed', 'Ground Truth', 'Prediction']
    
    # 计算误差
    error = np.abs(pred - gt)
    
    # 计算全局值域
    vmin = min(observed.min(), gt.min(), pred.min())
    vmax = max(observed.max(), gt.max(), pred.max())
    
    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 观测数据
    im1 = axes[0].imshow(observed, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0], fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 真值
    im2 = axes[1].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1], fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 预测
    im3 = axes[2].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title(titles[2], fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # 误差
    im4 = axes[3].imshow(error, cmap='Reds', vmin=0, vmax=error.max())
    axes[3].set_title('Absolute Error', fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], shrink=0.8)
    
    plt.tight_layout()
    return fig


def create_spectrum_plot(gt: np.ndarray, pred: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """创建频谱对比图的便捷接口
    
    Args:
        gt: 真值数据
        pred: 预测数据
        figsize: 图像尺寸
        
    Returns:
        matplotlib图像对象
    """
    # 计算2D FFT
    fft_gt = np.fft.fft2(gt)
    fft_pred = np.fft.fft2(pred)
    
    # 计算功率谱
    power_gt = np.abs(np.fft.fftshift(fft_gt)) ** 2
    power_pred = np.abs(np.fft.fftshift(fft_pred)) ** 2
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # GT功率谱
    im1 = ax1.imshow(np.log10(power_gt + 1e-10), cmap='hot')
    ax1.set_title('Ground Truth Power Spectrum (log)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Pred功率谱
    im2 = ax2.imshow(np.log10(power_pred + 1e-10), cmap='hot')
    ax2.set_title('Prediction Power Spectrum (log)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    return fig


def create_field_comparison(observed: torch.Tensor, 
                          gt: torch.Tensor, 
                          pred: torch.Tensor,
                          save_dir: str,
                          save_name: str = "field_comparison",
                          output_format: str = 'png') -> str:
    """创建场对比可视化的便捷接口
    
    Args:
        observed: 观测数据
        gt: 真值数据
        pred: 预测数据
        save_dir: 保存目录
        save_name: 保存文件名
        output_format: 输出格式
        
    Returns:
        保存路径
    """
    visualizer = PDEBenchVisualizer(save_dir, output_format=output_format)
    return visualizer.create_quadruplet_visualization(observed, gt, pred, save_name)


def create_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         save_dir: str,
                         save_name: str = "training_curves",
                         output_format: str = 'png') -> str:
    """创建训练曲线的便捷接口
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 保存目录
        save_name: 保存文件名
        output_format: 输出格式
        
    Returns:
        保存路径
    """
    visualizer = PDEBenchVisualizer(save_dir, output_format=output_format)
    return visualizer.plot_training_curves(train_losses, val_losses, save_name)


def create_power_spectrum(field: torch.Tensor,
                        save_dir: str,
                        save_name: str = "power_spectrum",
                        output_format: str = 'png') -> str:
    """创建功率谱的便捷接口
    
    Args:
        field: 场数据
        save_dir: 保存目录
        save_name: 保存文件名
        output_format: 输出格式
        
    Returns:
        保存路径
    """
    visualizer = PDEBenchVisualizer(save_dir, output_format=output_format)
    return visualizer.create_power_spectrum_plot(field, save_name)


if __name__ == "__main__":
    # 测试代码
    print("PDEBench可视化工具模块")