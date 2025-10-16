"""
PDEBench稀疏观测重建系统 - 完整评估脚本

实现多维度指标计算、可视化生成和横向对比表格功能
严格遵循开发手册的黄金法则和技术架构要求
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from datasets.pdebench import PDEBenchDataModule
from models.swin_unet import SwinUNet
from models.hybrid import HybridModel
from models.mlp import MLPModel
from ops.loss import TotalLoss
from ops.degradation import apply_degradation_operator
from utils.reproducibility import set_deterministic_mode
from utils.visualization import PDEBenchVisualizer


class MetricsCalculator:
    """指标计算器
    
    实现开发手册要求的所有评估指标：
    - Rel-L2、MAE、PSNR、SSIM
    - fRMSE-low/mid/high、bRMSE、cRMSE
    - H算子一致性误差 ||H(ŷ)−y||
    """
    
    def __init__(self, image_size: Tuple[int, int], boundary_width: int = 16):
        """
        Args:
            image_size: 图像尺寸 (H, W)
            boundary_width: 边界带宽度
        """
        self.image_size = image_size
        self.boundary_width = boundary_width
    
    def compute_rel_l2(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算相对L2误差
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            rel_l2: 相对L2误差 [B, C]
        """
        B, C = pred.shape[:2]
        
        # 计算L2范数
        pred_norm = torch.norm(pred.view(B, C, -1), p=2, dim=2)
        target_norm = torch.norm(target.view(B, C, -1), p=2, dim=2)
        diff_norm = torch.norm((pred - target).view(B, C, -1), p=2, dim=2)
        
        # 相对误差
        rel_l2 = diff_norm / (target_norm + 1e-8)
        
        return rel_l2
    
    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算平均绝对误差
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            mae: 平均绝对误差 [B, C]
        """
        B, C = pred.shape[:2]
        mae = torch.mean(torch.abs(pred - target).view(B, C, -1), dim=2)
        return mae
    
    def compute_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算均方误差
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            mse: 均方误差 [B, C]
        """
        B, C = pred.shape[:2]
        mse = torch.mean((pred - target).pow(2).view(B, C, -1), dim=2)
        return mse
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算峰值信噪比
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            psnr: PSNR [B, C]
        """
        mse = self.compute_mse(pred, target)
        
        # 计算数据范围
        B, C = pred.shape[:2]
        target_max = torch.max(target.view(B, C, -1), dim=2)[0]
        target_min = torch.min(target.view(B, C, -1), dim=2)[0]
        data_range = target_max - target_min
        
        # PSNR计算
        psnr = 20 * torch.log10(data_range / (torch.sqrt(mse) + 1e-8))
        
        return psnr
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, 
                    window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        """计算结构相似性指数
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            window_size: 窗口大小
            sigma: 高斯核标准差
            
        Returns:
            ssim: SSIM [B, C]
        """
        # 简化的SSIM实现
        B, C, H, W = pred.shape
        
        # 计算均值
        mu1 = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = torch.nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        # SSIM常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM计算
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 平均SSIM
        ssim = torch.mean(ssim_map.view(B, C, -1), dim=2)
        
        return ssim
    
    def compute_frmse(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算频域RMSE（低/中/高频段）
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            frmse_dict: 频域RMSE字典
        """
        B, C, H, W = pred.shape
        
        # FFT变换
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # 频率坐标
        kx = torch.fft.fftfreq(W, device=pred.device)
        ky = torch.fft.fftfreq(H, device=pred.device)
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2)
        
        # 频段划分
        k_max = torch.sqrt(torch.tensor((H//2)**2 + (W//2)**2, device=pred.device))
        low_mask = k_mag <= k_max / 3
        mid_mask = (k_mag > k_max / 3) & (k_mag <= 2 * k_max / 3)
        high_mask = k_mag > 2 * k_max / 3
        
        frmse_dict = {}
        
        for name, mask in [('low', low_mask), ('mid', mid_mask), ('high', high_mask)]:
            # 应用频段掩码
            pred_fft_masked = pred_fft * mask
            target_fft_masked = target_fft * mask
            
            # 逆FFT
            pred_filtered = torch.fft.ifft2(pred_fft_masked).real
            target_filtered = torch.fft.ifft2(target_fft_masked).real
            
            # 计算RMSE
            mse = torch.mean((pred_filtered - target_filtered).pow(2).view(B, C, -1), dim=2)
            rmse = torch.sqrt(mse)
            
            frmse_dict[f'frmse_{name}'] = rmse
        
        return frmse_dict
    
    def compute_brmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边界RMSE
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            brmse: 边界RMSE [B, C]
        """
        B, C, H, W = pred.shape
        w = self.boundary_width
        
        # 创建边界掩码
        mask = torch.zeros((H, W), device=pred.device)
        mask[:w, :] = 1  # 上边界
        mask[-w:, :] = 1  # 下边界
        mask[:, :w] = 1  # 左边界
        mask[:, -w:] = 1  # 右边界
        
        # 应用掩码
        pred_boundary = pred * mask.view(1, 1, H, W)
        target_boundary = target * mask.view(1, 1, H, W)
        
        # 计算RMSE
        diff_sq = (pred_boundary - target_boundary).pow(2)
        mse = torch.sum(diff_sq.view(B, C, -1), dim=2) / torch.sum(mask)
        brmse = torch.sqrt(mse)
        
        return brmse
    
    def compute_crmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算中心RMSE
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            crmse: 中心RMSE [B, C]
        """
        B, C, H, W = pred.shape
        w = self.boundary_width
        
        # 中心区域
        pred_center = pred[:, :, w:-w, w:-w]
        target_center = target[:, :, w:-w, w:-w]
        
        # 计算RMSE
        mse = torch.mean((pred_center - target_center).pow(2).view(B, C, -1), dim=2)
        crmse = torch.sqrt(mse)
        
        return crmse
    
    def compute_dc_error(self, pred: torch.Tensor, observation: torch.Tensor, 
                        task_params: Dict, denormalize_fn: Optional[callable] = None) -> torch.Tensor:
        """计算数据一致性误差 ||H(ŷ)−y||
        
        Args:
            pred: 预测结果 [B, C, H, W] (z-score域)
            observation: 观测数据 [B, C_obs, H_obs, W_obs]
            task_params: 任务参数
            denormalize_fn: 反归一化函数
            
        Returns:
            dc_error: 数据一致性误差 [B, C]
        """
        # 反归一化到原值域
        if denormalize_fn is not None:
            pred_orig = denormalize_fn(pred)
        else:
            pred_orig = pred
        
        # 应用观测算子H
        pred_obs = apply_degradation_operator(pred_orig, task_params)
        
        # 计算误差
        B, C = pred.shape[:2]
        diff = pred_obs - observation
        dc_error = torch.norm(diff.view(B, C, -1), p=2, dim=2)  # [B, C]
        
        return dc_error
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                           observation: torch.Tensor, task_params: Dict,
                           denormalize_fn: Optional[callable] = None) -> Dict[str, torch.Tensor]:
        """计算所有指标
        
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            observation: 观测数据
            task_params: 任务参数
            denormalize_fn: 反归一化函数
            
        Returns:
            metrics: 所有指标的字典
        """
        metrics = {}
        
        # 基础指标
        metrics['rel_l2'] = self.compute_rel_l2(pred, target)
        metrics['mae'] = self.compute_mae(pred, target)
        metrics['mse'] = self.compute_mse(pred, target)
        metrics['psnr'] = self.compute_psnr(pred, target)
        metrics['ssim'] = self.compute_ssim(pred, target)
        
        # 频域指标
        frmse_dict = self.compute_frmse(pred, target)
        metrics.update(frmse_dict)
        
        # 空间指标
        metrics['brmse'] = self.compute_brmse(pred, target)
        metrics['crmse'] = self.compute_crmse(pred, target)
        
        # H算子一致性
        metrics['h_error'] = self.compute_dc_error(
            pred, observation, task_params, denormalize_fn
        )
        
        return metrics


class Visualizer:
    """可视化生成器
    
    生成开发手册要求的标准图：
    - GT/Pred/Err热图（统一色标）
    - 功率谱（log）
    - 边界带局部放大
    - 失败案例分析
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用统一可视化工具
        self.visualizer = PDEBenchVisualizer(str(self.output_dir))
    
    def plot_comparison_heatmaps(
        self, 
        target: np.ndarray, 
        pred: np.ndarray, 
        case_name: str,
        channel_names: Optional[List[str]] = None
    ):
        """绘制GT/Pred/Error对比热图
        
        Args:
            target: 真实值 [C, H, W]
            pred: 预测值 [C, H, W]
            case_name: 案例名称
            channel_names: 通道名称列表
        """
        # 使用统一可视化工具
        self.visualizer.create_quadruplet_visualization(
            observation=None,  # 不显示观测
            ground_truth=target,
            prediction=pred,
            save_path=self.output_dir / f'{case_name}_comparison.png',
            title=f'Comparison: {case_name}',
            channel_names=channel_names
        )
    
    def plot_power_spectrum(
        self, 
        target: np.ndarray, 
        pred: np.ndarray, 
        case_name: str,
        channel_names: Optional[List[str]] = None
    ):
        """绘制功率谱对比图
        
        Args:
            target: 真实值 [C, H, W]
            pred: 预测值 [C, H, W]
            case_name: 案例名称
            channel_names: 通道名称列表
        """
        # 使用统一可视化工具
        self.visualizer.create_power_spectrum_comparison(
            ground_truth=target,
            prediction=pred,
            save_path=self.output_dir / f'{case_name}_power_spectrum.png',
            title=f'Power Spectrum: {case_name}',
            channel_names=channel_names
        )
    
    def plot_boundary_analysis(
        self, 
        target: np.ndarray, 
        pred: np.ndarray, 
        case_name: str,
        boundary_width: int = 16,
        channel_names: Optional[List[str]] = None
    ):
        """绘制边界带误差分析图
        
        Args:
            target: 真实值 [C, H, W]
            pred: 预测值 [C, H, W]
            case_name: 案例名称
            boundary_width: 边界带宽度
            channel_names: 通道名称列表
        """
        # 使用统一可视化工具
        self.visualizer.create_boundary_analysis(
            ground_truth=target,
            prediction=pred,
            save_path=self.output_dir / f'{case_name}_boundary_analysis.png',
            title=f'Boundary Analysis: {case_name}',
            boundary_width=boundary_width,
            channel_names=channel_names
        )
    
    def plot_failure_case_analysis(
        self, 
        target: np.ndarray, 
        pred: np.ndarray, 
        metrics: Dict[str, float],
        case_name: str,
        failure_type: str = "unknown"
    ):
        """绘制失败案例分析图
        
        Args:
            target: 真实值 [C, H, W]
            pred: 预测值 [C, H, W]
            metrics: 指标字典
            case_name: 案例名称
            failure_type: 失败类型
        """
        # 使用统一可视化工具
        self.visualizer.create_failure_case_analysis(
            ground_truth=target,
            prediction=pred,
            metrics=metrics,
            save_path=self.output_dir / f'{case_name}_failure_analysis.png',
            title=f'Failure Analysis: {case_name}',
            failure_type=failure_type
        )


class ModelComparator:
    """模型对比器
    
    生成横向对比表格：
    - 模型性能对比（均值±标准差）
    - 资源消耗统计（Params/FLOPs/显存/时延）
    - 显著性检验结果
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}  # 存储所有模型结果
    
    def add_model_results(
        self, 
        model_name: str, 
        metrics: Dict[str, List[float]], 
        resource_stats: Dict[str, float],
        config: Dict[str, Any]
    ):
        """添加模型结果
        
        Args:
            model_name: 模型名称
            metrics: 指标字典，每个指标包含多个种子的结果
            resource_stats: 资源统计
            config: 配置信息
        """
        self.results[model_name] = {
            'metrics': metrics,
            'resource_stats': resource_stats,
            'config': config
        }
    
    def generate_performance_table(self) -> str:
        """生成性能对比表格
        
        Returns:
            table_str: 表格字符串（Markdown格式）
        """
        if not self.results:
            return "No results to compare."
        
        # 表头
        metric_names = ['Rel-L2', 'MAE', 'PSNR', 'SSIM', 'fRMSE-low', 'fRMSE-mid', 'fRMSE-high', 'bRMSE', 'cRMSE', 'H-Error']
        header = "| Model | " + " | ".join(metric_names) + " |\n"
        separator = "|" + "|".join([" --- " for _ in range(len(metric_names) + 1)]) + "|\n"
        
        table_str = header + separator
        
        # 数据行
        for model_name, data in self.results.items():
            metrics = data['metrics']
            row = f"| {model_name} |"
            
            for metric_name in metric_names:
                metric_key = metric_name.lower().replace('-', '_')
                if metric_key in metrics:
                    values = metrics[metric_key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row += f" {mean_val:.4f}±{std_val:.4f} |"
                else:
                    row += " N/A |"
            
            table_str += row + "\n"
        
        return table_str
    
    def generate_resource_table(self) -> str:
        """生成资源消耗对比表格
        
        Returns:
            table_str: 表格字符串（Markdown格式）
        """
        if not self.results:
            return "No results to compare."
        
        # 表头
        resource_names = ['Params(M)', 'FLOPs(G)', 'Memory(GB)', 'Latency(ms)']
        header = "| Model | " + " | ".join(resource_names) + " |\n"
        separator = "|" + "|".join([" --- " for _ in range(len(resource_names) + 1)]) + "|\n"
        
        table_str = header + separator
        
        # 数据行
        for model_name, data in self.results.items():
            resource_stats = data['resource_stats']
            row = f"| {model_name} |"
            
            resource_keys = ['params_M', 'flops_G', 'peak_memory_GB', 'inference_latency_ms']
            for key in resource_keys:
                if key in resource_stats:
                    value = resource_stats[key]
                    if key == 'params_M':
                        row += f" {value:.2f} |"
                    elif key == 'flops_G':
                        row += f" {value:.1f} |"
                    elif key == 'peak_memory_GB':
                        row += f" {value:.2f} |"
                    elif key == 'inference_latency_ms':
                        row += f" {value:.1f} |"
                else:
                    row += " N/A |"
            
            table_str += row + "\n"
        
        return table_str
    
    def perform_significance_test(self, baseline_model: str, metric: str = 'rel_l2') -> Dict[str, Dict[str, float]]:
        """执行显著性检验
        
        Args:
            baseline_model: 基线模型名称
            metric: 检验的指标
            
        Returns:
            test_results: 检验结果字典
        """
        if baseline_model not in self.results:
            raise ValueError(f"Baseline model {baseline_model} not found.")
        
        baseline_values = self.results[baseline_model]['metrics'].get(metric, [])
        if not baseline_values:
            raise ValueError(f"Metric {metric} not found for baseline model.")
        
        test_results = {}
        
        for model_name, data in self.results.items():
            if model_name == baseline_model:
                continue
            
            model_values = data['metrics'].get(metric, [])
            if not model_values:
                continue
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_values, model_values)
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(model_values)) / 2)
            cohens_d = (np.mean(model_values) - np.mean(baseline_values)) / pooled_std
            
            test_results[model_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
        
        return test_results
    
    def save_comparison_report(self, baseline_model: str = None):
        """保存完整对比报告
        
        Args:
            baseline_model: 基线模型名称
        """
        report = "# Model Comparison Report\n\n"
        
        # 性能对比表
        report += "## Performance Comparison\n\n"
        report += self.generate_performance_table()
        report += "\n"
        
        # 资源对比表
        report += "## Resource Comparison\n\n"
        report += self.generate_resource_table()
        report += "\n"
        
        # 显著性检验
        if baseline_model and baseline_model in self.results:
            report += f"## Significance Test (vs {baseline_model})\n\n"
            
            try:
                test_results = self.perform_significance_test(baseline_model, 'rel_l2')
                
                header = "| Model | t-statistic | p-value | Cohen's d | Significant |\n"
                separator = "| --- | --- | --- | --- | --- |\n"
                report += header + separator
                
                for model_name, results in test_results.items():
                    significant = "✓" if results['significant'] else "✗"
                    report += f"| {model_name} | {results['t_statistic']:.4f} | {results['p_value']:.4f} | {results['cohens_d']:.4f} | {significant} |\n"
                
                report += "\n"
            except Exception as e:
                report += f"Significance test failed: {e}\n\n"
        
        # 保存报告
        with open(self.output_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Comparison report saved to {self.output_dir / 'comparison_report.md'}")


def create_denormalize_fn(mu: torch.Tensor, sigma: torch.Tensor):
    """创建反归一化函数
    
    Args:
        mu: 均值张量
        sigma: 标准差张量
        
    Returns:
        denormalize: 反归一化函数
    """
    def denormalize(x_z: torch.Tensor) -> torch.Tensor:
        if mu.dim() == 1:
            mu_expanded = mu.view(1, -1, 1, 1)
            sigma_expanded = sigma.view(1, -1, 1, 1)
        else:
            mu_expanded = mu
            sigma_expanded = sigma
        
        return x_z * sigma_expanded + mu_expanded
    
    return denormalize


def create_model_from_config(config: DictConfig) -> nn.Module:
    """从配置创建模型
    
    Args:
        config: 配置对象
        
    Returns:
        model: 模型实例
    """
    model_name = config.model.name.lower()
    model_params = config.model.params
    
    if model_name == "swinunet":
        model = SwinUNet(
            in_channels=model_params.in_channels,
            out_channels=model_params.out_channels,
            img_size=model_params.img_size,
            **model_params.get('kwargs', {})
        )
    elif model_name == "hybrid":
        model = HybridModel(
            in_channels=model_params.in_channels,
            out_channels=model_params.out_channels,
            img_size=model_params.img_size,
            **model_params.get('kwargs', {})
        )
    elif model_name == "mlp":
        model = MLPModel(
            in_channels=model_params.in_channels,
            out_channels=model_params.out_channels,
            img_size=model_params.img_size,
            **model_params.get('kwargs', {})
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def classify_failure_type(metrics: Dict[str, float]) -> str:
    """分类失败类型
    
    Args:
        metrics: 指标字典
        
    Returns:
        failure_type: 失败类型
    """
    rel_l2 = metrics.get('rel_l2', 0)
    brmse = metrics.get('brmse', 0)
    crmse = metrics.get('crmse', 0)
    h_error = metrics.get('h_error', 0)
    frmse_high = metrics.get('frmse_high', 0)
    
    # 边界层溢出
    if brmse > 2 * crmse:
        return "boundary_overflow"
    
    # 高频振铃
    if frmse_high > 2 * metrics.get('frmse_low', 0):
        return "ringing"
    
    # 数据一致性差
    if h_error > 0.1:
        return "phase_drift"
    
    # 能量偏差
    if rel_l2 > 0.2:
        return "energy_bias"
    
    return "unknown"


def get_environment_info() -> Dict[str, Any]:
    """获取环境信息
    
    Returns:
        env_info: 环境信息字典
    """
    import platform
    import torch
    
    env_info = {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'platform': platform.platform(),
        'hostname': platform.node()
    }
    
    return env_info


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """主评估函数
    
    Args:
        cfg: 配置对象
    """
    # 设置随机种子
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        set_deterministic_mode(True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(cfg.get('output_dir', 'paper_package/metrics'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'eval.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation with config: {OmegaConf.to_yaml(cfg)}")
    
    # 创建数据模块
    data_module = PDEBenchDataModule(cfg.data)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # 获取归一化统计量
    norm_stats = data_module.get_norm_stats()
    mu = norm_stats['mean'].to(device)
    sigma = norm_stats['std'].to(device)
    denormalize_fn = create_denormalize_fn(mu, sigma)
    
    # 创建指标计算器和可视化器
    metrics_calculator = MetricsCalculator(
        image_size=(cfg.data.image_size, cfg.data.image_size),
        boundary_width=cfg.get('boundary_width', 16)
    )
    
    visualizer = Visualizer(output_dir / 'visualizations')
    comparator = ModelComparator(output_dir)
    
    # 处理多个检查点（支持多种子对比）
    checkpoint_paths = cfg.get('checkpoint_paths', [])
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    
    all_results = {}
    
    for checkpoint_path in checkpoint_paths:
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = load_checkpoint(checkpoint_path, device)
        config = OmegaConf.create(checkpoint['config'])
        
        # 创建模型
        model = create_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 获取模型名称
        model_name = f"{config.model.name}_s{config.training.seed}"
        
        # 评估模型
        case_metrics = []
        case_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
                # 数据移动到设备
                target = batch['target'].to(device)
                observation = batch['observation'].to(device)
                
                # 构建模型输入
                if 'baseline' in batch:
                    baseline = batch['baseline'].to(device)
                else:
                    baseline = observation
                
                if 'coords' in batch:
                    coords = batch['coords'].to(device)
                else:
                    coords = None
                
                if 'mask' in batch:
                    mask = batch['mask'].to(device)
                else:
                    mask = None
                
                # 输入打包
                model_input = baseline
                if coords is not None:
                    model_input = torch.cat([model_input, coords], dim=1)
                if mask is not None:
                    model_input = torch.cat([model_input, mask], dim=1)
                
                task_params = batch['task_params']
                
                # 前向传播
                pred = model(model_input)
                
                # 计算指标
                batch_metrics = metrics_calculator.compute_all_metrics(
                    pred, target, observation, task_params, denormalize_fn
                )
                
                # 转换为numpy并聚合
                batch_metrics_np = {}
                for key, value in batch_metrics.items():
                    if isinstance(value, torch.Tensor):
                        # 按通道平均，然后按batch平均
                        batch_metrics_np[key] = torch.mean(value).item()
                    else:
                        batch_metrics_np[key] = value
                
                case_metrics.append(batch_metrics_np)
                
                # 保存案例结果（用于可视化）
                if batch_idx < cfg.get('max_vis_cases', 10):
                    case_results.append({
                        'target': target.cpu().numpy(),
                        'pred': pred.cpu().numpy(),
                        'observation': observation.cpu().numpy(),
                        'metrics': batch_metrics_np,
                        'case_name': f"{model_name}_case_{batch_idx}"
                    })
        
        # 聚合指标
        aggregated_metrics = {}
        for key in case_metrics[0].keys():
            values = [case[key] for case in case_metrics]
            aggregated_metrics[key] = values
        
        # 添加到对比器
        resource_stats = checkpoint.get('resource_stats', {})
        comparator.add_model_results(model_name, aggregated_metrics, resource_stats, config)
        
        all_results[model_name] = {
            'metrics': aggregated_metrics,
            'cases': case_results,
            'resource_stats': resource_stats
        }
        
        logger.info(f"Completed evaluation for {model_name}")
        
        # 打印平均指标
        for key, values in aggregated_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{key}: {mean_val:.6f} ± {std_val:.6f}")
    
    # 生成可视化
    logger.info("Generating visualizations...")
    
    for model_name, results in all_results.items():
        cases = results['cases']
        
        for case in cases[:cfg.get('max_vis_cases', 3)]:  # 限制可视化案例数量
            target = case['target'][0]  # 取第一个batch
            pred = case['pred'][0]
            metrics = case['metrics']
            case_name = case['case_name']
            
            # 基础对比图
            visualizer.plot_comparison_heatmaps(target, pred, case_name)
            
            # 功率谱图
            visualizer.plot_power_spectrum(target, pred, case_name)
            
            # 边界分析图
            visualizer.plot_boundary_analysis(target, pred, case_name)
            
            # 失败案例分析（如果性能较差）
            if metrics.get('rel_l2', 0) > cfg.get('failure_threshold', 0.1):
                failure_type = classify_failure_type(metrics)
                visualizer.plot_failure_case_analysis(target, pred, metrics, case_name, failure_type)
    
    # 生成对比报告
    logger.info("Generating comparison report...")
    baseline_model = cfg.get('baseline_model', None)
    comparator.save_comparison_report(baseline_model)
    
    # 保存详细结果
    detailed_results = {}
    for model_name, results in all_results.items():
        # 计算统计量
        metrics_stats = {}
        for key, values in results['metrics'].items():
            metrics_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
        
        detailed_results[model_name] = {
            'metrics': metrics_stats,
            'resource_stats': results['resource_stats'],
            'environment_info': get_environment_info()
        }
    
    # 保存到JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # 保存到JSONL（每个案例一行）
    with open(output_dir / 'case_metrics.jsonl', 'w') as f:
        for model_name, results in all_results.items():
            for i, case_metric in enumerate(case_metrics):
                record = {
                    'model': model_name,
                    'case_id': i,
                    **case_metric
                }
                f.write(json.dumps(record) + '\n')
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Visualizations saved to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()