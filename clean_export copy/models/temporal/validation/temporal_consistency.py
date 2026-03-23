"""
时序一致性验证机制
用于验证时序预测模型的物理一致性和数值稳定性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from pathlib import Path


class TemporalConsistencyValidator:
    """时序一致性验证器
    
    验证时序预测模型的以下特性：
    1. 物理一致性（PDE残差、能量守恒等）
    2. 数值稳定性（长期预测稳定性）
    3. 时间因果性（无未来信息泄露）
    4. 边界条件一致性
    5. 多尺度一致性
    """
    
    def __init__(
        self,
        pde_type: str = 'heat',
        tolerance: float = 1e-3,
        max_prediction_horizon: int = 100,
        enable_visualization: bool = True,
        save_dir: Optional[str] = None
    ):
        self.pde_type = pde_type
        self.tolerance = tolerance
        self.max_prediction_horizon = max_prediction_horizon
        self.enable_visualization = enable_visualization
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证指标
        self.validation_metrics = {
            'physics_residual': [],
            'energy_conservation': [],
            'causality_violation': [],
            'boundary_error': [],
            'long_term_stability': [],
            'multi_scale_consistency': []
        }
        
    def validate_physics_residual(self, predictions: torch.Tensor, 
                                ground_truth: Optional[torch.Tensor] = None,
                                dt: float = 1.0, dx: float = 1.0) -> Dict[str, float]:
        """验证PDE残差"""
        results = {}
        
        if self.pde_type == 'heat':
            residual = self._compute_heat_residual(predictions, dt, dx)
        elif self.pde_type == 'wave':
            residual = self._compute_wave_residual(predictions, dt, dx)
        elif self.pde_type == 'navier_stokes':
            residual = self._compute_ns_residual(predictions, dt, dx)
        else:
            residual = torch.zeros_like(predictions)
        
        # 计算残差范数
        residual_l2 = torch.norm(residual, p=2, dim=list(range(2, residual.dim())))
        residual_linf = torch.norm(residual, p=float('inf'), dim=list(range(2, residual.dim())))
        
        results['residual_l2_mean'] = residual_l2.mean().item()
        results['residual_linf_mean'] = residual_linf.mean().item()
        results['residual_l2_std'] = residual_l2.std().item()
        results['max_residual'] = torch.max(torch.abs(residual)).item()
        
        # 相对残差（如果有真值）
        if ground_truth is not None:
            relative_residual = residual / (torch.abs(ground_truth) + 1e-8)
            results['relative_residual_mean'] = torch.mean(torch.abs(relative_residual)).item()
        
        return results
    
    def _compute_heat_residual(self, u: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """计算热方程残差: u_t = α*u_xx"""
        if u.size(1) < 3:
            return torch.zeros_like(u)
        
        # 时间导数
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        
        # 空间二阶导数
        if u.dim() >= 3:
            u_xx = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)
            # 填充边界
            u_xx = F.pad(u_xx, (1, 1, 1, 1), mode='replicate')
        else:
            u_xx = torch.zeros_like(u_t)
        
        # 热方程残差（α=0.1）
        alpha = 0.1
        residual = u_t - alpha * u_xx
        
        return residual
    
    def _compute_wave_residual(self, u: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """计算波动方程残差: u_tt = c²*u_xx - γ*u_t"""
        if u.size(1) < 3:
            return torch.zeros_like(u)
        
        # 时间二阶导数
        u_tt = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dt ** 2)
        
        # 时间一阶导数
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        
        # 空间二阶导数
        if u.dim() >= 3:
            u_xx = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)
            u_xx = F.pad(u_xx, (1, 1, 1, 1), mode='replicate')
        else:
            u_xx = torch.zeros_like(u_tt)
        
        # 波动方程参数
        c_squared = 1.0  # 波速平方
        gamma = 0.01  # 阻尼系数
        
        residual = u_tt - c_squared * u_xx + gamma * u_t
        
        return residual
    
    def _compute_ns_residual(self, u: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """计算Navier-Stokes方程残差（简化版本）"""
        # 这里实现简化的NS方程残差
        if u.size(1) < 3:
            return torch.zeros_like(u)
        
        viscosity = 0.01
        
        # 对流项（简化）
        convection = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        
        # 扩散项（简化）
        if u.dim() >= 3:
            diffusion = viscosity * (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)
            diffusion = F.pad(diffusion, (1, 1, 1, 1), mode='replicate')
        else:
            diffusion = torch.zeros_like(convection)
        
        residual = convection - diffusion
        
        return residual
    
    def validate_energy_conservation(self, predictions: torch.Tensor) -> Dict[str, float]:
        """验证能量守恒"""
        results = {}
        
        # 计算总能量（L2范数）
        energy = torch.sum(predictions ** 2, dim=list(range(2, predictions.dim())))
        
        # 能量变化
        if predictions.size(1) >= 2:
            energy_change = energy[:, 1:] - energy[:, :-1]
            
            results['energy_change_mean'] = energy_change.mean().item()
            results['energy_change_std'] = energy_change.std().item()
            results['max_energy_change'] = torch.max(torch.abs(energy_change)).item()
            results['energy_drift'] = torch.abs(energy_change.sum(dim=1)).mean().item()
            
            # 能量守恒违反程度
            energy_violation = torch.abs(energy_change) / (energy[:, :-1] + 1e-8)
            results['relative_energy_violation'] = energy_violation.mean().item()
        else:
            results['energy_change_mean'] = 0.0
            results['energy_change_std'] = 0.0
            results['max_energy_change'] = 0.0
            results['energy_drift'] = 0.0
            results['relative_energy_violation'] = 0.0
        
        # 总能量统计
        results['total_energy_mean'] = energy.mean().item()
        results['total_energy_std'] = energy.std().item()
        
        return results
    
    def validate_causality(self, attention_weights: Optional[torch.Tensor] = None,
                          predictions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """验证时间因果性"""
        results = {}
        
        if attention_weights is not None:
            # 检查注意力权重是否违反因果性
            B, T, _ = attention_weights.shape[:3]
            
            causality_violations = []
            for t in range(T):
                # 检查未来时间步对当前时间步的影响
                if t + 1 < T:
                    future_influence = attention_weights[:, t, t+1:].sum(dim=-1)
                    causality_violations.append(future_influence)
            
            if causality_violations:
                all_violations = torch.cat(causality_violations)
                results['causality_violation_mean'] = all_violations.mean().item()
                results['causality_violation_max'] = all_violations.max().item()
                results['causality_violation_rate'] = (all_violations > self.tolerance).float().mean().item()
            else:
                results['causality_violation_mean'] = 0.0
                results['causality_violation_max'] = 0.0
                results['causality_violation_rate'] = 0.0
        
        # 通过预测结果检查因果性（检查时间相关性）
        if predictions is not None and predictions.size(1) >= 3:
            # 计算时间相关性矩阵
            B, T = predictions.shape[:2]
            predictions_flat = predictions.reshape(B, T, -1)
            
            correlations = []
            for b in range(B):
                # 计算时间相关性
                corr_matrix = torch.corrcoef(predictions_flat[b].T)
                if corr_matrix.size(0) > 1:
                    # 检查未来与现在的相关性
                    for t in range(T-1):
                        future_corr = torch.abs(corr_matrix[t, t+1:]).mean()
                        correlations.append(future_corr)
            
            if correlations:
                correlations = torch.stack(correlations)
                results['temporal_correlation_mean'] = correlations.mean().item()
                results['temporal_correlation_max'] = correlations.max().item()
            else:
                results['temporal_correlation_mean'] = 0.0
                results['temporal_correlation_max'] = 0.0
        
        return results
    
    def validate_boundary_conditions(self, predictions: torch.Tensor,
                                   expected_bc: Optional[Dict] = None) -> Dict[str, float]:
        """验证边界条件"""
        results = {}
        
        if predictions.dim() >= 3:
            # 提取边界值
            left_boundary = predictions[..., 0]
            right_boundary = predictions[..., -1]
            
            if predictions.dim() >= 4:
                top_boundary = predictions[..., 0, :]
                bottom_boundary = predictions[..., -1, :]
            
            # 边界变化统计
            left_variation = torch.std(left_boundary, dim=list(range(2, left_boundary.dim())))
            right_variation = torch.std(right_boundary, dim=list(range(2, right_boundary.dim())))
            
            results['left_boundary_variation_mean'] = left_variation.mean().item()
            results['right_boundary_variation_mean'] = right_variation.mean().item()
            results['boundary_variation_max'] = max(left_variation.max().item(), 
                                                    right_variation.max().item())
            
            # 边界一致性检查
            boundary_diff = torch.abs(left_boundary - right_boundary)
            results['boundary_consistency_error'] = boundary_diff.mean().item()
            
            # 与期望边界条件的比较
            if expected_bc is not None:
                if 'left' in expected_bc:
                    left_error = torch.abs(left_boundary - expected_bc['left'])
                    results['left_boundary_error'] = left_error.mean().item()
                
                if 'right' in expected_bc:
                    right_error = torch.abs(right_boundary - expected_bc['right'])
                    results['right_boundary_error'] = right_error.mean().item()
        
        return results
    
    def validate_long_term_stability(self, predictions: torch.Tensor) -> Dict[str, float]:
        """验证长期稳定性"""
        results = {}
        
        B, T = predictions.shape[:2]
        
        if T >= 10:  # 需要足够长的时间序列
            # 计算时间序列的统计特性
            mean_evolution = predictions.mean(dim=list(range(2, predictions.dim())))
            std_evolution = predictions.std(dim=list(range(2, predictions.dim())))
            
            # 检查均值漂移
            mean_drift = torch.abs(mean_evolution[:, -1] - mean_evolution[:, 0])
            results['mean_drift'] = mean_drift.mean().item()
            results['max_mean_drift'] = mean_drift.max().item()
            
            # 检查方差增长
            variance_growth = std_evolution[:, -1] / (std_evolution[:, 0] + 1e-8)
            results['variance_growth_mean'] = variance_growth.mean().item()
            results['variance_growth_max'] = variance_growth.max().item()
            
            # 检查指数增长趋势
            if T >= 20:
                # 拟合指数增长
                time_steps = torch.arange(T, device=predictions.device).float()
                
                growth_rates = []
                for b in range(B):
                    signal_energy = (predictions[b] ** 2).mean(dim=list(range(1, predictions[b].dim())))
                    
                    # 简单的指数拟合
                    log_energy = torch.log(signal_energy + 1e-8)
                    
                    # 线性回归拟合
                    time_mean = time_steps.mean()
                    log_mean = log_energy.mean()
                    
                    numerator = ((time_steps - time_mean) * (log_energy - log_mean)).sum()
                    denominator = ((time_steps - time_mean) ** 2).sum()
                    
                    if denominator > 0:
                        growth_rate = numerator / denominator
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    growth_rates = torch.stack(growth_rates)
                    results['exponential_growth_rate_mean'] = growth_rates.mean().item()
                    results['exponential_growth_rate_max'] = growth_rates.max().item()
                    
                    # 检查是否有过快的增长
                    unstable_rate = (growth_rates > 0.01).float().mean().item()
                    results['instability_rate'] = unstable_rate
        
        return results
    
    def validate_multi_scale_consistency(self, predictions: torch.Tensor) -> Dict[str, float]:
        """验证多尺度一致性"""
        results = {}
        
        B, T = predictions.shape[:2]
        
        if T >= 8:  # 需要足够长的时间序列进行多尺度分析
            scales = [1, 2, 4]
            scale_features = []
            
            for scale in scales:
                if T % scale == 0:
                    # 下采样
                    pooled = F.avg_pool1d(
                        predictions.reshape(B, T, -1).transpose(1, 2),
                        kernel_size=scale, stride=scale
                    ).transpose(1, 2)
                    scale_features.append(pooled)
            
            if len(scale_features) >= 2:
                # 计算不同尺度间的一致性
                consistencies = []
                
                for i in range(len(scale_features) - 1):
                    fine_scale = scale_features[i]
                    coarse_scale = scale_features[i + 1]
                    
                    # 上采样粗尺度以匹配细尺度
                    upsampled_coarse = F.interpolate(
                        coarse_scale.transpose(1, 2), 
                        size=fine_scale.size(1), 
                        mode='linear', align_corners=False
                    ).transpose(1, 2)
                    
                    # 计算一致性误差
                    consistency_error = torch.abs(fine_scale - upsampled_coarse)
                    consistencies.append(consistency_error.mean().item())
                
                if consistencies:
                    results['multi_scale_consistency_mean'] = np.mean(consistencies)
                    results['multi_scale_consistency_max'] = max(consistencies)
        
        return results
    
    def comprehensive_validation(self, predictions: torch.Tensor,
                               ground_truth: Optional[torch.Tensor] = None,
                               attention_weights: Optional[torch.Tensor] = None,
                               expected_bc: Optional[Dict] = None,
                               dt: float = 1.0, dx: float = 1.0) -> Dict[str, Union[float, bool]]:
        """综合验证"""
        all_results = {}
        
        # 1. PDE残差验证
        physics_results = self.validate_physics_residual(predictions, ground_truth, dt, dx)
        all_results.update(physics_results)
        
        # 2. 能量守恒验证
        energy_results = self.validate_energy_conservation(predictions)
        all_results.update(energy_results)
        
        # 3. 因果性验证
        causality_results = self.validate_causality(attention_weights, predictions)
        all_results.update(causality_results)
        
        # 4. 边界条件验证
        boundary_results = self.validate_boundary_conditions(predictions, expected_bc)
        all_results.update(boundary_results)
        
        # 5. 长期稳定性验证
        stability_results = self.validate_long_term_stability(predictions)
        all_results.update(stability_results)
        
        # 6. 多尺度一致性验证
        scale_results = self.validate_multi_scale_consistency(predictions)
        all_results.update(scale_results)
        
        # 总体评估
        all_results['overall_physics_valid'] = self._assess_overall_validity(all_results)
        all_results['validation_pass_rate'] = self._compute_pass_rate(all_results)
        
        # 存储验证结果
        for key, value in all_results.items():
            if isinstance(value, (int, float)):
                if key not in self.validation_metrics:
                    self.validation_metrics[key] = []
                self.validation_metrics[key].append(value)
        
        return all_results
    
    def _assess_overall_validity(self, results: Dict[str, float]) -> bool:
        """评估整体物理有效性"""
        key_metrics = [
            'residual_l2_mean',
            'energy_change_mean', 
            'causality_violation_mean',
            'boundary_consistency_error'
        ]
        
        thresholds = {
            'residual_l2_mean': 1e-2,
            'energy_change_mean': 1e-2,
            'causality_violation_mean': 1e-3,
            'boundary_consistency_error': 1e-2
        }
        
        valid = True
        for metric in key_metrics:
            if metric in results and results[metric] > thresholds.get(metric, self.tolerance):
                valid = False
                break
        
        return valid
    
    def _compute_pass_rate(self, results: Dict[str, float]) -> float:
        """计算通过率"""
        thresholds = {
            'residual_l2_mean': 1e-2,
            'energy_change_mean': 1e-2,
            'causality_violation_mean': 1e-3,
            'boundary_consistency_error': 1e-2,
            'mean_drift': 1e-1,
            'variance_growth_max': 2.0,
            'multi_scale_consistency_max': 1e-1
        }
        
        passed = 0
        total = 0
        
        for metric, value in results.items():
            if metric in thresholds:
                total += 1
                if value <= thresholds[metric]:
                    passed += 1
        
        return passed / total if total > 0 else 1.0
    
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        if not self.validation_metrics:
            return "No validation data available."
        
        report = []
        report.append("=" * 60)
        report.append("TEMPORAL CONSISTENCY VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"PDE Type: {self.pde_type}")
        report.append(f"Tolerance: {self.tolerance}")
        report.append(f"Validation Samples: {len(next(iter(self.validation_metrics.values())))}")
        report.append("")
        
        # 关键指标统计
        key_metrics = [
            'residual_l2_mean',
            'energy_change_mean',
            'causality_violation_mean',
            'boundary_consistency_error',
            'mean_drift',
            'instability_rate'
        ]
        
        report.append("KEY VALIDATION METRICS:")
        report.append("-" * 40)
        
        for metric in key_metrics:
            if metric in self.validation_metrics and self.validation_metrics[metric]:
                values = self.validation_metrics[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                max_val = np.max(values)
                
                report.append(f"{metric:25s}: {mean_val:.6f} ± {std_val:.6f} (max: {max_val:.6f})")
        
        report.append("")
        
        # 通过率统计
        if 'validation_pass_rate' in self.validation_metrics:
            pass_rates = self.validation_metrics['validation_pass_rate']
            report.append(f"AVERAGE PASS RATE: {np.mean(pass_rates):.2%}")
            report.append(f"MIN PASS RATE: {np.min(pass_rates):.2%}")
        
        report.append("")
        report.append("VALIDATION SUMMARY:")
        report.append("-" * 40)
        
        # 整体评估
        if 'overall_physics_valid' in self.validation_metrics:
            valid_rates = np.mean(self.validation_metrics['overall_physics_valid'])
            report.append(f"Overall Physics Validity Rate: {valid_rates:.2%}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_validation_results(self, save_path: Optional[str] = None):
        """绘制验证结果图表"""
        if not self.enable_visualization or not self.validation_metrics:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # 选择关键指标进行可视化
            plot_metrics = [
                'residual_l2_mean',
                'energy_change_mean', 
                'causality_violation_mean',
                'boundary_consistency_error',
                'mean_drift',
                'validation_pass_rate'
            ]
            
            for i, metric in enumerate(plot_metrics):
                if metric in self.validation_metrics and self.validation_metrics[metric]:
                    values = self.validation_metrics[metric]
                    axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting.")


class TemporalStabilityAnalyzer:
    """时序稳定性分析器"""
    
    def __init__(self, analysis_window: int = 10):
        self.analysis_window = analysis_window
        self.stability_metrics = {}
    
    def analyze_lyapunov_exponent(self, predictions: torch.Tensor) -> float:
        """分析Lyapunov指数（混沌特性）"""
        # 简化的Lyapunov指数估计
        B, T = predictions.shape[:2]
        
        if T < self.analysis_window * 2:
            return 0.0
        
        lyapunov_exponents = []
        
        for b in range(B):
            # 计算相邻轨迹的分离率
            trajectory = predictions[b].reshape(T, -1)
            
            separations = []
            for t in range(T - self.analysis_window):
                current = trajectory[t]
                future = trajectory[t + self.analysis_window]
                
                # 计算分离距离
                separation = torch.norm(future - current)
                separations.append(separation.item())
            
            if separations:
                # 估计Lyapunov指数
                separations = np.array(separations)
                time_points = np.arange(len(separations))
                
                # 线性拟合log(separation) vs time
                log_separations = np.log(separations + 1e-10)
                
                if len(log_separations) > 1:
                    # 简单线性回归
                    slope = np.polyfit(time_points, log_separations, 1)[0]
                    lyapunov_exponents.append(slope)
        
        if lyapunov_exponents:
            return np.mean(lyapunov_exponents)
        else:
            return 0.0
    
    def analyze_frequency_content(self, predictions: torch.Tensor) -> Dict[str, float]:
        """分析频率内容变化"""
        results = {}
        
        B, T = predictions.shape[:2]
        
        if T >= self.analysis_window:
            frequency_evolution = []
            
            for b in range(B):
                # 对每个时间序列进行FFT
                trajectory = predictions[b].reshape(T, -1)
                
                # 计算每个时间点的频率内容
                freqs = []
                for t in range(0, T - self.analysis_window, self.analysis_window // 2):
                    window = trajectory[t:t + self.analysis_window]
                    
                    # FFT
                    fft_result = torch.fft.rfft(window, dim=0)
                    magnitude = torch.abs(fft_result).mean(dim=1)
                    
                    # 计算主要频率
                    dominant_freq = torch.argmax(magnitude[1:]) + 1  # 排除DC分量
                    freqs.append(dominant_freq.item())
                
                if freqs:
                    frequency_evolution.extend(freqs)
            
            if frequency_evolution:
                frequency_evolution = np.array(frequency_evolution)
                results['dominant_frequency_mean'] = np.mean(frequency_evolution)
                results['dominant_frequency_std'] = np.std(frequency_evolution)
                results['frequency_drift'] = np.max(frequency_evolution) - np.min(frequency_evolution)
        
        return results


# 辅助函数
import torch.nn.functional as F