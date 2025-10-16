"""评测指标模块

实现PDEBench稀疏观测重建系统的所有评测指标：
- Rel-L2: 相对L2误差
- MAE: 平均绝对误差  
- PSNR: 峰值信噪比
- SSIM: 结构相似性指数
- fRMSE: 频域RMSE (low/mid/high)
- bRMSE: 边界RMSE
- cRMSE: 中心RMSE
- ||H(ŷ)−y||: 数据一致性误差

按照开发手册要求：
- 每通道先算，后等权平均
- 支持统计分析（均值±标准差）
- 支持显著性检验
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as ssim
import scipy.stats as stats

from ops.degradation import apply_degradation_operator


class MetricsCalculator:
    """指标计算器
    
    提供完整的评测指标计算功能
    支持批量计算和统计分析
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), 
                 boundary_width: int = 16,
                 freq_bands: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        Args:
            image_size: 图像尺寸 (H, W)
            boundary_width: 边界宽度（像素）
            freq_bands: 频段定义 {'band_name': (low_freq, high_freq)}
        """
        self.image_size = image_size
        self.boundary_width = boundary_width
        
        # 默认频段设置
        if freq_bands is None:
            max_freq = min(image_size) // 2
            self.freq_bands = {
                'low': (0, max_freq // 4),
                'mid': (max_freq // 4, max_freq // 2), 
                'high': (max_freq // 2, max_freq)
            }
        else:
            self.freq_bands = freq_bands
        
        # 预计算掩码
        self._precompute_masks()
    
    def update_image_size(self, new_size: Tuple[int, int]):
        """更新图像尺寸并重新计算掩码"""
        if new_size != self.image_size:
            self.image_size = new_size
            self._precompute_masks()
    
    def _precompute_masks(self) -> None:
        """预计算各种掩码"""
        H, W = self.image_size
        
        # 边界掩码
        self.boundary_mask = torch.zeros(H, W, dtype=torch.bool)
        bw = self.boundary_width
        self.boundary_mask[:bw, :] = True  # 上边界
        self.boundary_mask[-bw:, :] = True  # 下边界
        self.boundary_mask[:, :bw] = True  # 左边界
        self.boundary_mask[:, -bw:] = True  # 右边界
        
        # 中心掩码
        self.center_mask = ~self.boundary_mask
        
        # 频域掩码
        self.freq_masks = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = self._create_freq_mask(H, W, low_freq, high_freq)
            self.freq_masks[band_name] = mask
    
    def _create_freq_mask(self, H: int, W: int, low_freq: int, high_freq: int) -> torch.Tensor:
        """创建频域掩码"""
        # 创建频率网格
        ky = torch.fft.fftfreq(H, d=1.0).abs()
        kx = torch.fft.fftfreq(W, d=1.0).abs()
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
        
        # 径向频率
        k_radial = torch.sqrt(kx_grid**2 + ky_grid**2)
        
        # 频率范围掩码
        mask = (k_radial >= low_freq / max(H, W)) & (k_radial < high_freq / max(H, W))
        return mask
    
    def compute_rel_l2(self, pred: torch.Tensor, target: torch.Tensor, 
                       eps: float = 1e-8) -> torch.Tensor:
        """计算相对L2误差
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            eps: 数值稳定性常数
            
        Returns:
            rel_l2: 相对L2误差 [B, C] 或标量
        """
        # 按通道计算
        diff = pred - target
        mse = torch.mean(diff**2, dim=(-2, -1))  # [B, C]
        target_norm = torch.mean(target**2, dim=(-2, -1)) + eps  # [B, C]
        
        rel_l2 = torch.sqrt(mse / target_norm)
        
        return rel_l2
    
    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算平均绝对误差"""
        diff = torch.abs(pred - target)
        mae = torch.mean(diff, dim=(-2, -1))  # [B, C]
        return mae
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor, 
                     max_val: Optional[float] = None) -> torch.Tensor:
        """计算峰值信噪比
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 真实值 [B, C, H, W]  
            max_val: 最大值，如果为None则自动计算
            
        Returns:
            psnr: PSNR值 [B, C]
        """
        if max_val is None:
            max_val = torch.max(target)
        
        mse = torch.mean((pred - target)**2, dim=(-2, -1))  # [B, C]
        psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
        
        return psnr
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算结构相似性指数
        
        使用scikit-image的SSIM实现
        """
        B, C, H, W = pred.shape
        ssim_values = torch.zeros(B, C)
        
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        for b in range(B):
            for c in range(C):
                # 计算单通道SSIM
                ssim_val = ssim(
                    target_np[b, c], 
                    pred_np[b, c],
                    data_range=target_np[b, c].max() - target_np[b, c].min()
                )
                ssim_values[b, c] = ssim_val
        
        return ssim_values
    
    def compute_freq_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算频域RMSE
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            
        Returns:
            freq_rmse: 各频段RMSE字典
        """
        # 检查尺寸是否匹配，如果不匹配则调整
        if pred.shape != target.shape:
            # 将pred调整到target的尺寸
            pred = torch.nn.functional.interpolate(
                pred, size=target.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # 更新图像尺寸以匹配当前数据
        current_size = target.shape[-2:]
        if current_size != self.image_size:
            self.update_image_size(current_size)
        
        B, C, H, W = pred.shape
        freq_rmse = {}
        
        for band_name, mask in self.freq_masks.items():
            mask = mask.to(pred.device)
            
            # FFT变换
            pred_fft = torch.fft.fft2(pred)
            target_fft = torch.fft.fft2(target)
            
            # 应用频域掩码
            pred_fft_masked = pred_fft * mask.unsqueeze(0).unsqueeze(0)
            target_fft_masked = target_fft * mask.unsqueeze(0).unsqueeze(0)
            
            # 逆变换回空间域
            pred_filtered = torch.fft.ifft2(pred_fft_masked).real
            target_filtered = torch.fft.ifft2(target_fft_masked).real
            
            # 计算RMSE
            mse = torch.mean((pred_filtered - target_filtered)**2, dim=(-2, -1))
            rmse = torch.sqrt(mse)
            
            freq_rmse[band_name] = rmse
        
        return freq_rmse
    
    def compute_boundary_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边界RMSE"""
        mask = self.boundary_mask.to(pred.device)
        
        # 应用边界掩码
        pred_boundary = pred * mask.unsqueeze(0).unsqueeze(0)
        target_boundary = target * mask.unsqueeze(0).unsqueeze(0)
        
        # 计算边界区域的MSE
        diff_sq = (pred_boundary - target_boundary)**2
        mse = torch.sum(diff_sq, dim=(-2, -1)) / torch.sum(mask)  # [B, C]
        rmse = torch.sqrt(mse)
        
        return rmse
    
    def compute_center_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算中心RMSE"""
        mask = self.center_mask.to(pred.device)
        
        # 应用中心掩码
        pred_center = pred * mask.unsqueeze(0).unsqueeze(0)
        target_center = target * mask.unsqueeze(0).unsqueeze(0)
        
        # 计算中心区域的MSE
        diff_sq = (pred_center - target_center)**2
        mse = torch.sum(diff_sq, dim=(-2, -1)) / torch.sum(mask)  # [B, C]
        rmse = torch.sqrt(mse)
        
        return rmse
    
    def compute_data_consistency_error(self, pred: torch.Tensor, obs_data: Dict,
                                     norm_stats: Optional[Dict] = None) -> torch.Tensor:
        """计算数据一致性误差 ||H(ŷ)−y||
        
        Args:
            pred: 预测值（z-score域） [B, C, H, W]
            obs_data: 观测数据字典
            norm_stats: 归一化统计量
            
        Returns:
            dc_error: 数据一致性误差 [B, C]
        """
        # 反归一化到原值域
        if norm_stats is not None:
            mean = torch.tensor(norm_stats['mean']).to(pred.device).view(1, -1, 1, 1)
            std = torch.tensor(norm_stats['std']).to(pred.device).view(1, -1, 1, 1)
            pred_orig = pred * std + mean
        else:
            pred_orig = pred
        
        # 应用观测算子H
        pred_obs = apply_degradation_operator(pred_orig, obs_data)
        
        # 计算与观测数据的误差
        target_obs = obs_data['baseline']
        mse = torch.mean((pred_obs - target_obs)**2, dim=(-2, -1))  # [B, C]
        dc_error = torch.sqrt(mse)
        
        return dc_error
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                           obs_data: Optional[Dict] = None,
                           norm_stats: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """计算所有指标
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 真实值 [B, C, H, W]
            obs_data: 观测数据（用于数据一致性）
            norm_stats: 归一化统计量
            
        Returns:
            metrics: 所有指标的字典
        """
        metrics = {}
        
        # 基础指标
        metrics['rel_l2'] = self.compute_rel_l2(pred, target)
        metrics['mae'] = self.compute_mae(pred, target)
        metrics['psnr'] = self.compute_psnr(pred, target)
        metrics['ssim'] = self.compute_ssim(pred, target)
        
        # 频域指标
        freq_rmse = self.compute_freq_rmse(pred, target)
        for band_name, rmse in freq_rmse.items():
            metrics[f'frmse_{band_name}'] = rmse
        
        # 空间域指标
        metrics['brmse'] = self.compute_boundary_rmse(pred, target)
        metrics['crmse'] = self.compute_center_rmse(pred, target)
        
        # 数据一致性指标
        if obs_data is not None:
            metrics['dc_error'] = self.compute_data_consistency_error(pred, obs_data, norm_stats)
        
        return metrics


class StatisticalAnalyzer:
    """统计分析器
    
    提供多种子实验的统计分析功能
    包括均值、标准差、显著性检验等
    """
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: Dict[str, Union[float, torch.Tensor]]):
        """添加单次实验结果"""
        self.results.append(result)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算统计信息"""
        return self.aggregate_metrics(self.results)
    
    def generate_report(self, baseline_name: str = None) -> str:
        """生成统计报告"""
        if not self.results:
            return "No results to report."
        
        stats = self.compute_statistics()
        
        lines = []
        lines.append("Statistical Analysis Report")
        lines.append("=" * 50)
        
        for metric_name, metric_stats in stats.items():
            lines.append(f"\n{metric_name}:")
            lines.append(f"  Mean: {metric_stats['mean']:.6f}")
            lines.append(f"  Std:  {metric_stats['std']:.6f}")
            lines.append(f"  Min:  {metric_stats['min']:.6f}")
            lines.append(f"  Max:  {metric_stats['max']:.6f}")
            lines.append(f"  Count: {metric_stats['count']}")
        
        return "\n".join(lines)
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        """聚合多次实验的指标
        
        Args:
            metrics_list: 多次实验的指标列表
            
        Returns:
            aggregated: {'metric_name': {'mean': float, 'std': float, 'min': float, 'max': float}}
        """
        if not metrics_list:
            return {}
        
        # 获取所有指标名称
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        aggregated = {}
        
        for metric_name in metric_names:
            # 收集该指标的所有值
            values = []
            for metrics in metrics_list:
                if metric_name in metrics:
                    # 取通道平均值
                    metric_value = metrics[metric_name]
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.dim() > 0:
                            value = torch.mean(metric_value).item()
                        else:
                            value = metric_value.item()
                    elif isinstance(metric_value, (int, float)):
                        value = float(metric_value)
                    else:
                        continue  # 跳过无法处理的类型
                    
                    # 跳过NaN值
                    if not np.isnan(value):
                        values.append(value)
            
            if values:
                values = np.array(values)
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1) if len(values) > 1 else 0),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return aggregated
    
    def compute_significance_test(self, baseline_metrics: List[Dict[str, torch.Tensor]],
                                 method_metrics: List[Dict[str, torch.Tensor]],
                                 metric_name: str = 'rel_l2',
                                 alpha: float = 0.05) -> Dict[str, float]:
        """计算显著性检验
        
        Args:
            baseline_metrics: 基线方法的指标列表
            method_metrics: 对比方法的指标列表
            metric_name: 要检验的指标名称
            alpha: 显著性水平
            
        Returns:
            test_result: {'t_stat': float, 'p_value': float, 'cohen_d': float, 'significant': bool}
        """
        # 提取指标值
        baseline_values = []
        for metrics in baseline_metrics:
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                if isinstance(metric_value, torch.Tensor):
                    value = torch.mean(metric_value).item()
                elif isinstance(metric_value, (int, float)):
                    value = float(metric_value)
                else:
                    continue
                baseline_values.append(value)
        
        method_values = []
        for metrics in method_metrics:
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                if isinstance(metric_value, torch.Tensor):
                    value = torch.mean(metric_value).item()
                elif isinstance(metric_value, (int, float)):
                    value = float(metric_value)
                else:
                    continue
                method_values.append(value)
        
        if len(baseline_values) == 0 or len(method_values) == 0:
            return {'error': 'Insufficient data for significance test'}
        
        # 检查是否有足够的样本进行统计检验
        if len(baseline_values) < 2 or len(method_values) < 2:
            return {
                'error': 'Insufficient samples for significance test (need at least 2 samples per group)',
                'baseline_count': len(baseline_values),
                'method_count': len(method_values)
            }
        
        baseline_values = np.array(baseline_values)
        method_values = np.array(method_values)
        
        # 检查方差是否为零（所有值相同）
        baseline_var = np.var(baseline_values, ddof=1)
        method_var = np.var(method_values, ddof=1)
        
        if baseline_var == 0 and method_var == 0:
            # 两组方差都为0，直接比较均值
            mean_diff = np.mean(method_values) - np.mean(baseline_values)
            return {
                't_stat': float('inf') if mean_diff != 0 else 0.0,
                'p_value': 0.0 if mean_diff != 0 else 1.0,
                'effect_size': float('inf') if mean_diff != 0 else 0.0,
                'is_significant': mean_diff != 0,
                'alpha': alpha,
                'note': 'Zero variance in both groups'
            }
        
        try:
            # Paired t-test if same number of samples
            if len(baseline_values) == len(method_values):
                t_stat, p_value = stats.ttest_rel(baseline_values, method_values)
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(baseline_values, method_values)
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_var +
                                 (len(method_values) - 1) * method_var) /
                                (len(baseline_values) + len(method_values) - 2))
            
            if pooled_std == 0:
                cohen_d = 0.0
            else:
                cohen_d = (np.mean(method_values) - np.mean(baseline_values)) / pooled_std
            
            return {
                't_stat': float(t_stat) if not np.isnan(t_stat) else 0.0,
                'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                'effect_size': float(cohen_d) if not np.isnan(cohen_d) else 0.0,
                'is_significant': bool(p_value < alpha) if not np.isnan(p_value) else False,
                'alpha': alpha
            }
            
        except Exception as e:
            return {
                'error': f'Statistical test failed: {str(e)}',
                'baseline_values': baseline_values.tolist(),
                'method_values': method_values.tolist()
            }
    
    def generate_summary_table(self, results: Dict[str, Dict[str, Dict[str, float]]],
                              metric_names: List[str] = None) -> str:
        """生成汇总表格
        
        Args:
            results: {'method_name': {'metric_name': {'mean': float, 'std': float, ...}}}
            metric_names: 要包含的指标名称列表
            
        Returns:
            table_str: 格式化的表格字符串
        """
        if metric_names is None:
            # 获取所有指标名称
            all_metrics = set()
            for method_results in results.values():
                all_metrics.update(method_results.keys())
            metric_names = sorted(list(all_metrics))
        
        # 构建表格
        lines = []
        
        # 表头
        header = "Method"
        for metric_name in metric_names:
            header += f"\t{metric_name}"
        lines.append(header)
        
        # 数据行
        for method_name, method_results in results.items():
            row = method_name
            for metric_name in metric_names:
                if metric_name in method_results:
                    stats = method_results[metric_name]
                    mean = stats['mean']
                    std = stats['std']
                    row += f"\t{mean:.4f}±{std:.4f}"
                else:
                    row += "\t-"
            lines.append(row)
        
        return "\n".join(lines)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                       obs_data: Optional[Dict] = None,
                       norm_stats: Optional[Dict] = None,
                       image_size: Tuple[int, int] = (256, 256)) -> Dict[str, torch.Tensor]:
    """便捷函数：计算所有指标
    
    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        obs_data: 观测数据（用于数据一致性）
        norm_stats: 归一化统计量
        image_size: 图像尺寸
        
    Returns:
        metrics: 所有指标的字典
    """
    calculator = MetricsCalculator(image_size=image_size)
    return calculator.compute_all_metrics(pred, target, obs_data, norm_stats)


def aggregate_multi_seed_results(results_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    """便捷函数：聚合多种子结果
    
    Args:
        results_list: 多次实验结果列表
        
    Returns:
        aggregated: 聚合后的统计结果
    """
    analyzer = StatisticalAnalyzer()
    return analyzer.aggregate_metrics(results_list)


def compute_conservation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算守恒量指标
    
    对于物理场，检查质量、动量、能量等守恒量的保持情况
    
    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        
    Returns:
        conservation_metrics: 守恒量指标
    """
    metrics = {}
    
    # 质量守恒（总和）
    pred_mass = torch.sum(pred, dim=(-2, -1))  # [B, C]
    target_mass = torch.sum(target, dim=(-2, -1))  # [B, C]
    mass_error = torch.abs(pred_mass - target_mass) / (torch.abs(target_mass) + 1e-8)
    metrics['mass_conservation_error'] = mass_error
    
    # 能量守恒（L2范数的平方）
    pred_energy = torch.sum(pred**2, dim=(-2, -1))  # [B, C]
    target_energy = torch.sum(target**2, dim=(-2, -1))  # [B, C]
    energy_error = torch.abs(pred_energy - target_energy) / (target_energy + 1e-8)
    metrics['energy_conservation_error'] = energy_error
    
    # 动量守恒（一阶矩）
    H, W = pred.shape[-2:]
    y_coords = torch.arange(H, dtype=pred.dtype, device=pred.device).view(H, 1)
    x_coords = torch.arange(W, dtype=pred.dtype, device=pred.device).view(1, W)
    
    pred_momentum_y = torch.sum(pred * y_coords.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
    target_momentum_y = torch.sum(target * y_coords.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
    momentum_y_error = torch.abs(pred_momentum_y - target_momentum_y) / (torch.abs(target_momentum_y) + 1e-8)
    metrics['momentum_y_conservation_error'] = momentum_y_error
    
    pred_momentum_x = torch.sum(pred * x_coords.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
    target_momentum_x = torch.sum(target * x_coords.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
    momentum_x_error = torch.abs(pred_momentum_x - target_momentum_x) / (torch.abs(target_momentum_x) + 1e-8)
    metrics['momentum_x_conservation_error'] = momentum_x_error
    
    return metrics


def compute_spectral_analysis(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算频谱分析指标
    
    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        
    Returns:
        spectral_metrics: 频谱分析指标
    """
    metrics = {}
    
    # FFT变换
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    
    # 功率谱
    pred_power = torch.abs(pred_fft)**2
    target_power = torch.abs(target_fft)**2
    
    # 功率谱误差
    power_error = torch.mean((pred_power - target_power)**2, dim=(-2, -1))
    metrics['power_spectrum_mse'] = power_error
    
    # 相位误差
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    phase_diff = torch.abs(pred_phase - target_phase)
    # 处理相位的周期性
    phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
    phase_error = torch.mean(phase_diff**2, dim=(-2, -1))
    metrics['phase_mse'] = phase_error
    
    # 频域相关系数
    pred_fft_flat = pred_fft.view(pred_fft.shape[0], pred_fft.shape[1], -1)
    target_fft_flat = target_fft.view(target_fft.shape[0], target_fft.shape[1], -1)
    
    # 计算复数相关系数的实部
    correlation = torch.real(torch.sum(pred_fft_flat * torch.conj(target_fft_flat), dim=-1))
    pred_norm = torch.sqrt(torch.sum(torch.abs(pred_fft_flat)**2, dim=-1))
    target_norm = torch.sqrt(torch.sum(torch.abs(target_fft_flat)**2, dim=-1))
    
    freq_correlation = correlation / (pred_norm * target_norm + 1e-8)
    metrics['frequency_correlation'] = freq_correlation
    
    return metrics


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                   obs_data: Optional[Dict] = None,
                   norm_stats: Optional[Dict] = None,
                   image_size: Tuple[int, int] = (256, 256)) -> Dict[str, torch.Tensor]:
    """便捷函数：计算基础指标（兼容性函数）
    
    Args:
        pred: 预测值 [B, C, H, W]
        target: 真实值 [B, C, H, W]
        obs_data: 观测数据（用于数据一致性）
        norm_stats: 归一化统计量
        image_size: 图像尺寸
        
    Returns:
        metrics: 基础指标的字典
    """
    return compute_all_metrics(pred, target, obs_data, norm_stats, image_size)