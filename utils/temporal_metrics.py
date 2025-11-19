"""时序AR模型多步指标计算模块

实现时序自回归模型的专用评估指标：
- 多步预测指标：rel2_mean/last/worst
- 时序一致性指标
- 延迟统计
- 累积误差分析

按照开发手册要求：
- 每通道先算，后等权平均
- 支持统计分析（均值±标准差）
- 支持多步预测的详细分析
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from utils.metrics import MetricsCalculator


@dataclass
class TemporalMetrics:
    """时序指标数据类"""
    # 多步Rel-L2指标
    rel2_mean: float  # 所有步骤的平均Rel-L2
    rel2_last: float  # 最后一步的Rel-L2
    rel2_worst: float  # 最差步骤的Rel-L2
    
    # 多步MAE指标
    mae_mean: float   # 所有步骤的平均MAE
    mae_last: float   # 最后一步的MAE
    mae_worst: float  # 最差步骤的MAE
    
    # 时序一致性
    temporal_consistency: float  # 时序一致性指标
    
    # 延迟统计
    inference_latency_ms: float  # 推理延迟（毫秒）
    per_step_latency_ms: float   # 每步平均延迟（毫秒）
    
    # 累积误差
    error_accumulation_rate: float  # 误差累积率
    
    # 详细步骤指标
    step_wise_rel2: List[float]  # 每步的Rel-L2
    step_wise_mae: List[float]   # 每步的MAE


class TemporalMetricsCalculator:
    """时序指标计算器
    
    专门用于时序AR模型的多步预测指标计算
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            image_size: 图像尺寸 (H, W)
        """
        self.image_size = image_size
        self.base_calculator = MetricsCalculator(image_size)
    
    def compute_temporal_metrics(self, 
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                inference_time: Optional[float] = None) -> TemporalMetrics:
        """计算完整的时序指标
        
        Args:
            predictions: 预测序列 [B, T, C, H, W]
            targets: 目标序列 [B, T, C, H, W]
            inference_time: 推理时间（秒）
            
        Returns:
            TemporalMetrics: 完整的时序指标
        """
        if len(predictions.shape) != 5 or len(targets.shape) != 5:
            raise ValueError("输入必须是5D张量 [B, T, C, H, W]")
        
        B, T, C, H, W = predictions.shape
        
        # 计算每步的指标
        step_wise_rel2 = []
        step_wise_mae = []
        
        for t in range(T):
            pred_t = predictions[:, t]  # [B, C, H, W]
            target_t = targets[:, t]    # [B, C, H, W]
            
            # 计算该步骤的指标
            rel2_t = self.base_calculator.compute_rel_l2(pred_t, target_t)
            mae_t = self.base_calculator.compute_mae(pred_t, target_t)
            
            # 取平均（跨批次和通道）
            step_wise_rel2.append(torch.mean(rel2_t).item())
            step_wise_mae.append(torch.mean(mae_t).item())
        
        # 计算汇总指标
        rel2_mean = np.mean(step_wise_rel2)
        rel2_last = step_wise_rel2[-1]
        rel2_worst = np.max(step_wise_rel2)
        
        mae_mean = np.mean(step_wise_mae)
        mae_last = step_wise_mae[-1]
        mae_worst = np.max(step_wise_mae)
        
        # 计算时序一致性
        temporal_consistency = self._compute_temporal_consistency(predictions)
        
        # 计算误差累积率
        error_accumulation_rate = self._compute_error_accumulation_rate(step_wise_rel2)
        
        # 计算延迟统计
        if inference_time is not None:
            inference_latency_ms = inference_time * 1000
            per_step_latency_ms = inference_latency_ms / T
        else:
            inference_latency_ms = 0.0
            per_step_latency_ms = 0.0
        
        return TemporalMetrics(
            rel2_mean=rel2_mean,
            rel2_last=rel2_last,
            rel2_worst=rel2_worst,
            mae_mean=mae_mean,
            mae_last=mae_last,
            mae_worst=mae_worst,
            temporal_consistency=temporal_consistency,
            inference_latency_ms=inference_latency_ms,
            per_step_latency_ms=per_step_latency_ms,
            error_accumulation_rate=error_accumulation_rate,
            step_wise_rel2=step_wise_rel2,
            step_wise_mae=step_wise_mae
        )
    
    def _compute_temporal_consistency(self, predictions: torch.Tensor) -> float:
        """计算时序一致性指标
        
        通过计算相邻时间步之间的平滑度来衡量时序一致性
        """
        B, T, C, H, W = predictions.shape
        
        if T < 2:
            return 1.0  # 单步预测认为完全一致
        
        # 计算相邻时间步的差异
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]  # [B, T-1, C, H, W]
        
        # 计算平均时序变化率
        temporal_variation = torch.mean(torch.abs(temporal_diff))
        
        # 计算数据的总体变化范围
        data_range = torch.max(predictions) - torch.min(predictions)
        
        # 时序一致性 = 1 - (时序变化率 / 数据范围)
        consistency = 1.0 - (temporal_variation / (data_range + 1e-8))
        consistency = torch.clamp(consistency, 0.0, 1.0)
        
        return consistency.item()
    
    def _compute_error_accumulation_rate(self, step_wise_errors: List[float]) -> float:
        """计算误差累积率
        
        通过拟合误差随时间的增长趋势来计算累积率
        """
        if len(step_wise_errors) < 2:
            return 0.0
        
        # 使用线性回归拟合误差增长趋势
        time_steps = np.arange(len(step_wise_errors))
        
        # 计算斜率（误差增长率）
        if len(step_wise_errors) > 1:
            slope = np.polyfit(time_steps, step_wise_errors, 1)[0]
            # 归一化到初始误差
            initial_error = step_wise_errors[0] if step_wise_errors[0] > 0 else 1e-8
            accumulation_rate = slope / initial_error
        else:
            accumulation_rate = 0.0
        
        return accumulation_rate
    
    def compute_rollout_metrics(self, 
                               model: torch.nn.Module,
                               initial_condition: torch.Tensor,
                               target_sequence: torch.Tensor,
                               rollout_steps: int,
                               device: torch.device) -> TemporalMetrics:
        """计算自回归展开的指标
        
        Args:
            model: AR模型
            initial_condition: 初始条件 [B, C, H, W]
            target_sequence: 目标序列 [B, T, C, H, W]
            rollout_steps: 展开步数
            device: 计算设备
            
        Returns:
            TemporalMetrics: 时序指标
        """
        model.eval()
        
        B, C, H, W = initial_condition.shape
        predictions = torch.zeros(B, rollout_steps, C, H, W, device=device)
        
        # 记录推理时间
        start_time = time.time()
        
        with torch.no_grad():
            current_state = initial_condition
            
            for t in range(rollout_steps):
                # 单步预测
                pred = model(current_state)
                predictions[:, t] = pred
                
                # 更新状态用于下一步预测
                current_state = pred
        
        inference_time = time.time() - start_time
        
        # 确保目标序列长度匹配
        if target_sequence.shape[1] > rollout_steps:
            target_sequence = target_sequence[:, :rollout_steps]
        elif target_sequence.shape[1] < rollout_steps:
            # 如果目标序列较短，只计算可用步数的指标
            rollout_steps = target_sequence.shape[1]
            predictions = predictions[:, :rollout_steps]
        
        return self.compute_temporal_metrics(predictions, target_sequence, inference_time)
    
    def compute_batch_statistics(self, 
                                metrics_list: List[TemporalMetrics]) -> Dict[str, Dict[str, float]]:
        """计算批量指标的统计信息
        
        Args:
            metrics_list: 多个样本的指标列表
            
        Returns:
            Dict: 统计信息 {'metric_name': {'mean': float, 'std': float, 'min': float, 'max': float}}
        """
        if not metrics_list:
            return {}
        
        # 收集所有指标值
        rel2_mean_values = [m.rel2_mean for m in metrics_list]
        rel2_last_values = [m.rel2_last for m in metrics_list]
        rel2_worst_values = [m.rel2_worst for m in metrics_list]
        
        mae_mean_values = [m.mae_mean for m in metrics_list]
        mae_last_values = [m.mae_last for m in metrics_list]
        mae_worst_values = [m.mae_worst for m in metrics_list]
        
        temporal_consistency_values = [m.temporal_consistency for m in metrics_list]
        inference_latency_values = [m.inference_latency_ms for m in metrics_list]
        per_step_latency_values = [m.per_step_latency_ms for m in metrics_list]
        error_accumulation_values = [m.error_accumulation_rate for m in metrics_list]
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            """计算统计信息"""
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return {
            'rel2_mean': compute_stats(rel2_mean_values),
            'rel2_last': compute_stats(rel2_last_values),
            'rel2_worst': compute_stats(rel2_worst_values),
            'mae_mean': compute_stats(mae_mean_values),
            'mae_last': compute_stats(mae_last_values),
            'mae_worst': compute_stats(mae_worst_values),
            'temporal_consistency': compute_stats(temporal_consistency_values),
            'inference_latency_ms': compute_stats(inference_latency_values),
            'per_step_latency_ms': compute_stats(per_step_latency_values),
            'error_accumulation_rate': compute_stats(error_accumulation_values)
        }
    
    def format_metrics_report(self, 
                             metrics: TemporalMetrics,
                             statistics: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """格式化指标报告
        
        Args:
            metrics: 单个样本的指标
            statistics: 批量统计信息（可选）
            
        Returns:
            str: 格式化的报告
        """
        report = []
        report.append("=" * 60)
        report.append("时序AR模型评估报告")
        report.append("=" * 60)
        
        # 多步Rel-L2指标
        report.append("\n📊 多步Rel-L2指标:")
        report.append(f"  平均Rel-L2: {metrics.rel2_mean:.6f}")
        report.append(f"  最后步Rel-L2: {metrics.rel2_last:.6f}")
        report.append(f"  最差步Rel-L2: {metrics.rel2_worst:.6f}")
        
        # 多步MAE指标
        report.append("\n📊 多步MAE指标:")
        report.append(f"  平均MAE: {metrics.mae_mean:.6f}")
        report.append(f"  最后步MAE: {metrics.mae_last:.6f}")
        report.append(f"  最差步MAE: {metrics.mae_worst:.6f}")
        
        # 时序特性
        report.append("\n⏱️ 时序特性:")
        report.append(f"  时序一致性: {metrics.temporal_consistency:.4f}")
        report.append(f"  误差累积率: {metrics.error_accumulation_rate:.6f}")
        
        # 性能统计
        report.append("\n🚀 性能统计:")
        report.append(f"  总推理延迟: {metrics.inference_latency_ms:.2f} ms")
        report.append(f"  每步平均延迟: {metrics.per_step_latency_ms:.2f} ms")
        
        # 步骤详情
        if len(metrics.step_wise_rel2) <= 10:  # 只显示较短序列的详情
            report.append("\n📈 步骤详情:")
            for i, (rel2, mae) in enumerate(zip(metrics.step_wise_rel2, metrics.step_wise_mae)):
                report.append(f"  步骤 {i+1}: Rel-L2={rel2:.6f}, MAE={mae:.6f}")
        
        # 批量统计（如果提供）
        if statistics:
            report.append("\n📊 批量统计 (均值±标准差):")
            for metric_name, stats in statistics.items():
                report.append(f"  {metric_name}: {stats['mean']:.6f}±{stats['std']:.6f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)