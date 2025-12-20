"""
数据一致性检查工具 - Data Consistency Checker
用于验证训练流程中的数据一致性，确保观测算子H与训练DC使用相同的实现和配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from omegaconf import DictConfig
import logging
from pathlib import Path
from datetime import datetime


class DataConsistencyChecker:
    """数据一致性检查器"""
    
    def __init__(self, config: Optional[DictConfig] = None, tolerance: float = 1e-8):
        self.config = config
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
        # 检查配置
        self.required_keys = [
            'data.degradation.downsample_factor',
            'data.degradation.blur_kernel_size',
            'data.degradation.blur_sigma',
            'data.degradation.crop_size',
            'data.degradation.crop_strategy'
        ]

    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        required = [
            ("model", "spatial", "in_channels"),
            ("data", "normalize"),
        ]
        missing = []
        for path in required:
            cur: Any = config
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if not ok:
                missing.append(".".join(path))

        return {
            "valid": len(missing) == 0,
            "missing_fields": missing,
        }
        
    def validate_config(self) -> bool:
        """验证配置完整性"""
        if self.config is None:
            return False
        for key in self.required_keys:
            if not self._check_nested_key(self.config, key):
                self.logger.warning(f"Missing configuration key: {key}")
                return False
        return True
    
    def _check_nested_key(self, config: DictConfig, key_path: str) -> bool:
        """检查嵌套键是否存在"""
        keys = key_path.split('.')
        current = config
        for key in keys:
            if not hasattr(current, key):
                return False
            current = getattr(current, key)
        return True
    
    def check_observation_consistency(
        self, 
        ground_truth: torch.Tensor, 
        observed_data: torch.Tensor,
        degradation_operator: nn.Module,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        检查观测一致性
        
        Args:
            ground_truth: 真实值 [B, T, C, H, W]
            observed_data: 观测值 [B, T, C, H', W']
            degradation_operator: 降质算子
            sample_indices: 要检查的样本索引列表
            
        Returns:
            一致性检查结果
        """
        if sample_indices is None:
            sample_indices = list(range(min(100, ground_truth.shape[0])))  # 默认检查前100个样本
        
        results = {
            'consistent': True,
            'max_error': 0.0,
            'mean_error': 0.0,
            'failed_samples': [],
            'sample_errors': []
        }
        
        for idx in sample_indices:
            if idx >= ground_truth.shape[0]:
                break
                
            # 应用降质算子到真实值
            gt_sample = ground_truth[idx:idx+1]  # 保持batch维度
            degraded_gt = degradation_operator(gt_sample)
            
            # 比较观测值
            observed_sample = observed_data[idx:idx+1]
            
            # 处理尺寸不匹配
            if degraded_gt.shape != observed_sample.shape:
                self.logger.warning(f"Shape mismatch: degraded {degraded_gt.shape} vs observed {observed_sample.shape}")
                # 使用插值对齐
                degraded_gt = F.interpolate(
                    degraded_gt.view(-1, *degraded_gt.shape[2:]),
                    size=observed_sample.shape[2:4],
                    mode='bilinear',
                    align_corners=False
                ).view(observed_sample.shape)
            
            # 计算误差
            error = torch.abs(degraded_gt - observed_sample)
            max_error = error.max().item()
            mean_error = error.mean().item()
            
            # 记录样本误差
            results['sample_errors'].append({
                'sample_idx': idx,
                'max_error': max_error,
                'mean_error': mean_error
            })
            
            # 检查是否通过一致性检查
            if max_error > self.tolerance:
                results['consistent'] = False
                results['failed_samples'].append({
                    'sample_idx': idx,
                    'max_error': max_error,
                    'mean_error': mean_error
                })
                self.logger.warning(f"Sample {idx} failed consistency check: max_error={max_error:.2e}, mean_error={mean_error:.2e}")
            
            # 更新全局误差
            results['max_error'] = max(results['max_error'], max_error)
            results['mean_error'] = (results['mean_error'] * len(results['sample_errors']) + mean_error) / len(results['sample_errors'])
        
        return results
    
    def check_data_pipeline_consistency(
        self,
        raw_data: torch.Tensor,
        processed_data: torch.Tensor,
        data_pipeline: Any,
        check_normalization: bool = True
    ) -> Dict[str, Any]:
        """
        检查数据处理管道一致性
        
        Args:
            raw_data: 原始数据
            processed_data: 处理后的数据
            data_pipeline: 数据处理管道
            check_normalization: 是否检查归一化
            
        Returns:
            一致性检查结果
        """
        issues: List[str] = []
        shape_ok = self._check_shape_consistency(raw_data, processed_data)
        range_ok = self._check_value_range(processed_data)
        results: Dict[str, Any] = {
            'consistent': True,
            'issues': issues,
            'shape_check': shape_ok,
            'range_check': range_ok,
            'normalization_check': None,
        }
        
        # 形状一致性检查
        if not results['shape_check']:
            results['consistent'] = False
            issues.append("shape_inconsistent")
            self.logger.error("Shape inconsistency detected in data pipeline")
        
        # 值域检查
        if not results['range_check']:
            results['consistent'] = False
            issues.append("value_range_inconsistent")
            self.logger.warning("Value range inconsistency detected")
        
        # 归一化检查
        if check_normalization:
            results['normalization_check'] = self._check_normalization_consistency(processed_data)
            if not results['normalization_check']['is_normalized']:
                issues.append("normalization_inconsistent")
                self.logger.warning("Data appears to be not properly normalized")
        
        return results
    
    def _check_shape_consistency(self, raw_data: torch.Tensor, processed_data: torch.Tensor) -> bool:
        """检查形状一致性"""
        if raw_data.dim() != processed_data.dim():
            self.logger.error(f"Dim mismatch: raw_dim={raw_data.dim()}, processed_dim={processed_data.dim()}")
            return False

        if raw_data.dim() == 4:
            if raw_data.shape[0] != processed_data.shape[0]:
                self.logger.error(f"Batch dimension mismatch: raw={raw_data.shape[0]}, processed={processed_data.shape[0]}")
                return False
            if raw_data.shape[1] != processed_data.shape[1]:
                self.logger.error(f"Channel dimension mismatch: raw={raw_data.shape[1]}, processed={processed_data.shape[1]}")
                return False
            if raw_data.shape[2:] != processed_data.shape[2:]:
                self.logger.error(f"Spatial dimension mismatch: raw={raw_data.shape[2:]}, processed={processed_data.shape[2:]}")
                return False
            return True

        if raw_data.dim() == 5:
            if raw_data.shape[1] != processed_data.shape[1]:
                self.logger.error(f"Time dimension mismatch: raw={raw_data.shape[1]}, processed={processed_data.shape[1]}")
                return False
            if raw_data.shape[2] != processed_data.shape[2]:
                self.logger.error(f"Channel dimension mismatch: raw={raw_data.shape[2]}, processed={processed_data.shape[2]}")
                return False
            return True

        if raw_data.shape != processed_data.shape:
            self.logger.error(f"Shape mismatch: raw={raw_data.shape}, processed={processed_data.shape}")
            return False
        return True
    
    def _check_value_range(self, data: torch.Tensor, valid_range: Tuple[float, float] = (-10.0, 10.0)) -> bool:
        """检查值域是否合理"""
        min_val = data.min().item()
        max_val = data.max().item()
        
        if min_val < valid_range[0] or max_val > valid_range[1]:
            self.logger.warning(f"Value range [{min_val:.3f}, {max_val:.3f}] outside expected range {valid_range}")
            return False
        
        return True
    
    def _check_normalization_consistency(self, data: torch.Tensor, tolerance: float = 3.0) -> Dict[str, Any]:
        """检查归一化一致性"""
        mean = data.mean().item()
        std = data.std().item()
        
        # 检查是否接近标准正态分布（均值为0，标准差为1）
        mean_deviation = abs(mean)
        std_deviation = abs(std - 1.0)
        
        results = {
            'mean': mean,
            'std': std,
            'mean_deviation': mean_deviation,
            'std_deviation': std_deviation,
            'is_normalized': mean_deviation < tolerance and std_deviation < tolerance
        }
        
        return results
    
    def check_temporal_consistency(
        self,
        pred_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        temporal_smoothness_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        检查时序一致性
        
        Args:
            pred_sequence: 预测序列 [B, T, C, H, W]
            target_sequence: 目标序列 [B, T, C, H, W]
            temporal_smoothness_threshold: 时序平滑度阈值
            
        Returns:
            时序一致性检查结果
        """
        results = {
            'consistent': True,
            'temporal_smoothness': 0.0,
            'temporal_correlation': 0.0,
            'error_growth_rate': 0.0
        }
        
        B, T, C, H, W = pred_sequence.shape
        
        # 计算时序平滑度
        pred_diff = pred_sequence[:, 1:] - pred_sequence[:, :-1]
        target_diff = target_sequence[:, 1:] - target_sequence[:, :-1]
        
        # 时序变化一致性
        smoothness_error = torch.abs(pred_diff - target_diff).mean().item()
        results['temporal_smoothness'] = 1.0 - smoothness_error
        
        # 时序相关性
        pred_temporal = pred_sequence.reshape(B, T, -1)
        target_temporal = target_sequence.reshape(B, T, -1)
        
        correlations = []
        for b in range(B):
            for i in range(pred_temporal.shape[2]):
                pred_seq = pred_temporal[b, :, i]
                target_seq = target_temporal[b, :, i]
                
                # 计算皮尔逊相关系数
                pred_mean = pred_seq.mean()
                target_mean = target_seq.mean()
                
                pred_centered = pred_seq - pred_mean
                target_centered = target_seq - target_mean
                
                pred_norm = torch.sqrt(torch.sum(pred_centered ** 2))
                target_norm = torch.sqrt(torch.sum(target_centered ** 2))
                
                if pred_norm > 1e-8 and target_norm > 1e-8:
                    correlation = torch.sum(pred_centered * target_centered) / (pred_norm * target_norm)
                    correlations.append(correlation.item())
        
        results['temporal_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # 误差增长率
        errors = torch.linalg.vector_norm(pred_sequence - target_sequence, dim=(2, 3, 4))  # [B, T]
        error_growth_rates = []
        
        for b in range(B):
            for t in range(1, T):
                if errors[b, t-1].item() > 1e-8:
                    growth_rate = (errors[b, t].item() - errors[b, t-1].item()) / errors[b, t-1].item()
                    error_growth_rates.append(growth_rate)
        
        results['error_growth_rate'] = np.mean(error_growth_rates) if error_growth_rates else 0.0
        
        # 一致性判断
        results['consistent'] = (
            results['temporal_smoothness'] > (1.0 - temporal_smoothness_threshold) and
            results['temporal_correlation'] > 0.7 and
            abs(results['error_growth_rate']) < 0.5
        )
        
        if not results['consistent']:
            self.logger.warning(
                f"Temporal consistency check failed: "
                f"smoothness={results['temporal_smoothness']:.3f}, "
                f"correlation={results['temporal_correlation']:.3f}, "
                f"error_growth={results['error_growth_rate']:.3f}"
            )
        
        return results
    
    def generate_consistency_report(
        self,
        observation_results: Dict[str, Any],
        pipeline_results: Dict[str, Any],
        temporal_results: Dict[str, Any]
    ) -> str:
        """生成一致性检查报告"""
        report_lines = [
            "=== Data Consistency Check Report ===",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. Observation Consistency:",
            f"   Overall: {'PASSED' if observation_results['consistent'] else 'FAILED'}",
            f"   Max Error: {observation_results['max_error']:.2e}",
            f"   Mean Error: {observation_results['mean_error']:.2e}",
            f"   Failed Samples: {len(observation_results['failed_samples'])}",
            "",
            "2. Data Pipeline Consistency:",
            f"   Overall: {'PASSED' if pipeline_results['consistent'] else 'FAILED'}",
            f"   Shape Check: {'PASSED' if pipeline_results['shape_check'] else 'FAILED'}",
            f"   Range Check: {'PASSED' if pipeline_results['range_check'] else 'FAILED'}",
        ]
        
        if pipeline_results['normalization_check']:
            norm_check = pipeline_results['normalization_check']
            report_lines.extend([
                f"   Normalization: {'PASSED' if norm_check['is_normalized'] else 'FAILED'}",
                f"   Mean Deviation: {norm_check['mean_deviation']:.3f}",
                f"   Std Deviation: {norm_check['std_deviation']:.3f}"
            ])
        
        report_lines.extend([
            "",
            "3. Temporal Consistency:",
            f"   Overall: {'PASSED' if temporal_results['consistent'] else 'FAILED'}",
            f"   Temporal Smoothness: {temporal_results['temporal_smoothness']:.3f}",
            f"   Temporal Correlation: {temporal_results['temporal_correlation']:.3f}",
            f"   Error Growth Rate: {temporal_results['error_growth_rate']:.3f}"
        ])
        
        return "\n".join(report_lines)
    
    def save_consistency_report(self, report: str, output_path: str):
        """保存一致性检查报告"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Consistency report saved to {output_path}")


class DegradationEquivalenceChecker:
    """降质等价性检查器 - 确保训练DC与观测H使用相同实现"""
    
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def check_equivalence(
        self,
        degradation_op1: nn.Module,
        degradation_op2: nn.Module,
        test_data: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        检查两个降质算子是否等价
        
        Args:
            degradation_op1: 第一个降质算子（通常是观测H）
            degradation_op2: 第二个降质算子（通常是训练DC）
            test_data: 测试数据
            num_samples: 检查样本数
            
        Returns:
            等价性检查结果
        """
        results = {
            'equivalent': True,
            'max_mse': 0.0,
            'mean_mse': 0.0,
            'failed_cases': [],
            'sample_results': []
        }
        
        # 随机选择样本
        sample_indices = torch.randperm(test_data.shape[0])[:num_samples]
        
        for i, idx in enumerate(sample_indices):
            sample = test_data[idx:idx+1]
            
            # 应用两个降质算子
            result1 = degradation_op1(sample)
            result2 = degradation_op2(sample)
            
            # 计算MSE
            mse = F.mse_loss(result1, result2).item()
            
            # 记录结果
            sample_result = {
                'sample_idx': idx.item(),
                'mse': mse,
                'shape1': list(result1.shape),
                'shape2': list(result2.shape)
            }
            results['sample_results'].append(sample_result)
            
            # 检查是否通过
            if mse > self.tolerance:
                results['equivalent'] = False
                results['failed_cases'].append(sample_result)
                self.logger.warning(f"Sample {idx.item()} failed equivalence check: MSE={mse:.2e}")
            
            # 更新全局误差
            results['max_mse'] = max(results['max_mse'], mse)
            results['mean_mse'] = (results['mean_mse'] * i + mse) / (i + 1)

        results['mse_error'] = results['mean_mse']
        results['max_error'] = results['max_mse']
        
        return results
    
    def generate_equivalence_report(self, results: Dict[str, Any]) -> str:
        """生成等价性检查报告"""
        report_lines = [
            "=== Degradation Equivalence Check Report ===",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Overall: {'EQUIVALENT' if results['equivalent'] else 'NOT EQUIVALENT'}",
            f"Max MSE: {results['max_mse']:.2e}",
            f"Mean MSE: {results['mean_mse']:.2e}",
            f"Failed Cases: {len(results['failed_cases'])}/{len(results['sample_results'])}",
            "",
            "Sample Results (first 10):"
        ]
        
        for i, sample in enumerate(results['sample_results'][:10]):
            report_lines.append(
                f"  Sample {sample['sample_idx']}: MSE={sample['mse']:.2e}, "
                f"Shapes: {sample['shape1']} vs {sample['shape2']}"
            )
        
        return "\n".join(report_lines)
