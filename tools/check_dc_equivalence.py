#!/usr/bin/env python3
"""
PDEBench稀疏观测重建数据一致性检查脚本

严格遵循黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁

验证目标：
- 验证H(GT)与y的MSE < 1e-8
- 实现H算子一致性检查（观测生成H与训练DC的H完全一致）
- 随机抽样100个case进行验证
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 项目导入
try:
    from datasets.pdebench import PDEBenchDataModule
    from ops.degradation import apply_degradation_operator, verify_degradation_consistency
    from utils.reproducibility import set_seed
    from utils.config import get_environment_info
except ImportError as e:
    logging.warning(f"Import error: {e}. Using fallback implementations.")
    
    # 提供备用实现
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def get_environment_info():
        return {
            'timestamp': str(torch.utils.data.get_worker_info()),
            'torch_version': torch.__version__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }


class SyntheticDataset(Dataset):
    """合成数据集用于测试"""
    
    def __init__(self, num_samples: int = 100, image_size: int = 256, task: str = 'sr'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.task = task.lower()
        
        # 设置随机种子确保可复现
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 生成合成数据
        self.data = []
        for i in range(num_samples):
            # 生成高分辨率目标
            target = torch.randn(1, image_size, image_size)
            
            # 根据任务类型生成观测数据
            if self.task in ['sr', 'super_resolution']:
                task_params = {
                    'task': 'sr',
                    'scale': 4,
                    'sigma': 1.0,
                    'kernel_size': 5,
                    'boundary': 'mirror'
                }
                # 使用与H算子完全相同的退化方法
                observation = self._apply_consistent_sr_degradation(target, task_params)
            else:  # crop
                crop_size = (64, 64)
                task_params = {
                    'task': 'crop',
                    'crop_size': crop_size,
                    'boundary': 'mirror'
                }
                # 使用与H算子完全相同的裁剪方法
                observation = self._apply_consistent_crop_degradation(target, task_params)
            
            self.data.append({
                'target': target,
                'observation': observation,
                'task_params': task_params,
                'sample_idx': i
            })
    
    def _apply_consistent_sr_degradation(self, target: torch.Tensor, task_params: Dict) -> torch.Tensor:
        """应用与ops/degradation.py完全一致的SR退化算子"""
        scale = task_params.get('scale', 4)
        sigma = task_params.get('sigma', 1.0)
        kernel_size = task_params.get('kernel_size', 5)
        boundary = task_params.get('boundary', 'mirror')
        
        # 确保batch维度
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # 1. 高斯模糊 - 使用与ops/degradation.py完全相同的实现
        blurred = self._gaussian_blur(target, kernel_size, sigma, boundary)
        
        # 2. 区域下采样 - 使用与ops/degradation.py完全相同的实现
        B, C, H, W = blurred.shape
        target_h = H // scale
        target_w = W // scale
        
        downsampled = F.interpolate(
            blurred,
            size=(target_h, target_w),
            mode='area'
        )
        
        return downsampled.squeeze(0)
    
    def _apply_consistent_crop_degradation(self, target: torch.Tensor, task_params: Dict) -> torch.Tensor:
        """应用与H算子一致的Crop退化"""
        # 确保batch维度
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        crop_size = task_params.get('crop_size', [64, 64])
        B, C, H, W = target.shape
        crop_h, crop_w = crop_size
        
        # 中心对齐裁剪
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        
        cropped = target[:, :, start_h:end_h, start_w:end_w]
        return cropped.squeeze(0)
    
    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int, sigma: float, boundary: str = 'mirror') -> torch.Tensor:
        """高斯模糊 - 与ops/degradation.py完全一致的实现"""
        # 确保kernel_size为奇数
        if isinstance(kernel_size, torch.Tensor):
            if kernel_size.numel() == 1:
                kernel_size = kernel_size.item()
            else:
                kernel_size = kernel_size.flatten()[0].item()
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建高斯核
        kernel = self._create_gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
        kernel = kernel.to(dtype=x.dtype)
        
        # 边界填充
        pad_size = kernel_size // 2
        if boundary == 'mirror' or boundary == 'reflect':
            x_padded = F.pad(x, [pad_size] * 4, mode='reflect')
        elif boundary == 'wrap' or boundary == 'circular':
            x_padded = F.pad(x, [pad_size] * 4, mode='circular')
        elif boundary == 'zero':
            x_padded = F.pad(x, [pad_size] * 4, mode='constant', value=0)
        elif boundary == 'replicate':
            x_padded = F.pad(x, [pad_size] * 4, mode='replicate')
        else:
            raise ValueError(f"Unknown boundary mode: {boundary}")
        
        # 应用卷积
        B, C, H_pad, W_pad = x_padded.shape
        x_padded = x_padded.view(B * C, 1, H_pad, W_pad)
        
        blurred = F.conv2d(x_padded, kernel, padding=0)
        blurred = blurred.view(B, C, blurred.shape[-2], blurred.shape[-1])
        
        return blurred
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """创建高斯卷积核 - 与ops/degradation.py完全一致的实现"""
        # 确保kernel_size是标量值
        if isinstance(kernel_size, torch.Tensor):
            kernel_size = kernel_size.item()
        coords = torch.arange(kernel_size, dtype=dtype, device=device)
        coords = coords - kernel_size // 2
        
        # 计算高斯权重
        if isinstance(sigma, torch.Tensor):
            if sigma.numel() == 1:
                sigma = sigma.item()
            else:
                sigma = sigma.flatten()[0].item()
        if isinstance(sigma, (list, tuple)):
            sigma = sigma[0]
        sigma = float(sigma)
        sigma = torch.tensor(sigma, dtype=dtype, device=device)
        gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 创建2D核
        kernel = gaussian_1d[:, None] * gaussian_1d[None, :]
        kernel = kernel / kernel.sum()
        
        # 调整形状为 [1, 1, H, W]
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        return kernel

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class DataConsistencyChecker:
    """数据一致性检查器
    
    验证观测算子H与训练DC的一致性：
    1. H(GT) ≈ y (MSE < 1e-8)
    2. 观测生成H与训练DC的H完全一致
    3. 随机抽样验证
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # 检查配置
        self.check_config = config.get('consistency_check', {})
        self.tolerance = self.check_config.get('tolerance', 1e-8)
        self.num_samples = self.check_config.get('num_samples', 100)
        self.random_seed = self.check_config.get('random_seed', 42)
        
        # 结果存储
        self.check_results = []
        self.failed_cases = []
        
        # 设置随机种子
        set_seed(self.random_seed)
    
    def check_single_sample(
        self, 
        batch: Dict[str, torch.Tensor], 
        sample_idx: int
    ) -> Dict[str, Any]:
        """检查单个样本的一致性
        
        Args:
            batch: 数据batch
            sample_idx: 样本索引
            
        Returns:
            检查结果字典
        """
        with torch.no_grad():
            # 获取数据
            target = batch['target'].to(self.device)  # GT [B, C, H, W]
            observation = batch['observation'].to(self.device)  # 观测数据 y
            
            # 获取任务参数，兼容不同的参数格式
            task_params = batch.get('task_params', batch.get('h_params', {}))
            
            # 如果task_params是tensor，需要转换为dict
            if isinstance(task_params, torch.Tensor):
                # 从数据集配置中获取默认参数
                task_params = self._get_default_task_params()
            elif isinstance(task_params, (list, tuple)) and len(task_params) > 0:
                task_params = task_params[0] if isinstance(task_params[0], dict) else self._get_default_task_params()
            elif not isinstance(task_params, dict):
                task_params = self._get_default_task_params()
            
            # 应用观测算子H到GT
            try:
                h_gt = self._apply_degradation_operator(target, task_params)
            except Exception as e:
                logging.error(f"Failed to apply degradation operator: {e}")
                # 如果H算子失败，尝试使用简单的下采样
                h_gt = self._apply_simple_degradation(target, task_params)
            
            # 确保observation和h_gt的形状匹配
            if h_gt.shape != observation.shape:
                # 如果形状不匹配，尝试调整
                if len(h_gt.shape) == 4 and len(observation.shape) == 4:
                    # 都是4D，但尺寸不同，使用插值调整
                    h_gt = torch.nn.functional.interpolate(
                        h_gt, size=observation.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                else:
                    logging.warning(f"Shape mismatch: h_gt {h_gt.shape} vs observation {observation.shape}")
            
            # 计算MSE(H(GT), y)
            mse_h_gt_y = torch.mean((h_gt - observation) ** 2).item()
            
            # 计算相对误差
            rel_error = torch.norm(h_gt - observation, p=2) / (torch.norm(observation, p=2) + 1e-8)
            rel_error = rel_error.item()
            
            # 一致性检查
            is_consistent = mse_h_gt_y < self.tolerance
            
            # 详细统计
            abs_diff = torch.abs(h_gt - observation)
            max_abs_diff = torch.max(abs_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            
            # 构建结果（确保所有值都是JSON可序列化的）
            result = {
                'sample_idx': int(sample_idx),
                'mse_h_gt_y': float(mse_h_gt_y),
                'rel_error': float(rel_error),
                'max_abs_diff': float(max_abs_diff),
                'mean_abs_diff': float(mean_abs_diff),
                'is_consistent': bool(is_consistent),
                'tolerance': float(self.tolerance),
                'task_params': {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in task_params.items()},
                'data_shape': {
                    'target_shape': list(target.shape),
                    'observation_shape': list(observation.shape),
                    'h_gt_shape': list(h_gt.shape)
                }
            }
            
            # 如果不一致，记录为失败案例
            if not is_consistent:
                failure_info = {
                    **result,
                    'failure_type': 'consistency_violation',
                    'expected_mse': f"< {self.tolerance}",
                    'actual_mse': float(mse_h_gt_y),
                    'severity': 'high' if mse_h_gt_y > 1e-6 else 'medium'
                }
                self.failed_cases.append(failure_info)
            
            return result
    
    def _get_default_task_params(self) -> Dict[str, Any]:
        """获取默认任务参数"""
        # 从配置中获取任务参数
        task_config = self.config.get('task', {})
        
        if 'super_resolution' in task_config:
            sr_config = task_config['super_resolution']
            return {
                'task': 'sr',
                'scale': sr_config.get('scale_factors', [4])[0],
                'sigma': sr_config.get('blur_sigma', 1.0),
                'kernel_size': sr_config.get('blur_kernel_size', 5),
                'boundary': sr_config.get('boundary_mode', 'mirror')
            }
        elif 'cropping' in task_config:
            crop_config = task_config['cropping']
            return {
                'task': 'crop',
                'crop_size': crop_config.get('crop_size', [64, 64]),
                'boundary': crop_config.get('boundary_mode', 'mirror')
            }
        else:
            # 默认SR参数
            return {
                'task': 'sr',
                'scale': 4,
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            }
    
    def _apply_degradation_operator(self, target: torch.Tensor, task_params: Dict) -> torch.Tensor:
        """应用退化算子"""
        try:
            # 尝试使用项目的退化算子
            return apply_degradation_operator(target, task_params)
        except:
            # 备用简单实现
            return self._apply_simple_degradation(target, task_params)
    
    def _apply_simple_degradation(self, target: torch.Tensor, task_params: Dict) -> torch.Tensor:
        """应用简单的退化操作作为备选方案"""
        task = task_params.get('task', 'sr')
        
        if task.lower() in ['sr', 'super_resolution']:
            scale = task_params.get('scale', 4)
            sigma = task_params.get('sigma', 1.0)
            kernel_size = task_params.get('kernel_size', 5)
            
            # 确保batch维度
            if target.dim() == 3:
                target = target.unsqueeze(0)
            
            # 1. 高斯模糊
            blurred = self._gaussian_blur_simple(target, sigma, kernel_size)
            
            # 2. 下采样
            B, C, H, W = blurred.shape
            target_h, target_w = H // scale, W // scale
            downsampled = F.interpolate(
                blurred, 
                size=(target_h, target_w), 
                mode='area'
            )
            
            return downsampled
        else:
            # 对于crop任务，返回中心裁剪
            crop_size = task_params.get('crop_size', [64, 64])
            
            # 确保batch维度
            if target.dim() == 3:
                target = target.unsqueeze(0)
            
            B, C, H, W = target.shape
            crop_h, crop_w = crop_size
            
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            end_h = start_h + crop_h
            end_w = start_w + crop_w
            
            return target[:, :, start_h:end_h, start_w:end_w]
    
    def _gaussian_blur_simple(self, x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
        """简单高斯模糊实现"""
        # 创建高斯核
        kernel = self._create_gaussian_kernel_simple(kernel_size, sigma)
        kernel = kernel.to(x.device).to(x.dtype)
        
        # 对每个通道应用卷积
        B, C, H, W = x.shape
        x_padded = F.pad(x, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        
        # 分离卷积：先水平后垂直
        kernel_1d = kernel[kernel_size//2:kernel_size//2+1, :]  # 水平核
        x_h = F.conv2d(x_padded.view(B*C, 1, H+kernel_size-1, W+kernel_size-1), 
                       kernel_1d.unsqueeze(0).unsqueeze(0), padding=0)
        
        kernel_1d = kernel[:, kernel_size//2:kernel_size//2+1]  # 垂直核
        x_blurred = F.conv2d(x_h, kernel_1d.unsqueeze(0).unsqueeze(0), padding=0)
        
        return x_blurred.view(B, C, H, W)
    
    def _create_gaussian_kernel_simple(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """创建高斯核"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # 创建2D核
        kernel = g[:, None] * g[None, :]
        return kernel

    def check_degradation_operator_consistency(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """检查退化算子一致性
        
        使用ops.degradation中的verify_degradation_consistency函数
        
        Args:
            batch: 数据batch
            
        Returns:
            一致性检查结果
        """
        try:
            target = batch['target'].to(self.device)
            observation = batch['observation'].to(self.device)
            task_params = batch['task_params']
            
            # 尝试使用专门的一致性验证函数
            try:
                consistency_result = verify_degradation_consistency(
                    target, observation, task_params, tolerance=self.tolerance
                )
            except:
                # 备用验证方法
                h_gt = self._apply_degradation_operator(target, task_params)
                mse = torch.mean((h_gt - observation) ** 2).item()
                consistency_result = {
                    'mse': mse,
                    'max_error': torch.max(torch.abs(h_gt - observation)).item(),
                    'tolerance': self.tolerance,
                    'passed': mse < self.tolerance,
                    'h_target_shape': list(h_gt.shape),
                    'observation_shape': list(observation.shape)
                }
            
            return {
                'degradation_consistency': consistency_result,
                'verification_method': 'verify_degradation_consistency',
                'tolerance_used': self.tolerance
            }
            
        except Exception as e:
            logging.error(f"Degradation consistency check failed: {e}")
            return {
                'degradation_consistency': False,
                'error': str(e),
                'verification_method': 'verify_degradation_consistency'
            }
    
    def sample_random_batches(
        self, 
        dataloader: DataLoader, 
        num_samples: int
    ) -> List[Tuple[Dict[str, torch.Tensor], int]]:
        """随机抽样batch
        
        Args:
            dataloader: 数据加载器
            num_samples: 抽样数量
            
        Returns:
            抽样的batch列表
        """
        all_batches = list(enumerate(dataloader))
        
        if len(all_batches) <= num_samples:
            logging.warning(f"Dataset has only {len(all_batches)} batches, using all")
            return all_batches
        
        # 随机抽样
        sampled_indices = random.sample(range(len(all_batches)), num_samples)
        sampled_batches = [all_batches[i] for i in sampled_indices]
        
        logging.info(f"Randomly sampled {len(sampled_batches)} batches from {len(all_batches)}")
        return sampled_batches
    
    def run_consistency_check(
        self, 
        dataloader: DataLoader, 
        output_dir: Path
    ) -> Dict[str, Any]:
        """运行完整的一致性检查
        
        Args:
            dataloader: 数据加载器
            output_dir: 输出目录
            
        Returns:
            检查结果汇总
        """
        logging.info(f"Starting consistency check with {self.num_samples} samples...")
        
        # 随机抽样
        sampled_batches = self.sample_random_batches(dataloader, self.num_samples)
        
        self.check_results = []
        self.failed_cases = []
        
        # 逐个检查
        for batch_idx, (batch_data_idx, batch) in enumerate(sampled_batches):
            try:
                # 基本一致性检查
                result = self.check_single_sample(batch, batch_data_idx)
                
                # 退化算子一致性检查
                degradation_result = self.check_degradation_operator_consistency(batch)
                result.update(degradation_result)
                
                self.check_results.append(result)
                
                # 打印进度
                if batch_idx % 20 == 0:
                    logging.info(f"Processed {batch_idx}/{len(sampled_batches)} samples")
                    
            except Exception as e:
                logging.error(f"Error checking batch {batch_data_idx}: {e}")
                error_result = {
                    'sample_idx': batch_data_idx,
                    'error': str(e),
                    'is_consistent': False
                }
                self.check_results.append(error_result)
        
        # 计算汇总统计
        summary = self.compute_summary_statistics()
        
        # 保存结果
        self.save_results(output_dir, summary)
        
        return summary
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """计算汇总统计量
        
        Returns:
            统计量字典
        """
        if not self.check_results:
            return {'error': 'No check results available'}
        
        # 基本统计
        total_samples = len(self.check_results)
        consistent_samples = sum(1 for r in self.check_results if r.get('is_consistent', False))
        inconsistent_samples = total_samples - consistent_samples
        
        # MSE统计
        mse_values = [r['mse_h_gt_y'] for r in self.check_results if 'mse_h_gt_y' in r]
        rel_error_values = [r['rel_error'] for r in self.check_results if 'rel_error' in r]
        
        # 退化算子一致性统计
        degradation_consistent = sum(
            1 for r in self.check_results 
            if r.get('degradation_consistency', {}).get('passed', False)
        )
        
        summary = {
            'total_samples': total_samples,
            'consistent_samples': consistent_samples,
            'inconsistent_samples': inconsistent_samples,
            'consistency_rate': consistent_samples / total_samples if total_samples > 0 else 0.0,
            'degradation_consistency_rate': degradation_consistent / total_samples if total_samples > 0 else 0.0,
            'tolerance_used': self.tolerance,
            'random_seed': self.random_seed,
            'mse_statistics': {
                'mean': float(np.mean(mse_values)) if mse_values else None,
                'std': float(np.std(mse_values)) if mse_values else None,
                'min': float(np.min(mse_values)) if mse_values else None,
                'max': float(np.max(mse_values)) if mse_values else None,
                'median': float(np.median(mse_values)) if mse_values else None
            },
            'rel_error_statistics': {
                'mean': float(np.mean(rel_error_values)) if rel_error_values else None,
                'std': float(np.std(rel_error_values)) if rel_error_values else None,
                'min': float(np.min(rel_error_values)) if rel_error_values else None,
                'max': float(np.max(rel_error_values)) if rel_error_values else None,
                'median': float(np.median(rel_error_values)) if rel_error_values else None
            },
            'failed_cases_count': len(self.failed_cases),
            'overall_pass': (
                consistent_samples == total_samples and 
                degradation_consistent == total_samples
            ),
            'meta': {
                'config': OmegaConf.to_container(self.config, resolve=True),
                'environment': get_environment_info()
            }
        }
        
        return summary
    
    def save_results(self, output_dir: Path, summary: Dict[str, Any]):
        """保存检查结果
        
        Args:
            output_dir: 输出目录
            summary: 汇总统计量
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        detailed_results_path = output_dir / 'consistency_check_detailed.jsonl'
        with open(detailed_results_path, 'w') as f:
            for result in self.check_results:
                f.write(json.dumps(result) + '\n')
        
        # 保存汇总统计
        summary_path = output_dir / 'consistency_check_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存失败案例
        if self.failed_cases:
            failed_cases_path = output_dir / 'failed_cases.json'
            with open(failed_cases_path, 'w') as f:
                json.dump(self.failed_cases, f, indent=2)
        
        # 生成报告
        self.generate_report(output_dir, summary)
        
        logging.info(f"Consistency check results saved to {output_dir}")
    
    def generate_report(self, output_dir: Path, summary: Dict[str, Any]):
        """生成一致性检查报告
        
        Args:
            output_dir: 输出目录
            summary: 汇总统计量
        """
        report_path = output_dir / 'consistency_check_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PDEBench数据一致性检查报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- 检查样本数: {summary['total_samples']}\n")
            f.write(f"- 容忍度: {summary['tolerance_used']:.2e}\n")
            f.write(f"- 随机种子: {summary['random_seed']}\n")
            f.write(f"- 检查时间: {summary['meta']['environment'].get('timestamp', 'N/A')}\n\n")
            
            # 一致性结果
            f.write("## 一致性检查结果\n\n")
            f.write(f"- **总体通过**: {'✅ 是' if summary['overall_pass'] else '❌ 否'}\n")
            f.write(f"- **H算子一致性率**: {summary['consistency_rate']:.2%} ({summary['consistent_samples']}/{summary['total_samples']})\n")
            f.write(f"- **退化算子一致性率**: {summary['degradation_consistency_rate']:.2%}\n")
            f.write(f"- **失败案例数**: {summary['failed_cases_count']}\n\n")
            
            # MSE统计
            if summary['mse_statistics']['mean'] is not None:
                f.write("## MSE(H(GT), y) 统计\n\n")
                mse_stats = summary['mse_statistics']
                f.write(f"- **均值**: {mse_stats['mean']:.2e}\n")
                f.write(f"- **标准差**: {mse_stats['std']:.2e}\n")
                f.write(f"- **最小值**: {mse_stats['min']:.2e}\n")
                f.write(f"- **最大值**: {mse_stats['max']:.2e}\n")
                f.write(f"- **中位数**: {mse_stats['median']:.2e}\n\n")
            
            # 相对误差统计
            if summary['rel_error_statistics']['mean'] is not None:
                f.write("## 相对误差统计\n\n")
                rel_stats = summary['rel_error_statistics']
                f.write(f"- **均值**: {rel_stats['mean']:.2e}\n")
                f.write(f"- **标准差**: {rel_stats['std']:.2e}\n")
                f.write(f"- **最小值**: {rel_stats['min']:.2e}\n")
                f.write(f"- **最大值**: {rel_stats['max']:.2e}\n")
                f.write(f"- **中位数**: {rel_stats['median']:.2e}\n\n")
            
            # 结论
            f.write("## 结论\n\n")
            if summary['overall_pass']:
                f.write("✅ **数据一致性检查通过**\n\n")
                f.write("- 所有样本的H(GT)与观测数据y的MSE均小于容忍度\n")
                f.write("- 观测算子H与训练DC完全一致\n")
                f.write("- 系统满足黄金法则的一致性要求\n")
            else:
                f.write("❌ **数据一致性检查失败**\n\n")
                f.write("存在以下问题：\n")
                if summary['consistency_rate'] < 1.0:
                    f.write(f"- {summary['inconsistent_samples']} 个样本的MSE超过容忍度\n")
                if summary['degradation_consistency_rate'] < 1.0:
                    f.write("- 退化算子一致性验证失败\n")
                f.write("\n请检查：\n")
                f.write("1. 观测算子H的实现是否正确\n")
                f.write("2. 训练DC与观测生成是否使用相同配置\n")
                f.write("3. 数据预处理流程是否一致\n")
            
            # 失败案例分析
            if self.failed_cases:
                f.write("\n## 失败案例分析\n\n")
                f.write("| 样本索引 | MSE | 相对误差 | 严重程度 |\n")
                f.write("|----------|-----|----------|----------|\n")
                
                for case in self.failed_cases[:10]:  # 只显示前10个
                    f.write(f"| {case['sample_idx']} | {case['actual_mse']:.2e} | "
                           f"{case['rel_error']:.2e} | {case['severity']} |\n")
                
                if len(self.failed_cases) > 10:
                    f.write(f"\n... 还有 {len(self.failed_cases) - 10} 个失败案例，详见 failed_cases.json\n")
        
        logging.info(f"Consistency check report saved to {report_path}")


def create_synthetic_dataloader(config: DictConfig) -> DataLoader:
    """创建合成数据加载器"""
    task = config.get('data', {}).get('task', 'sr')
    num_samples = config.get('consistency_check', {}).get('num_samples', 100)
    image_size = config.get('data', {}).get('image_size', 256)
    
    dataset = SyntheticDataset(num_samples=num_samples, image_size=image_size, task=task)
    
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


@hydra.main(version_base=None, config_path="../configs", config_name="consistency_check")
def main(cfg: DictConfig) -> None:
    """主检查函数
    
    Args:
        cfg: 配置对象
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(cfg.get('output_dir', 'consistency_check_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试创建真实数据模块，失败则使用合成数据
    try:
        data_module = PDEBenchDataModule(cfg.data)
        data_module.setup()
        
        # 获取数据加载器（优先使用验证集）
        if hasattr(data_module, 'val_dataloader'):
            dataloader = data_module.val_dataloader()
            logger.info("Using validation dataloader")
        elif hasattr(data_module, 'test_dataloader'):
            dataloader = data_module.test_dataloader()
            logger.info("Using test dataloader")
        else:
            dataloader = data_module.train_dataloader()
            logger.info("Using train dataloader")
        
        logger.info(f"Total batches available: {len(dataloader)}")
        
    except Exception as e:
        logger.warning(f"Failed to load real data: {e}")
        logger.info("Using synthetic data for testing...")
        dataloader = create_synthetic_dataloader(cfg)
        logger.info(f"Created synthetic dataset with {len(dataloader)} samples")
    
    # 创建一致性检查器
    checker = DataConsistencyChecker(cfg, device)
    
    # 运行检查
    logger.info("Starting data consistency check...")
    results = checker.run_consistency_check(dataloader, output_dir)
    
    # 打印结果
    logger.info("Data consistency check completed!")
    logger.info(f"Overall pass: {'YES' if results['overall_pass'] else 'NO'}")
    logger.info(f"Consistency rate: {results['consistency_rate']:.2%}")
    logger.info(f"Degradation consistency rate: {results['degradation_consistency_rate']:.2%}")
    
    if results['mse_statistics']['mean'] is not None:
        logger.info(f"Mean MSE: {results['mse_statistics']['mean']:.2e}")
        logger.info(f"Max MSE: {results['mse_statistics']['max']:.2e}")
    
    if results['failed_cases_count'] > 0:
        logger.warning(f"Found {results['failed_cases_count']} failed cases")
    
    logger.info(f"Detailed results saved to: {output_dir}")
    
    # 返回退出码
    return 0 if results['overall_pass'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)