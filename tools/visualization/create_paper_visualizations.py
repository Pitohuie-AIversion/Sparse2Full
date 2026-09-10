"""论文级可视化生成脚本

生成符合论文发表标准的可视化图表，包括：
- GT/Pred/Error热图（统一色标）
- 功率谱图（log尺度）
- 边界带局部放大分析
- 失败案例分析和改进建议
- 频域分析图表
- 空间域分析图表

严格按照技术架构文档要求实现标准化可视化。

使用方法：
python tools/create_paper_visualizations.py --config configs/config.yaml --checkpoint runs/exp/checkpoints/best.pth
python tools/create_paper_visualizations.py --results_dir runs/ --output_dir paper_package/figs/
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 使用统一的可视化工具，不直接导入matplotlib
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import PDEBenchDataModule
from models import create_model
from utils.checkpoint import load_checkpoint
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
from utils.visualization import PDEBenchVisualizer
from ops.degradation import apply_degradation_operator

# 设置绘图风格
# 使用统一的可视化工具，不直接设置matplotlib样式


class PaperVisualizationGenerator:
    """论文级可视化生成器
    
    生成符合论文发表标准的可视化图表
    """
    
    def __init__(self, config: DictConfig, output_dir: str = "./paper_package/figs"):
        self.config = config
        self.device = torch.device(config.get('experiment', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger('visualization', self.output_dir / 'visualization.log')
        
        # 初始化统一可视化工具
        self.visualizer = PDEBenchVisualizer(
            output_dir=str(self.output_dir),
            logger=self.logger
        )
        
        # 初始化数据模块
        self.data_module = PDEBenchDataModule(config.data)
        self.data_module.setup()
        
        # 初始化指标计算器
        image_size = config.data.get('image_size', 256)
        boundary_width = config.get('evaluation', {}).get('boundary_width', 16)
        self.metrics_calculator = MetricsCalculator(
            image_size=(image_size, image_size),
            boundary_width=boundary_width
        )
        
        # 初始化观测算子配置
        self.degradation_config = config.data.get('degradation', {})
        
        # 失败案例分析配置
        self.failure_analysis = {
            'error_threshold_high': 0.1,    # 高误差阈值
            'error_threshold_medium': 0.05, # 中等误差阈值
            'boundary_analysis_width': 32,  # 边界分析宽度
            'spectral_analysis_bins': 50    # 频谱分析bins
        }
    
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """加载模型"""
        # 创建一个简单的模型配置
        model_config = DictConfig({
            'model': {
                'name': 'SwinUNet',
                'params': {
                    'in_channels': 3,
                     'out_channels': 3,
                    'img_size': 256,
                    'kwargs': {
                        'patch_size': 4,
                        'window_size': 8,
                        'depths': [2, 2, 6, 2],
                        'num_heads': [3, 6, 12, 24],
                        'embed_dim': 96
                    }
                }
            }
        })
        
        model = create_model(model_config)
        model = model.to(self.device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model.eval()
        return model
    
    def generate_standard_visualizations(self, model: nn.Module, num_samples: int = 10) -> None:
        """生成标准可视化图表"""
        self.logger.info("Generating standard visualizations...")
        
        # 获取测试数据
        test_loader = self.data_module.test_dataloader()
        
        # 收集样本
        samples = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(samples) >= num_samples:
                    break
                
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 模型推理
                baseline = batch['baseline']
                # 确保输入通道数正确
                if baseline.shape[1] == 1:
                    # 如果输入是单通道，复制到3通道
                    baseline = baseline.repeat(1, 3, 1, 1)
                prediction = model(baseline)
                # 如果预测是多通道，只取第一个通道
                if prediction.shape[1] > 1:
                    prediction = prediction[:, :1, :, :]
                
                # 收集样本
                for j in range(baseline.shape[0]):
                    if len(samples) >= num_samples:
                        break
                    
                    sample = {
                        'baseline': baseline[j].cpu().numpy(),
                        'ground_truth': batch.get('ground_truth', batch.get('u', batch.get('target', baseline)))[j].cpu().numpy(),
                        'prediction': prediction[j].cpu().numpy(),
                        'coords': batch['coords'][j].cpu().numpy() if 'coords' in batch else None,
                        'mask': batch['mask'][j].cpu().numpy() if 'mask' in batch else None,
                        'sample_id': f"sample_{len(samples):03d}"
                    }
                    samples.append(sample)
        
        # 为每个样本生成可视化
        for i, sample in enumerate(tqdm(samples, desc="Creating visualizations")):
            self._create_sample_visualization(sample, i)
        
        # 生成汇总可视化
        self._create_summary_visualizations(samples)
        
        self.logger.info(f"Standard visualizations saved to {self.output_dir}")
    
    def _create_sample_visualization(self, sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """为单个样本创建可视化"""
        sample_id = sample['sample_id']
        
        # 准备张量数据
        gt_tensor = torch.from_numpy(sample['ground_truth']).unsqueeze(0)
        pred_tensor = torch.from_numpy(sample['prediction']).unsqueeze(0)
        
        # 如果有观测数据，也准备张量
        if sample.get('baseline') is not None:
            obs_tensor = torch.from_numpy(sample['baseline']).unsqueeze(0)
        else:
            obs_tensor = None
        
        # 1. 四联图可视化（观测/GT/预测/误差）
        if obs_tensor is not None:
            self.visualizer.create_quadruplet_visualization(
                obs_tensor, gt_tensor, pred_tensor, 
                save_name=f"quadruplet_{sample_id}"
            )
        else:
            # 三联图可视化（GT/预测/误差）
            self.visualizer.create_field_comparison(
                gt_tensor, pred_tensor, 
                save_name=f"comparison_{sample_id}"
            )
        
        # 2. 功率谱对比图
        self.visualizer.plot_power_spectrum_comparison(
            gt_tensor, pred_tensor, 
            save_name=f"power_spectrum_{sample_id}"
        )
        
        # 3. 边界带分析图
        self.visualizer.plot_boundary_analysis(
            gt_tensor, pred_tensor, 
            save_name=f"boundary_{sample_id}",
            boundary_width=self.failure_analysis['boundary_analysis_width']
        )
    
    def _create_triple_heatmap(self, sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """创建GT/Pred/Error三联热图"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 处理多通道情况（取第一个通道或平均）
        if len(gt.shape) == 3:
            gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
            pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
        else:
            gt_vis, pred_vis = gt, pred
        
        # 转换为张量格式
        gt_tensor = torch.from_numpy(gt_vis).float().unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_vis).float().unsqueeze(0).unsqueeze(0)
        
        # 使用统一的可视化接口
        self.visualizer.plot_field_comparison(
            gt_tensor, pred_tensor,
            save_name=f'triple_heatmap_{sample_idx:03d}',
            title=f'Sample {sample_idx} - Field Comparison'
        )
    
    def _create_power_spectrum_plot(self, sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """创建功率谱对比图"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 处理多通道情况
        if len(gt.shape) == 3:
            gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
            pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
        else:
            gt_vis, pred_vis = gt, pred
        
        # 转换为张量格式
        gt_tensor = torch.from_numpy(gt_vis).float().unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_vis).float().unsqueeze(0).unsqueeze(0)
        
        # 使用统一的可视化接口
        self.visualizer.plot_power_spectrum_comparison(
            gt_tensor, pred_tensor,
            save_name=f'power_spectrum_{sample_idx:03d}'
        )
    
    def _create_boundary_analysis(self, sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """创建边界带分析图"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 处理多通道情况
        if len(gt.shape) == 3:
            gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
            pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
        else:
            gt_vis, pred_vis = gt, pred
        
        # 转换为张量格式
        gt_tensor = torch.from_numpy(gt_vis).float().unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_vis).float().unsqueeze(0).unsqueeze(0)
        
        # 使用统一的可视化接口
        self.visualizer.plot_boundary_analysis(
            gt_tensor, pred_tensor,
            save_name=f'boundary_analysis_{sample_idx:03d}',
            boundary_width=self.failure_analysis['boundary_analysis_width']
        )
    
    def _create_frequency_analysis(self, sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """创建频域分析图"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 处理多通道情况
        if len(gt.shape) == 3:
            gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
            pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
        else:
            gt_vis, pred_vis = gt, pred
        
        # 计算频域误差
        gt_fft = np.fft.fft2(gt_vis)
        pred_fft = np.fft.fft2(pred_vis)
        
        # 频域分段分析（按Nyquist频率的1/4和1/2分段）
        h, w = gt_vis.shape
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        nyquist = min(h, w) // 2
        low_freq_mask = r <= nyquist // 4
        mid_freq_mask = (r > nyquist // 4) & (r <= nyquist // 2)
        high_freq_mask = r > nyquist // 2
        
        # 转换为张量格式
        gt_tensor = torch.from_numpy(gt_vis).float().unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_vis).float().unsqueeze(0).unsqueeze(0)
        
        # 使用统一的可视化接口
        self.visualizer.plot_frequency_analysis(
            gt_tensor, pred_tensor,
            save_name=f'frequency_analysis_{sample_idx:03d}'
        )
    
    def _assess_error_level(self, sample: Dict[str, np.ndarray]) -> str:
        """评估样本的误差水平"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 计算相对L2误差
        rel_l2 = np.linalg.norm(gt - pred) / np.linalg.norm(gt)
        
        if rel_l2 > self.failure_analysis['error_threshold_high']:
            return 'high'
        elif rel_l2 > self.failure_analysis['error_threshold_medium']:
            return 'medium'
        else:
            return 'low'
    
    def _create_failure_analysis(self, sample: Dict[str, np.ndarray], sample_idx: int, error_level: str) -> None:
        """创建失败案例分析"""
        gt = sample['ground_truth']
        pred = sample['prediction']
        
        # 处理多通道情况
        if len(gt.shape) == 3:
            gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
            pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
        else:
            gt_vis, pred_vis = gt, pred
        
        error = np.abs(gt_vis - pred_vis)
        rel_error = error / (np.abs(gt_vis) + 1e-8)
        
        # 失败类型分析
        failure_types = self._analyze_failure_types(gt_vis, pred_vis, error)
        
        # 转换为PyTorch张量用于统一可视化接口
        gt_tensor = torch.from_numpy(gt_vis).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        pred_tensor = torch.from_numpy(pred_vis).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 使用统一的可视化接口
        self.visualizer.create_failure_case_analysis(
            gt_tensor, pred_tensor, 
            sample.get('metrics', {}),
            save_name=f'failure_case_{sample_idx}',
            failure_type=', '.join(failure_types)
        )
    
    def _analyze_failure_types(self, gt: np.ndarray, pred: np.ndarray, error: np.ndarray) -> Dict[str, Any]:
        """分析失败类型"""
        failure_types = {}
        
        # 1. 边界层溢出检测
        h, w = gt.shape
        boundary_width = 16
        boundary_mask = np.zeros_like(gt, dtype=bool)
        boundary_mask[:boundary_width, :] = True
        boundary_mask[-boundary_width:, :] = True
        boundary_mask[:, :boundary_width] = True
        boundary_mask[:, -boundary_width:] = True
        
        boundary_error = error[boundary_mask].mean()
        center_error = error[~boundary_mask].mean()
        
        failure_types['boundary_overflow'] = {
            'detected': boundary_error > 2 * center_error,
            'boundary_error': boundary_error,
            'center_error': center_error,
            'ratio': boundary_error / center_error if center_error > 0 else float('inf')
        }
        
        # 2. 相位漂移检测
        gt_fft = np.fft.fft2(gt)
        pred_fft = np.fft.fft2(pred)
        
        gt_phase = np.angle(gt_fft)
        pred_phase = np.angle(pred_fft)
        phase_diff = np.abs(gt_phase - pred_phase)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
        
        failure_types['phase_drift'] = {
            'detected': np.mean(phase_diff) > 0.5,
            'mean_phase_error': np.mean(phase_diff),
            'max_phase_error': np.max(phase_diff)
        }
        
        # 3. 振铃效应检测
        # 检测高频振荡
        laplacian = cv2.Laplacian(pred.astype(np.float32), cv2.CV_32F)
        gt_laplacian = cv2.Laplacian(gt.astype(np.float32), cv2.CV_32F)
        
        pred_oscillation = np.std(laplacian)
        gt_oscillation = np.std(gt_laplacian)
        
        failure_types['ringing'] = {
            'detected': pred_oscillation > 2 * gt_oscillation,
            'pred_oscillation': pred_oscillation,
            'gt_oscillation': gt_oscillation,
            'ratio': pred_oscillation / gt_oscillation if gt_oscillation > 0 else float('inf')
        }
        
        # 4. 能量偏差检测
        gt_energy = np.sum(gt**2)
        pred_energy = np.sum(pred**2)
        energy_ratio = pred_energy / gt_energy if gt_energy > 0 else float('inf')
        
        failure_types['energy_bias'] = {
            'detected': abs(energy_ratio - 1.0) > 0.2,
            'energy_ratio': energy_ratio,
            'gt_energy': gt_energy,
            'pred_energy': pred_energy
        }
        
        # 5. 局部特征丢失检测
        # 使用结构相似性检测
        from skimage.metrics import structural_similarity as ssim
        
        # 分块计算SSIM
        block_size = 32
        ssim_map = np.zeros((h // block_size, w // block_size))
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                gt_block = gt[i:i+block_size, j:j+block_size]
                pred_block = pred[i:i+block_size, j:j+block_size]
                
                if gt_block.std() > 1e-6:  # 避免常数块
                    ssim_val = ssim(gt_block, pred_block, data_range=gt_block.max()-gt_block.min())
                    ssim_map[i//block_size, j//block_size] = ssim_val
        
        low_ssim_ratio = np.sum(ssim_map < 0.7) / ssim_map.size
        
        failure_types['feature_loss'] = {
            'detected': low_ssim_ratio > 0.3,
            'low_ssim_ratio': low_ssim_ratio,
            'mean_ssim': np.mean(ssim_map),
            'min_ssim': np.min(ssim_map)
        }
        
        return failure_types
    
    def _generate_failure_analysis_text(self, failure_types: Dict[str, Any], error_level: str) -> str:
        """生成失败分析文本"""
        text = f"FAILURE ANALYSIS REPORT (Error Level: {error_level.upper()})\n"
        text += "=" * 60 + "\n\n"
        
        detected_failures = []
        
        # 检查各种失败类型
        if failure_types['boundary_overflow']['detected']:
            detected_failures.append("边界层溢出 (Boundary Overflow)")
            text += f"• 边界层溢出: 边界误差 ({failure_types['boundary_overflow']['boundary_error']:.4f}) "
            text += f"是中心误差 ({failure_types['boundary_overflow']['center_error']:.4f}) 的 "
            text += f"{failure_types['boundary_overflow']['ratio']:.1f} 倍\n"
        
        if failure_types['phase_drift']['detected']:
            detected_failures.append("相位漂移 (Phase Drift)")
            text += f"• 相位漂移: 平均相位误差 {failure_types['phase_drift']['mean_phase_error']:.3f} rad, "
            text += f"最大相位误差 {failure_types['phase_drift']['max_phase_error']:.3f} rad\n"
        
        if failure_types['ringing']['detected']:
            detected_failures.append("振铃效应 (Ringing)")
            text += f"• 振铃效应: 预测振荡强度 ({failure_types['ringing']['pred_oscillation']:.4f}) "
            text += f"是真值的 {failure_types['ringing']['ratio']:.1f} 倍\n"
        
        if failure_types['energy_bias']['detected']:
            detected_failures.append("能量偏差 (Energy Bias)")
            text += f"• 能量偏差: 能量比率 {failure_types['energy_bias']['energy_ratio']:.3f} "
            text += f"(偏差 {abs(failure_types['energy_bias']['energy_ratio']-1)*100:.1f}%)\n"
        
        if failure_types['feature_loss']['detected']:
            detected_failures.append("局部特征丢失 (Feature Loss)")
            text += f"• 特征丢失: {failure_types['feature_loss']['low_ssim_ratio']*100:.1f}% 区域SSIM < 0.7, "
            text += f"平均SSIM {failure_types['feature_loss']['mean_ssim']:.3f}\n"
        
        text += "\n"
        
        # 改进建议
        text += "IMPROVEMENT SUGGESTIONS:\n"
        text += "-" * 30 + "\n"
        
        if not detected_failures:
            text += "• 整体性能良好，可考虑进一步优化超参数\n"
        else:
            if "边界层溢出" in detected_failures:
                text += "• 边界处理: 增强边界条件约束，使用镜像填充或周期边界\n"
            
            if "相位漂移" in detected_failures:
                text += "• 相位保持: 添加频域损失，使用复数网络或相位约束\n"
            
            if "振铃效应" in detected_failures:
                text += "• 振铃抑制: 降低学习率，增加正则化，使用平滑损失\n"
            
            if "能量偏差" in detected_failures:
                text += "• 能量守恒: 添加能量守恒损失，使用归一化层\n"
            
            if "局部特征丢失" in detected_failures:
                text += "• 特征保持: 增加感受野，使用多尺度损失，提高模型容量\n"
        
        text += "\n"
        text += "RECOMMENDED ACTIONS:\n"
        text += "-" * 20 + "\n"
        text += "1. 调整损失函数权重 (λ_s, λ_dc)\n"
        text += "2. 优化网络架构 (深度、宽度、跳跃连接)\n"
        text += "3. 改进数据增强策略\n"
        text += "4. 调整训练超参数 (学习率、批大小)\n"
        
        return text
    
    def _create_summary_visualizations(self, samples: List[Dict[str, np.ndarray]]) -> None:
        """创建汇总可视化"""
        self.logger.info("Creating summary visualizations...")
        
        # 收集失败案例
        failure_cases = []
        for i, sample in enumerate(samples):
            error_level = self._assess_error_level(sample)
            if error_level in ['high', 'medium']:
                gt_tensor = torch.from_numpy(sample['ground_truth']).unsqueeze(0)
                pred_tensor = torch.from_numpy(sample['prediction']).unsqueeze(0)
                failure_cases.append({
                    'pred': pred_tensor,
                    'target': gt_tensor,
                    'error_type': error_level
                })
        
        # 使用统一接口创建失败案例分析
        if failure_cases:
            self.visualizer.create_failure_case_analysis(
                failure_cases, save_name="failure_cases_summary"
            )
        
        # 创建指标汇总（如果有多个模型的话）
        # 这里可以扩展为比较不同模型的性能
        metrics_data = {
            'Current Model': {
                'Rel-L2': self._calculate_average_rel_l2(samples),
                'MAE': self._calculate_average_mae(samples),
                'PSNR': self._calculate_average_psnr(samples)
            }
        }
        
        self.visualizer.create_metrics_summary_plot(
             metrics_data, save_name="metrics_summary"
         )
    
    def _calculate_average_rel_l2(self, samples: List[Dict[str, np.ndarray]]) -> float:
        """计算平均相对L2误差"""
        rel_l2_errors = []
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            rel_l2 = np.linalg.norm(gt - pred) / np.linalg.norm(gt)
            rel_l2_errors.append(rel_l2)
        return np.mean(rel_l2_errors)
    
    def _calculate_average_mae(self, samples: List[Dict[str, np.ndarray]]) -> float:
        """计算平均绝对误差"""
        mae_errors = []
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            mae = np.mean(np.abs(gt - pred))
            mae_errors.append(mae)
        return np.mean(mae_errors)
    
    def _calculate_average_psnr(self, samples: List[Dict[str, np.ndarray]]) -> float:
        """计算平均PSNR"""
        psnr_values = []
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            mse = np.mean((gt - pred) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                max_val = max(gt.max(), pred.max())
                psnr = 20 * np.log10(max_val / np.sqrt(mse))
            psnr_values.append(psnr)
        return np.mean(psnr_values)
    
    def _create_error_distribution_summary(self, samples: List[Dict[str, np.ndarray]]) -> None:
        """创建误差分布汇总图"""
        all_errors = []
        all_rel_errors = []
        
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            
            if len(gt.shape) == 3:
                gt = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
                pred = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
            
            error = np.abs(gt - pred)
            rel_error = error / (np.abs(gt) + 1e-8)
            
            all_errors.extend(error.flatten())
            all_rel_errors.extend(rel_error.flatten())
        
        # 使用统一的可视化接口
        self.visualizer.create_error_distribution_plot(
            all_errors, all_rel_errors,
            save_name='error_distribution_summary'
        )
    
    def _create_performance_radar_summary(self, samples: List[Dict[str, np.ndarray]]) -> None:
        """创建性能雷达图汇总"""
        # 计算各项指标
        metrics_list = []
        
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            
            # 转换为torch tensor进行指标计算
            gt_tensor = torch.from_numpy(gt).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred).unsqueeze(0)
            
            # 计算指标
            metrics = self.metrics_calculator.compute_all_metrics(pred_tensor, gt_tensor)
            metrics_list.append(metrics)
        
        # 聚合指标
        aggregated_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 创建雷达图
        metrics_to_plot = ['rel_l2', 'mae', 'psnr', 'ssim']
        values = []
        labels = []
        
        for metric in metrics_to_plot:
            if metric in aggregated_metrics:
                value = aggregated_metrics[metric]['mean']
                # 标准化到0-1范围（越大越好）
                if metric in ['rel_l2', 'mae']:
                    normalized_value = 1 / (1 + value)  # 误差指标取倒数
                else:
                    normalized_value = value / 100 if metric == 'psnr' else value
                
                values.append(normalized_value)
                labels.append(metric.upper())
        
        if values:
            # 使用统一的可视化接口
            self.visualizer.create_radar_chart(
                values, labels,
                save_name='performance_radar_summary',
                title='Overall Performance Radar Chart'
            )
    
    def _create_failure_cases_summary(self, samples: List[Dict[str, np.ndarray]]) -> None:
        """创建失败案例汇总"""
        failure_stats = {
            'boundary_overflow': 0,
            'phase_drift': 0,
            'ringing': 0,
            'energy_bias': 0,
            'feature_loss': 0
        }
        
        error_levels = {'high': 0, 'medium': 0, 'low': 0}
        
        # 分析所有样本
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            
            if len(gt.shape) == 3:
                gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
                pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
            else:
                gt_vis, pred_vis = gt, pred
            
            error = np.abs(gt_vis - pred_vis)
            
            # 评估误差水平
            error_level = self._assess_error_level(sample)
            error_levels[error_level] += 1
            
            # 分析失败类型
            failure_types = self._analyze_failure_types(gt_vis, pred_vis, error)
            
            for failure_type, analysis in failure_types.items():
                if analysis['detected']:
                    failure_stats[failure_type] += 1
        
        # 使用统一的可视化接口
        self.visualizer.create_failure_summary_plot(
            error_levels, failure_stats,
            save_name='failure_cases_summary'
        )
    
    def _create_frequency_performance_summary(self, samples: List[Dict[str, np.ndarray]]) -> None:
        """创建频域性能汇总"""
        low_freq_errors = []
        mid_freq_errors = []
        high_freq_errors = []
        
        for sample in samples:
            gt = sample['ground_truth']
            pred = sample['prediction']
            
            if len(gt.shape) == 3:
                gt_vis = gt[0] if gt.shape[0] == 1 else np.mean(gt, axis=0)
                pred_vis = pred[0] if pred.shape[0] == 1 else np.mean(pred, axis=0)
            else:
                gt_vis, pred_vis = gt, pred
            
            # 计算频域误差
            gt_fft = np.fft.fft2(gt_vis)
            pred_fft = np.fft.fft2(pred_vis)
            
            h, w = gt_vis.shape
            center = (h // 2, w // 2)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            nyquist = min(h, w) // 2
            low_freq_mask = r <= nyquist // 4
            mid_freq_mask = (r > nyquist // 4) & (r <= nyquist // 2)
            high_freq_mask = r > nyquist // 2
            
            gt_fft_shift = np.fft.fftshift(gt_fft)
            pred_fft_shift = np.fft.fftshift(pred_fft)
            freq_error = np.abs(gt_fft_shift - pred_fft_shift)
            
            low_freq_errors.append(freq_error[low_freq_mask].mean())
            mid_freq_errors.append(freq_error[mid_freq_mask].mean())
            high_freq_errors.append(freq_error[high_freq_mask].mean())
        
        # 使用统一的可视化接口
        freq_bands = ['Low\n(0-π/4)', 'Mid\n(π/4-π/2)', 'High\n(π/2-π)']
        freq_data = [low_freq_errors, mid_freq_errors, high_freq_errors]
        
        self.visualizer.create_frequency_performance_plot(
            freq_bands, freq_data,
            save_name='frequency_performance_summary'
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Paper Visualization Generator')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--results_dir', type=str,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./paper_package/figs',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建可视化生成器
    visualizer = PaperVisualizationGenerator(config, args.output_dir)
    
    try:
        # 加载模型
        if args.checkpoint:
            model = visualizer.load_model(args.checkpoint)
            
            # 生成可视化
            visualizer.generate_standard_visualizations(model, args.num_samples)
            
            print(f"Visualizations generated successfully!")
            print(f"Output directory: {args.output_dir}")
            
        else:
            print("No checkpoint specified. Please provide --checkpoint argument.")
            
    except Exception as e:
        logging.error(f"Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main()