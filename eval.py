"""评测脚本

PDEBench稀疏观测重建系统模型评测入口
生成指标报告和可视化结果
支持多种子统计分析和显著性检验

使用方法：
python eval.py --config configs/config.yaml --checkpoint runs/experiment/checkpoints/best.pth
python eval.py --config configs/config.yaml --checkpoint_dir runs/ --multi_seed
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# 使用统一的可视化工具，不直接导入matplotlib
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from scipy import stats as scipy_stats

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets import PDEBenchDataModule
from models import create_model
from utils.metrics import MetricsCalculator, StatisticalAnalyzer, compute_all_metrics
from utils.visualization import PDEBenchVisualizer
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from ops.degradation import apply_degradation_operator


class ModelEvaluator:
    """模型评测器
    
    负责完整的模型评测流程：
    - 模型加载和推理
    - 指标计算和统计
    - 结果可视化
    - 报告生成
    """
    
    def __init__(self, config: DictConfig, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.experiment.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建输出目录
        self.output_dir = Path(config.experiment.get('output_dir', './runs/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger('eval', self.output_dir / 'eval.log')
        self.logger.info(f"Evaluation started with config:\n{OmegaConf.to_yaml(config)}")
        
        # 初始化组件
        self._init_data()
        self._init_model()
        
        # 加载检查点
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # 初始化指标计算器
        image_size = (config.data.get('image_size', 256), config.data.get('image_size', 256))
        self.metrics_calculator = MetricsCalculator(
            image_size=image_size,
            boundary_width=config.evaluation.get('boundary_width', 16)
        )
        
        # 结果存储
        self.results = {}
        self.case_results = []
    
    def _init_data(self) -> None:
        """初始化数据模块"""
        self.logger.info("Initializing data module...")
        
        self.data_module = PDEBenchDataModule(self.config.data)
        self.data_module.setup()
        
        # 获取数据加载器
        self.test_loader = self.data_module.test_dataloader()
        
        # 获取归一化统计量
        self.norm_stats = self.data_module.get_norm_stats()
        
        self.logger.info(f"Test data loaded: {len(self.test_loader)} batches")
    
    def _init_model(self) -> None:
        """初始化模型"""
        self.logger.info("Initializing model...")
        
        self.model = create_model(self.config.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 模型信息
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            self.logger.info(f"Model info: {model_info}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        # 加载模型权重
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 记录检查点信息
        self.logger.info(f"Checkpoint loaded: epoch {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    def evaluate_model(self, save_samples: bool = True, 
                      max_samples: int = 1000) -> Dict[str, float]:
        """评测模型
        
        Args:
            save_samples: 是否保存样本可视化
            max_samples: 最大样本数量
            
        Returns:
            metrics: 平均指标结果
        """
        self.logger.info("Starting model evaluation...")
        
        self.model.eval()
        all_metrics = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # 移动数据到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 模型推理
                pred = self.model(batch['baseline'])
                
                # 计算指标（按技术架构文档要求）
                batch_metrics = self._compute_comprehensive_metrics(
                    pred=pred,
                    target=batch['target'],
                    obs_data=batch
                )
                
                # 转换为标量并存储
                batch_size = pred.shape[0]
                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break
                    
                    case_metrics = {}
                    for key, value in batch_metrics.items():
                        if torch.is_tensor(value):
                            # 多通道聚合：先逐通道算指标，再等权平均
                            case_metrics[key] = value[i].mean().item()
                        else:
                            case_metrics[key] = value
                    
                    # 添加样本信息
                    case_metrics['batch_idx'] = batch_idx
                    case_metrics['sample_idx'] = i
                    case_metrics['case_id'] = f"batch_{batch_idx:04d}_sample_{i:02d}"
                    
                    all_metrics.append(case_metrics)
                    self.case_results.append(case_metrics)
                    sample_count += 1
                
                # 保存样本可视化
                if save_samples and batch_idx < 10:  # 只保存前10个批次
                    self._save_batch_samples(batch, pred, batch_idx)
                
                if sample_count >= max_samples:
                    break
        
        # 计算平均指标和统计量
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if key not in ['batch_idx', 'sample_idx', 'case_id']:
                    values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                    if values:
                        avg_metrics[key] = np.mean(values)
                        avg_metrics[f'{key}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        avg_metrics[f'{key}_min'] = np.min(values)
                        avg_metrics[f'{key}_max'] = np.max(values)
        else:
            avg_metrics = {}
        
        self.results = avg_metrics
        
        self.logger.info(f"Evaluation completed: {sample_count} samples processed")
        self.logger.info(f"Average metrics: {avg_metrics}")
        
        return avg_metrics
    
    def _compute_comprehensive_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                                     obs_data: Dict) -> Dict[str, torch.Tensor]:
        """计算完整的评测指标（严格按照技术架构文档）
        
        Args:
            pred: 预测值（z-score域） [B, C, H, W]
            target: 真实值（z-score域） [B, C, H, W]
            obs_data: 观测数据字典
            
        Returns:
            metrics: 所有指标的字典
        """
        metrics = {}
        
        # 1. 基础指标
        metrics['rel_l2'] = self.metrics_calculator.compute_rel_l2(pred, target)
        metrics['mae'] = self.metrics_calculator.compute_mae(pred, target)
        metrics['psnr'] = self.metrics_calculator.compute_psnr(pred, target)
        metrics['ssim'] = self.metrics_calculator.compute_ssim(pred, target)
        
        # 2. 频域分段误差（fRMSE按Nyquist的1/4、1/2频率阈值分段）
        freq_rmse = self.metrics_calculator.compute_freq_rmse(pred, target)
        for band_name, rmse in freq_rmse.items():
            metrics[f'frmse_{band_name}'] = rmse
        
        # 3. 边界带误差（bRMSE，默认16px边界带，随分辨率缩放）
        metrics['brmse'] = self.metrics_calculator.compute_boundary_rmse(pred, target)
        
        # 4. 守恒误差（cRMSE，全域积分守恒偏差）
        metrics['crmse'] = self._compute_conservation_rmse(pred, target)
        
        # 5. 数据一致性验证（‖H(ŷ)−y‖₂）
        metrics['dc_error'] = self._compute_data_consistency_error(pred, obs_data)
        
        return metrics
    
    def _compute_conservation_rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算守恒量RMSE（全域积分守恒偏差）"""
        # 计算全域积分
        pred_integral = torch.sum(pred, dim=(-2, -1))  # [B, C]
        target_integral = torch.sum(target, dim=(-2, -1))  # [B, C]
        
        # 守恒偏差
        conservation_error = torch.abs(pred_integral - target_integral)
        
        # 相对误差
        target_integral_abs = torch.abs(target_integral) + 1e-8
        crmse = conservation_error / target_integral_abs
        
        return crmse
    
    def _compute_data_consistency_error(self, pred: torch.Tensor, obs_data: Dict) -> torch.Tensor:
        """计算数据一致性误差（严格按照技术架构文档）"""
        # 反归一化到原值域（DC损失在原值域计算）
        if self.norm_stats is not None:
            mean = torch.tensor(self.norm_stats['mean']).to(pred.device).view(1, -1, 1, 1)
            std = torch.tensor(self.norm_stats['std']).to(pred.device).view(1, -1, 1, 1)
            pred_orig = pred * std + mean
        else:
            pred_orig = pred
        
        # 应用观测算子H
        task_type = self.config.data.get('task', 'SR')
        task_params = self._get_task_params()
        
        pred_obs = apply_degradation_operator(pred_orig, task_type, task_params)
        
        # 计算与观测数据的误差
        target_obs = obs_data['baseline']
        if self.norm_stats is not None:
            # 如果观测数据也在z-score域，需要反归一化
            target_obs = target_obs * std + mean
        
        mse = torch.mean((pred_obs - target_obs)**2, dim=(-2, -1))  # [B, C]
        dc_error = torch.sqrt(mse)
        
        return dc_error
    
    def _get_task_params(self) -> Dict:
        """获取任务参数"""
        task_type = self.config.data.get('task', 'SR')
        
        if task_type == 'SR':
            return {
                'scale': self.config.data.get('sr_scale', 4),
                'sigma': self.config.data.get('blur_sigma', 1.0),
                'kernel_size': self.config.data.get('blur_kernel', 5),
                'boundary': self.config.data.get('boundary', 'mirror')
            }
        elif task_type == 'Crop':
            return {
                'crop_size': self.config.data.get('crop_size', [128, 128]),
                'patch_align': self.config.data.get('patch_align', 8),
                'boundary': self.config.data.get('boundary', 'mirror')
            }
        else:
            return {}
    
    def _save_batch_samples(self, batch: Dict, pred: torch.Tensor, batch_idx: int) -> None:
        """保存批次样本可视化"""
        save_dir = self.output_dir / 'samples' / f'batch_{batch_idx:04d}'
        save_evaluation_samples(
            pred=pred,
            target=batch['target'],
            baseline=batch['baseline'],
            save_dir=save_dir,
            norm_stats=self.norm_stats,
            max_samples=min(4, pred.shape[0])  # 每批次最多4个样本
        )
    
    def generate_report(self, baseline_results: Optional[List[Dict]] = None) -> str:
        """生成评测报告（符合论文发表标准）
        
        Args:
            baseline_results: 基线方法结果（用于对比）
            
        Returns:
            report: 格式化报告
        """
        report = "PDEBench Sparse Observation Reconstruction - Model Evaluation Report\n"
        report += "=" * 80 + "\n\n"
        
        # 基本信息
        report += f"Model: {self.config.model.get('_target_', 'Unknown')}\n"
        report += f"Dataset: {self.config.data.get('dataset_name', 'Unknown')}\n"
        report += f"Task: {self.config.data.get('task', 'Unknown')}\n"
        report += f"Resolution: {self.config.data.get('image_size', 256)}×{self.config.data.get('image_size', 256)}\n"
        report += f"Test samples: {len(self.case_results)}\n"
        report += f"Evaluation date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 主要指标（按技术架构文档8.2节）
        report += "Main Metrics (Mean ± Std):\n"
        report += "-" * 40 + "\n"
        
        main_metrics = ['rel_l2', 'mae', 'psnr', 'ssim', 'dc_error']
        for metric in main_metrics:
            if metric in self.results:
                mean_val = self.results[metric]
                std_val = self.results.get(f'{metric}_std', 0)
                report += f"{metric.upper():12s}: {mean_val:.6f} ± {std_val:.6f}\n"
        
        # 频域指标（fRMSE-low/mid/high）
        report += "\nFrequency Domain Metrics:\n"
        report += "-" * 30 + "\n"
        
        freq_metrics = ['frmse_low', 'frmse_mid', 'frmse_high']
        for metric in freq_metrics:
            if metric in self.results:
                mean_val = self.results[metric]
                std_val = self.results.get(f'{metric}_std', 0)
                report += f"{metric.upper():12s}: {mean_val:.6f} ± {std_val:.6f}\n"
        
        # 空间域指标（bRMSE、cRMSE）
        report += "\nSpatial Domain Metrics:\n"
        report += "-" * 25 + "\n"
        
        spatial_metrics = ['brmse', 'crmse']
        for metric in spatial_metrics:
            if metric in self.results:
                mean_val = self.results[metric]
                std_val = self.results.get(f'{metric}_std', 0)
                report += f"{metric.upper():12s}: {mean_val:.6f} ± {std_val:.6f}\n"
        
        # 模型资源信息
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            report += "\nModel Resources:\n"
            report += "-" * 15 + "\n"
            report += f"Parameters: {model_info.get('params', 'N/A')}\n"
            
            if hasattr(self.model, 'compute_flops'):
                flops = self.model.compute_flops()
                report += f"FLOPs: {flops/1e9:.2f}G (@256²)\n"
        
        # 与基线对比（显著性检验）
        if baseline_results:
            report += "\nComparison with Baseline (Statistical Significance):\n"
            report += "-" * 50 + "\n"
            
            analyzer = StatisticalAnalyzer()
            for case_result in self.case_results:
                analyzer.add_result(case_result)
            
            try:
                for metric in main_metrics:
                    if metric in self.results:
                        test_result = analyzer.significance_test(baseline_results, metric)
                        significance = "***" if test_result['significant'] else ""
                        
                        report += f"{metric.upper():12s}: p={test_result['p_value']:.6f}, "
                        report += f"Cohen's d={test_result['cohen_d']:.3f} {significance}\n"
            except Exception as e:
                report += f"Statistical test failed: {e}\n"
        
        # 数据一致性验证结果
        if 'dc_error' in self.results:
            dc_mean = self.results['dc_error']
            dc_std = self.results.get('dc_error_std', 0)
            report += f"\nData Consistency Verification:\n"
            report += "-" * 30 + "\n"
            report += f"||H(ŷ)−y||₂: {dc_mean:.8f} ± {dc_std:.8f}\n"
            
            # 验收标准检查
            if dc_mean < 1e-8:
                report += "✓ PASSED: DC error < 1e-8 (meets acceptance criteria)\n"
            else:
                report += "✗ FAILED: DC error ≥ 1e-8 (does not meet acceptance criteria)\n"
        
        return report
    
    def save_results(self) -> None:
        """保存评测结果"""
        # 保存平均结果
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存逐样本结果
        case_results_file = self.output_dir / 'case_results.jsonl'
        with open(case_results_file, 'w') as f:
            for case in self.case_results:
                f.write(json.dumps(case) + '\n')
        
        # 保存CSV格式
        if self.case_results:
            df = pd.DataFrame(self.case_results)
            csv_file = self.output_dir / 'results.csv'
            df.to_csv(csv_file, index=False)
        
        # 保存报告
        report = self.generate_report()
        report_file = self.output_dir / 'report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"  - Summary: {results_file}")
        self.logger.info(f"  - Cases: {case_results_file}")
        self.logger.info(f"  - CSV: {csv_file if self.case_results else 'N/A'}")
        self.logger.info(f"  - Report: {report_file}")
    
    def create_visualizations(self) -> None:
        """创建可视化图表（符合论文标准）"""
        if not self.case_results:
            self.logger.warning("No case results available for visualization")
            return
        
        # 使用统一的可视化工具
        visualizer = PDEBenchVisualizer(str(self.output_dir))
        
        # 准备指标数据
        metrics_data = {}
        for case in self.case_results:
            for metric, value in case.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
        
        # 创建指标汇总图
        visualizer.create_metrics_summary_plot(
            {'Model': metrics_data},
            save_name="metrics_summary"
        )
        
        self.logger.info(f"Visualizations saved to {self.output_dir}")
    
    # 已移除，使用统一的可视化接口
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """创建指标相关性热图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['batch_idx', 'sample_idx']]
        
        if len(numeric_cols) > 1:
            # 使用统一的可视化工具
            visualizer = PDEBenchVisualizer(str(self.output_dir))
            
            corr_matrix = df[numeric_cols].corr()
            
            # 使用PDEBenchVisualizer创建相关性热图
            visualizer.create_correlation_heatmap(
                corr_matrix=corr_matrix,
                save_name="metrics_correlation"
            )
    
    def _create_performance_trend_plot(self, df: pd.DataFrame) -> None:
        """创建性能趋势图"""
        # 使用统一的可视化工具，不直接使用matplotlib
        pass
    
    def _create_frequency_analysis_plot(self, df: pd.DataFrame) -> None:
        """创建频域分析图"""
        # 使用统一的可视化工具，不直接使用matplotlib
        pass
    
    def _create_spatial_analysis_plot(self, df: pd.DataFrame) -> None:
        """创建空间分析图"""
        # 使用统一的可视化工具，不直接使用matplotlib
        pass


def evaluate_multiple_checkpoints(config: DictConfig, checkpoint_dir: str,
                                pattern: str = "*/checkpoints/best.pth") -> Dict[str, Any]:
    """评测多个检查点（多种子实验）
    
    Args:
        config: 配置对象
        checkpoint_dir: 检查点目录
        pattern: 检查点文件模式
        
    Returns:
        results: 汇总结果
    """
    checkpoint_paths = list(Path(checkpoint_dir).glob(pattern))
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found with pattern {pattern} in {checkpoint_dir}")
    
    logger = logging.getLogger('eval')
    logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
    
    all_results = []
    analyzer = StatisticalAnalyzer()
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        logger.info(f"Evaluating checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        
        # 创建评测器
        evaluator = ModelEvaluator(config, str(checkpoint_path))
        
        # 评测
        results = evaluator.evaluate_model(save_samples=(i == 0))  # 只保存第一个的样本
        
        # 保存结果
        exp_name = checkpoint_path.parent.parent.name
        results['experiment'] = exp_name
        results['checkpoint_path'] = str(checkpoint_path)
        
        all_results.append(results)
        analyzer.add_result(results)
        
        # 保存单个实验结果
        exp_output_dir = Path(config.experiment.get('output_dir', './runs/evaluation')) / 'evaluation' / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(exp_output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # 统计分析
    stats = analyzer.compute_statistics()
    
    # 生成汇总报告
    summary_report = analyzer.generate_report()
    
    # 保存汇总结果
    summary_dir = Path(config.experiment.get('output_dir', './runs/evaluation')) / 'evaluation' / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(summary_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    with open(summary_dir / 'summary_report.txt', 'w') as f:
        f.write(summary_report)
    
    logger.info(f"Multi-checkpoint evaluation completed")
    logger.info(f"Summary saved to {summary_dir}")
    
    return {
        'all_results': all_results,
        'statistics': stats,
        'summary_report': summary_report
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directory containing multiple checkpoints')
    parser.add_argument('--multi_seed', action='store_true',
                       help='Evaluate multiple seeds and compute statistics')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--save_samples', action='store_true', default=True,
                       help='Save sample visualizations')
    
    args, unknown = parser.parse_known_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.multi_seed and args.checkpoint_dir:
            # 多种子评测
            results = evaluate_multiple_checkpoints(config, args.checkpoint_dir)
            print("Multi-seed evaluation completed!")
            print(f"Results saved to {config.experiment.get('output_dir', './runs/evaluation')}/evaluation/summary/")
            
        elif args.checkpoint:
            # 单个检查点评测
            evaluator = ModelEvaluator(config, args.checkpoint)
            results = evaluator.evaluate_model(
                save_samples=args.save_samples,
                max_samples=args.max_samples
            )
            
            evaluator.save_results()
            evaluator.create_visualizations()
            
            print("Evaluation completed!")
            print(f"Results saved to {evaluator.output_dir}")
            
            # 打印主要指标
            print("\nMain Results:")
            for metric in ['rel_l2', 'mae', 'psnr', 'ssim', 'dc_error']:
                if metric in results:
                    std_val = results.get(f'{metric}_std', 0)
                    print(f"  {metric}: {results[metric]:.6f} ± {std_val:.6f}")
        
        else:
            print("Please specify either --checkpoint or --checkpoint_dir with --multi_seed")
            return
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()