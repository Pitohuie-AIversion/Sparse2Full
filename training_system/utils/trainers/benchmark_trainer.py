#!/usr/bin/env python3
"""
基准测试训练器 - 支持多模型、多配置、多种子对比
遵循黄金法则，确保公平比较和可复现性
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from training_system.utils.trainers.trainer import PDEBenchTrainer
from training_system.utils.trainers.curriculum_trainer import CurriculumTrainer
# from utils.metrics import compute_resource_usage  # 暂时注释
# from utils.performance import PerformanceMonitor   # 暂时注释

logger = logging.getLogger(__name__)


class BenchmarkTrainer:
    """基准测试训练器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.benchmark_config = config.benchmark
        self.results = []
        self.output_dir = Path(config.experiment.output_dir) / config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"基准测试训练器初始化完成，将测试 {len(self.benchmark_config.models)} 个模型")
    
    def run_single_experiment(self, model_config: Dict, observation_config: Dict, 
                            loss_config: Dict, seed: int) -> Dict[str, Any]:
        """运行单个实验"""
        
        # 构建实验配置
        exp_config = OmegaConf.create({
            'experiment': {
                'name': f"{model_config['name']}_{observation_config['name']}_{loss_config['name']}_seed{seed}",
                'device': self.config.experiment.device,
                'seed': seed,
                'output_dir': str(self.output_dir)
            },
            'data': OmegaConf.merge(self.config.data, observation_config.get('config', {}).get('data', {})),
            'model': model_config,
            'training': self.config.training,
            'loss': OmegaConf.merge(self.config.loss, loss_config.get('config', {}).get('loss', {})),
            'validation': self.config.validation,
            'performance_monitoring': self.config.performance_monitoring,
            'logging': self.config.logging,
            'visualization': self.config.visualization,
            'paper_package': self.config.paper_package
        })
        
        logger.info(f"运行实验: {exp_config.experiment.name}")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 创建训练器
        if getattr(self.config, 'curriculum', {}).get('enabled', False):
            trainer = CurriculumTrainer(exp_config)
        else:
            trainer = PDEBenchTrainer(exp_config)
        
        # 运行训练
        start_time = time.time()
        
        try:
            training_results = trainer.train()
            
            # 计算资源使用
            resource_usage = compute_resource_usage(trainer.model)
            
            # 性能监控数据
            performance_stats = {}
            if trainer.performance_monitor:
                performance_stats = trainer.performance_monitor.get_stats()
            
            experiment_result = {
                'experiment_name': exp_config.experiment.name,
                'model_name': model_config['name'],
                'observation_name': observation_config['name'],
                'loss_name': loss_config['name'],
                'seed': seed,
                'training_time': time.time() - start_time,
                'best_val_metric': training_results['best_val_metric'],
                'total_epochs': training_results['total_epochs'],
                'resource_usage': resource_usage,
                'performance_stats': performance_stats,
                'config_snapshot': OmegaConf.to_yaml(exp_config),
                'status': 'completed'
            }
            
            logger.info(f"实验完成: {exp_config.experiment.name}, 最佳指标: {training_results['best_val_metric']:.6f}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"实验失败: {exp_config.experiment.name}, 错误: {str(e)}")
            
            return {
                'experiment_name': exp_config.experiment.name,
                'model_name': model_config['name'],
                'observation_name': observation_config['name'],
                'loss_name': loss_config['name'],
                'seed': seed,
                'training_time': time.time() - start_time,
                'best_val_metric': float('inf'),
                'total_epochs': 0,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """运行完整的基准测试"""
        logger.info("开始基准测试...")
        
        benchmark_results = {
            'config': OmegaConf.to_yaml(self.config),
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': [],
            'summary': {},
            'statistical_analysis': {}
        }
        
        # 获取所有配置组合
        models = self.benchmark_config.models
        observations = self.benchmark_config.observations
        loss_configs = self.benchmark_config.loss_configs
        seeds = self.benchmark_config.seeds
        
        total_experiments = len(models) * len(observations) * len(loss_configs) * len(seeds)
        logger.info(f"总实验数: {total_experiments}")
        
        # 运行所有实验
        experiment_counter = 0
        
        for model_config in models:
            for observation_config in observations:
                for loss_config in loss_configs:
                    for seed in seeds:
                        experiment_counter += 1
                        
                        logger.info(f"实验 {experiment_counter}/{total_experiments}")
                        
                        try:
                            result = self.run_single_experiment(
                                model_config, observation_config, loss_config, seed
                            )
                            benchmark_results['experiments'].append(result)
                            
                        except Exception as e:
                            logger.error(f"实验运行时错误: {str(e)}")
                            benchmark_results['experiments'].append({
                                'experiment_name': f"{model_config['name']}_{observation_config['name']}_{loss_config['name']}_seed{seed}",
                                'model_name': model_config['name'],
                                'observation_name': observation_config['name'],
                                'loss_name': loss_config['name'],
                                'seed': seed,
                                'status': 'runtime_error',
                                'error': str(e)
                            })
        
        # 生成汇总统计
        benchmark_results['summary'] = self._generate_summary(benchmark_results['experiments'])
        
        # 统计分析
        benchmark_results['statistical_analysis'] = self._perform_statistical_analysis(benchmark_results['experiments'])
        
        # 保存结果
        benchmark_results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # 生成报告
        self._generate_benchmark_report(benchmark_results)
        
        logger.info(f"基准测试完成，结果保存在: {results_file}")
        
        return benchmark_results
    
    def _generate_summary(self, experiments: List[Dict]) -> Dict[str, Any]:
        """生成实验汇总"""
        summary = {
            'total_experiments': len(experiments),
            'completed_experiments': 0,
            'failed_experiments': 0,
            'model_comparison': {},
            'observation_comparison': {},
            'loss_comparison': {}
        }
        
        # 按不同维度分组
        model_results = {}
        observation_results = {}
        loss_results = {}
        
        for exp in experiments:
            if exp['status'] == 'completed':
                summary['completed_experiments'] += 1
                
                # 模型分组
                model_name = exp['model_name']
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(exp['best_val_metric'])
                
                # 观测分组
                obs_name = exp['observation_name']
                if obs_name not in observation_results:
                    observation_results[obs_name] = []
                observation_results[obs_name].append(exp['best_val_metric'])
                
                # 损失分组
                loss_name = exp['loss_name']
                if loss_name not in loss_results:
                    loss_results[loss_name] = []
                loss_results[loss_name].append(exp['best_val_metric'])
                
            else:
                summary['failed_experiments'] += 1
        
        # 计算统计信息
        summary['model_comparison'] = self._compute_group_stats(model_results)
        summary['observation_comparison'] = self._compute_group_stats(observation_results)
        summary['loss_comparison'] = self._compute_group_stats(loss_results)
        
        return summary
    
    def _compute_group_stats(self, group_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """计算分组统计"""
        stats = {}
        
        for group_name, values in group_data.items():
            if values:
                stats[group_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return stats
    
    def _perform_statistical_analysis(self, experiments: List[Dict]) -> Dict[str, Any]:
        """执行统计分析"""
        from scipy import stats
        
        analysis = {
            'significance_tests': {},
            'correlation_analysis': {},
            'best_configurations': {}
        }
        
        # 获取完成的实验
        completed_exps = [exp for exp in experiments if exp['status'] == 'completed']
        
        if not completed_exps:
            return analysis
        
        # 模型显著性测试
        model_groups = {}
        for exp in completed_exps:
            model_name = exp['model_name']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(exp['best_val_metric'])
        
        # T检验
        if len(model_groups) >= 2:
            model_names = list(model_groups.keys())
            group1 = model_groups[model_names[0]]
            group2 = model_groups[model_names[1]]
            
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                analysis['significance_tests']['model_comparison'] = {
                    'models': [model_names[0], model_names[1]],
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # 最佳配置
        best_exp = min(completed_exps, key=lambda x: x['best_val_metric'])
        analysis['best_configurations'] = {
            'best_overall': {
                'experiment_name': best_exp['experiment_name'],
                'model': best_exp['model_name'],
                'observation': best_exp['observation_name'],
                'loss': best_exp['loss_name'],
                'seed': best_exp['seed'],
                'metric': best_exp['best_val_metric']
            }
        }
        
        return analysis
    
    def _generate_benchmark_report(self, results: Dict[str, Any]):
        """生成基准测试报告"""
        report_file = self.output_dir / "benchmark_report.md"
        
        with open(report_file, "w") as f:
            f.write("# PDEBench 基准测试报告\n\n")
            f.write(f"开始时间: {results['start_time']}\n")
            f.write(f"结束时间: {results['end_time']}\n\n")
            
            # 实验汇总
            f.write("## 实验汇总\n\n")
            summary = results['summary']
            f.write(f"- 总实验数: {summary['total_experiments']}\n")
            f.write(f"- 完成实验: {summary['completed_experiments']}\n")
            f.write(f"- 失败实验: {summary['failed_experiments']}\n\n")
            
            # 模型对比
            f.write("## 模型对比\n\n")
            f.write("| 模型 | 平均指标 | 标准差 | 最小值 | 最大值 | 实验数 |\n")
            f.write("|------|----------|--------|--------|--------|--------|\n")
            
            for model_name, stats in summary['model_comparison'].items():
                f.write(f"| {model_name} | {stats['mean']:.6f} | {stats['std']:.6f} | "
                       f"{stats['min']:.6f} | {stats['max']:.6f} | {stats['count']} |\n")
            
            f.write("\n")
            
            # 观测对比
            f.write("## 观测配置对比\n\n")
            f.write("| 观测配置 | 平均指标 | 标准差 | 最小值 | 最大值 | 实验数 |\n")
            f.write("|----------|----------|--------|--------|--------|--------|\n")
            
            for obs_name, stats in summary['observation_comparison'].items():
                f.write(f"| {obs_name} | {stats['mean']:.6f} | {stats['std']:.6f} | "
                       f"{stats['min']:.6f} | {stats['max']:.6f} | {stats['count']} |\n")
            
            f.write("\n")
            
            # 损失对比
            f.write("## 损失配置对比\n\n")
            f.write("| 损失配置 | 平均指标 | 标准差 | 最小值 | 最大值 | 实验数 |\n")
            f.write("|----------|----------|--------|--------|--------|--------|\n")
            
            for loss_name, stats in summary['loss_comparison'].items():
                f.write(f"| {loss_name} | {stats['mean']:.6f} | {stats['std']:.6f} | "
                       f"{stats['min']:.6f} | {stats['max']:.6f} | {stats['count']} |\n")
            
            f.write("\n")
            
            # 统计分析
            f.write("## 统计分析\n\n")
            analysis = results['statistical_analysis']
            
            if 'best_configurations' in analysis and 'best_overall' in analysis['best_configurations']:
                best = analysis['best_configurations']['best_overall']
                f.write(f"**最佳配置**: {best['experiment_name']}\n")
                f.write(f"- 模型: {best['model']}\n")
                f.write(f"- 观测: {best['observation']}\n")
                f.write(f"- 损失: {best['loss']}\n")
                f.write(f"- 种子: {best['seed']}\n")
                f.write(f"- 指标: {best['metric']:.6f}\n\n")
            
            # 显著性测试
            if 'significance_tests' in analysis and analysis['significance_tests']:
                sig_test = analysis['significance_tests']['model_comparison']
                f.write(f"**模型显著性测试**: t={sig_test['t_statistic']:.4f}, p={sig_test['p_value']:.4f}\n")
                f.write(f"显著性差异: {'是' if sig_test['significant'] else '否'}\n\n")
            
            f.write("## 详细结果\n\n")
            f.write("完整结果见 `benchmark_results.json`\n")
        
        logger.info(f"基准测试报告生成完成: {report_file}")