#!/usr/bin/env python3
"""
增强版实验结果汇总脚本

专门为横向对比实验设计，支持多模型、多种子的全面统计分析。

功能:
- 自动识别对比实验模式
- 生成论文级别的对比表格
- 提供模型排名和显著性分析
- 支持资源消耗对比
- 生成可视化报告

使用方法:
    python tools/enhanced_summarize.py --runs_dir runs/ --output paper_package/comparison/
    python tools/enhanced_summarize.py --runs_dir runs/ --baseline_method unet --output paper_package/comparison/
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import StatisticalAnalyzer


class EnhancedComparisonSummarizer:
    """增强版对比实验汇总器"""
    
    def __init__(self, runs_dir: str, output_dir: str, baseline_method: str = None):
        """
        Args:
            runs_dir: 实验结果目录
            output_dir: 输出目录
            baseline_method: 基线方法名称
        """
        self.runs_dir = Path(runs_dir)
        self.output_dir = Path(output_dir)
        self.baseline_method = baseline_method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logging()
        
        # 结果存储
        self.all_results = {}  # {method_name: {seed: metrics}}
        self.method_configs = {}  # {method_name: config}
        self.resource_stats = {}  # {method_name: resource_stats}
        
        # 统计分析器
        self.analyzer = StatisticalAnalyzer()
        
        # 对比实验专用配置
        self.comparison_config = {
            'main_metrics': ['rel_l2', 'mae', 'psnr', 'ssim'],
            'resource_metrics': ['params_m', 'flops_g', 'memory_gb', 'latency_ms'],
            'significance_level': 0.05,
            'min_seeds_for_stats': 3,  # 最少种子数才做统计
        }
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('EnhancedComparisonSummarizer')
        logger.setLevel(logging.INFO)
        
        # 创建处理器
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.output_dir / 'comparison_summary.log')
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def collect_comparison_results(self) -> None:
        """收集对比实验结果"""
        self.logger.info(f"开始收集对比实验结果: {self.runs_dir}")
        
        # 识别对比实验模式
        comparison_patterns = self._identify_comparison_experiments()
        
        if comparison_patterns:
            self.logger.info(f"检测到对比实验模式: {len(comparison_patterns)} 组实验")
            self._collect_by_patterns(comparison_patterns)
        else:
            self.logger.info("未检测到对比实验模式，使用标准收集方式")
            self._collect_all_experiments()
    
    def _identify_comparison_experiments(self) -> List[Dict]:
        """识别对比实验模式"""
        patterns = []
        
        # 查找可能的对比实验组
        exp_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        
        # 按模型名称分组
        model_groups = {}
        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            
            # 尝试解析实验名称
            model_name, seed = self._parse_experiment_name(exp_name)
            
            if model_name:
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append((exp_dir, seed))
        
        # 识别对比组
        if len(model_groups) >= 2:  # 至少2个模型才构成对比
            # 检查是否每个模型都有多个种子
            valid_comparison = all(
                len(seeds) >= self.comparison_config['min_seeds_for_stats']
                for seeds in model_groups.values()
            )
            
            if valid_comparison:
                patterns.append({
                    'type': 'model_comparison',
                    'models': list(model_groups.keys()),
                    'groups': model_groups,
                    'total_experiments': sum(len(seeds) for seeds in model_groups.values())
                })
        
        return patterns
    
    def _collect_by_patterns(self, patterns: List[Dict]) -> None:
        """按识别的模式收集结果"""
        for pattern in patterns:
            self.logger.info(f"收集对比模式: {pattern['type']}")
            
            if pattern['type'] == 'model_comparison':
                self._collect_model_comparison(pattern)
    
    def _collect_model_comparison(self, pattern: Dict) -> None:
        """收集模型对比实验结果"""
        for model_name, exp_list in pattern['groups'].items():
            self.all_results[model_name] = {}
            
            for exp_dir, seed in exp_list:
                # 查找结果文件
                metrics_file = exp_dir / "metrics_summary.json"
                config_file = exp_dir / "config_merged.yaml"
                resource_file = exp_dir / "resource_stats.json"
                
                if not metrics_file.exists():
                    self.logger.warning(f"跳过 {exp_dir.name}: 缺少结果文件")
                    continue
                
                try:
                    # 加载指标
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # 加载配置
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        self.method_configs[model_name] = config
                    
                    # 加载资源统计
                    if resource_file.exists():
                        with open(resource_file, 'r') as f:
                            resource_stats = json.load(f)
                        self.resource_stats[model_name] = resource_stats
                    
                    self.all_results[model_name][seed] = metrics
                    self.logger.info(f"加载成功: {model_name} (seed {seed})")
                    
                except Exception as e:
                    self.logger.error(f"加载失败 {exp_dir.name}: {e}")
    
    def _collect_all_experiments(self) -> None:
        """标准方式收集所有实验"""
        # 遍历所有实验目录
        for exp_dir in self.runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 解析实验名称
            exp_name = exp_dir.name
            method_name, seed = self._parse_experiment_name(exp_name)
            
            if method_name is None:
                self.logger.warning(f"跳过 {exp_name}: 无法解析方法名称")
                continue
            
            # 查找结果文件
            results_file = exp_dir / "metrics_summary.json"
            config_file = exp_dir / "config_merged.yaml"
            resource_file = exp_dir / "resource_stats.json"
            
            if not results_file.exists():
                self.logger.warning(f"跳过 {exp_name}: 缺少结果文件")
                continue
            
            try:
                # 加载结果
                with open(results_file, 'r') as f:
                    metrics = json.load(f)
                
                # 加载配置
                config = None
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                
                # 加载资源统计
                resource_stats = None
                if resource_file.exists():
                    with open(resource_file, 'r') as f:
                        resource_stats = json.load(f)
                
                # 存储结果
                if method_name not in self.all_results:
                    self.all_results[method_name] = {}
                    self.method_configs[method_name] = config
                    if resource_stats:
                        self.resource_stats[method_name] = resource_stats
                
                self.all_results[method_name][seed] = metrics
                self.logger.info(f"加载成功: {method_name} (seed {seed})")
                
            except Exception as e:
                self.logger.error(f"加载失败 {exp_name}: {e}")
                continue
    
    def _parse_experiment_name(self, exp_name: str) -> Tuple[Optional[str], Optional[int]]:
        """解析实验名称，提取方法名和种子"""
        # 尝试多种解析策略
        
        # 策略1: 查找种子标记 (s123, seed123, _123)
        import re
        
        seed_patterns = [
            r's(\d+)',      # s123
            r'seed(\d+)',   # seed123
            r'_([\d]{3,})', # _123
        ]
        
        seed = None
        method_name = exp_name
        
        for pattern in seed_patterns:
            match = re.search(pattern, exp_name)
            if match:
                seed = int(match.group(1))
                # 移除种子部分得到方法名
                method_name = re.sub(pattern, '', exp_name).strip('_-')
                break
        
        # 策略2: 如果没有找到种子，尝试解析模型名
        if seed is None:
            # 查找已知模型名
            known_models = ['swin_unet', 'unet', 'fno2d', 'segformer', 'unet_plus_plus', 
                          'hybrid', 'mlp', 'vit', 'transformer']
            
            for model in known_models:
                if model in exp_name.lower():
                    method_name = model
                    # 尝试从剩余部分提取种子
                    remaining = exp_name.lower().replace(model, '')
                    seed_match = re.search(r'(\d{3,})', remaining)
                    if seed_match:
                        seed = int(seed_match.group(1))
                    break
        
        # 策略3: 回退到简单解析
        if seed is None and len(exp_name) > 10:
            # 假设最后几个数字是种子
            digits = re.findall(r'\d+', exp_name)
            if digits:
                seed = int(digits[-1])
                method_name = exp_name.rstrip('0123456789').rstrip('_-')
        
        return method_name, seed
    
    def aggregate_comparison_results(self) -> Dict:
        """聚合对比实验结果"""
        self.logger.info("开始聚合对比实验结果")
        
        aggregated = {}
        
        for method_name, seeds_results in self.all_results.items():
            if len(seeds_results) < self.comparison_config['min_seeds_for_stats']:
                self.logger.warning(f"跳过 {method_name}: 种子数不足 ({len(seeds_results)} < {self.comparison_config['min_seeds_for_stats']})")
                continue
            
            # 转换为指标列表格式
            metrics_list = []
            for seed, metrics in seeds_results.items():
                # 提取主要指标
                tensor_metrics = {}
                for metric_name in self.comparison_config['main_metrics']:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if isinstance(value, (int, float)):
                            tensor_metrics[metric_name] = np.array([value])
                        elif isinstance(value, dict) and 'mean' in value:
                            tensor_metrics[metric_name] = np.array([value['mean']])
                
                if tensor_metrics:
                    metrics_list.append(tensor_metrics)
            
            if metrics_list:
                # 使用统计分析器聚合
                method_aggregated = self._aggregate_metrics_list(metrics_list)
                aggregated[method_name] = method_aggregated
                self.logger.info(f"聚合完成: {method_name} ({len(metrics_list)} 个种子)")
        
        return aggregated
    
    def _aggregate_metrics_list(self, metrics_list: List[Dict]) -> Dict:
        """聚合指标列表"""
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
                    values.append(metrics[metric_name])
            
            if values:
                values = np.concatenate(values)
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                    'values': values.tolist()
                }
        
        return aggregated
    
    def compute_significance_tests(self, aggregated_results: Dict) -> Dict:
        """计算显著性检验"""
        self.logger.info("开始显著性检验")
        
        if self.baseline_method is None or self.baseline_method not in self.all_results:
            # 自动选择基线（样本数最多的方法）
            baseline_method = max(self.all_results.keys(), 
                                key=lambda x: len(self.all_results[x]))
            self.logger.info(f"自动选择基线方法: {baseline_method}")
        else:
            baseline_method = self.baseline_method
        
        significance_results = {}
        
        # 准备基线数据
        baseline_metrics_list = []
        for seed, metrics in self.all_results[baseline_method].items():
            tensor_metrics = {}
            for metric_name in self.comparison_config['main_metrics']:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)):
                        tensor_metrics[metric_name] = np.array([value])
                    elif isinstance(value, dict) and 'mean' in value:
                        tensor_metrics[metric_name] = np.array([value['mean']])
            
            if tensor_metrics:
                baseline_metrics_list.append(tensor_metrics)
        
        # 对每个方法进行显著性检验
        for method_name, seeds_results in self.all_results.items():
            if method_name == baseline_method:
                continue
            
            if len(seeds_results) < self.comparison_config['min_seeds_for_stats']:
                continue
            
            # 准备方法数据
            method_metrics_list = []
            for seed, metrics in seeds_results.items():
                tensor_metrics = {}
                for metric_name in self.comparison_config['main_metrics']:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if isinstance(value, (int, float)):
                            tensor_metrics[metric_name] = np.array([value])
                        elif isinstance(value, dict) and 'mean' in value:
                            tensor_metrics[metric_name] = np.array([value['mean']])
                
                if tensor_metrics:
                    method_metrics_list.append(tensor_metrics)
            
            # 计算显著性检验
            method_significance = {}
            for metric_name in self.comparison_config['main_metrics']:
                try:
                    test_result = self._compute_statistical_test(
                        baseline_metrics_list, method_metrics_list, metric_name
                    )
                    method_significance[metric_name] = test_result
                except Exception as e:
                    self.logger.warning(f"显著性检验失败 {method_name}-{metric_name}: {e}")
                    method_significance[metric_name] = {'error': str(e)}
            
            significance_results[method_name] = method_significance
        
        return significance_results
    
    def _compute_statistical_test(self, baseline_list: List[Dict], 
                                 method_list: List[Dict], 
                                 metric_name: str) -> Dict:
        """计算统计检验"""
        # 提取数值
        baseline_values = []
        method_values = []
        
        for metrics in baseline_list:
            if metric_name in metrics:
                baseline_values.extend(metrics[metric_name])
        
        for metrics in method_list:
            if metric_name in metrics:
                method_values.extend(metrics[metric_name])
        
        if not baseline_values or not method_values:
            return {'error': '缺少数据'}
        
        # 计算基本统计
        baseline_mean = np.mean(baseline_values)
        method_mean = np.mean(method_values)
        
        # Paired t-test (如果样本数相同) 或独立t-test
        from scipy import stats
        
        if len(baseline_values) == len(method_values):
            # Paired t-test
            try:
                t_stat, p_value = stats.ttest_rel(method_values, baseline_values)
                test_type = 'paired_ttest'
            except:
                # 如果配对失败，使用独立t-test
                t_stat, p_value = stats.ttest_ind(method_values, baseline_values)
                test_type = 'independent_ttest'
        else:
            # 独立t-test
            t_stat, p_value = stats.ttest_ind(method_values, baseline_values)
            test_type = 'independent_ttest'
        
        # Cohen's d (效应量)
        pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) + 
                             (len(method_values) - 1) * np.var(method_values, ddof=1)) / 
                            (len(baseline_values) + len(method_values) - 2))
        
        cohens_d = (method_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # 结果解释
        effect_size = 'small'
        if abs(cohens_d) >= 0.8:
            effect_size = 'large'
        elif abs(cohens_d) >= 0.5:
            effect_size = 'medium'
        
        significance = 'significant' if p_value < self.comparison_config['significance_level'] else 'not_significant'
        
        return {
            'test_type': test_type,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significance': significance,
            'effect_size': effect_size,
            'cohens_d': float(cohens_d),
            'baseline_mean': float(baseline_mean),
            'method_mean': float(method_mean),
            'improvement': float((method_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
        }
    
    def generate_enhanced_comparison_tables(self, aggregated_results: Dict, 
                                          significance_results: Dict = None) -> Dict[str, str]:
        """生成增强版对比表格"""
        self.logger.info("生成增强版对比表格")
        
        tables = {}
        
        # 1. 主要结果对比表
        tables['main_comparison'] = self._generate_main_comparison_table(aggregated_results, significance_results)
        
        # 2. 资源消耗对比表
        if self.resource_stats:
            tables['resource_comparison'] = self._generate_resource_comparison_table()
        
        # 3. 模型排名表
        tables['model_ranking'] = self._generate_model_ranking_table(aggregated_results)
        
        # 4. 显著性检验汇总表
        if significance_results:
            tables['significance_summary'] = self._generate_significance_summary_table(significance_results)
        
        return tables
    
    def _generate_main_comparison_table(self, aggregated_results: Dict, 
                                       significance_results: Dict = None) -> str:
        """生成主要结果对比表"""
        if not aggregated_results:
            return "% 没有可汇总的结果"
        
        # 按性能排序（Rel-L2 升序）
        sorted_methods = sorted(aggregated_results.items(), 
                                key=lambda x: x[1].get('rel_l2', {}).get('mean', float('inf')))
        
        # LaTeX表格
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{模型横向对比实验结果 (均值±标准差)}",
            "\\label{tab:model_comparison}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l" + "c" * len(self.comparison_config['main_metrics']) + "}",
            "\\toprule",
            "\\textbf{模型} & " + " & ".join([f"\\textbf{{{m.upper()}}}" for m in self.comparison_config['main_metrics']]) + " \\\",
            "\\midrule"
        ]
        
        # 数据行
        for i, (method_name, metrics) in enumerate(sorted_methods):
            # 排名标记
            rank_mark = f"{{\bfseries({i+1})}}" if i < 3 else f"({i+1})"
            
            row_parts = [f"{method_name.replace('_', ' ').title()} {rank_mark}"]
            
            for metric_name in self.comparison_config['main_metrics']:
                if metric_name in metrics:
                    stats = metrics[metric_name]
                    mean = stats['mean']
                    std = stats['std']
                    
                    # 格式化数值
                    if metric_name in ['rel_l2', 'mae']:
                        value_str = f"{mean:.4f}±{std:.4f}"
                    elif metric_name == 'psnr':
                        value_str = f"{mean:.2f}±{std:.2f}"
                    elif metric_name == 'ssim':
                        value_str = f"{mean:.3f}±{std:.3f}"
                    else:
                        value_str = f"{mean:.3f}±{std:.3f}"
                    
                    # 显著性标记
                    if significance_results and method_name in significance_results:
                        sig_result = significance_results[method_name].get(metric_name, {})
                        if sig_result.get('significance') == 'significant':
                            if sig_result.get('improvement', 0) > 0:
                                value_str = f"\\textbf{{{value_str}}}\\textsuperscript{{+}}"  # 改进显著
                            else:
                                value_str = f"\\textbf{{{value_str}}}\\textsuperscript{{-}}"  # 下降显著
                    
                    row_parts.append(value_str)
                else:
                    row_parts.append("--")
            
            lines.append(" & ".join(row_parts) + " \\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def _generate_resource_comparison_table(self) -> str:
        """生成资源消耗对比表"""
        if not self.resource_stats:
            return "% 没有资源统计数据"
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{模型资源消耗对比}",
            "\\label{tab:resource_comparison}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{模型} & \\textbf{参数量(M)} & \\textbf{FLOPs(G)} & \\textbf{显存(GB)} & \\textbf{延迟(ms)} \\\\",
            "\\midrule"
        ]
        
        for method_name, stats in self.resource_stats.items():
            row_parts = [method_name.replace('_', ' ').title()]
            
            for metric in self.comparison_config['resource_metrics']:
                value = stats.get(metric, '--')
                if isinstance(value, (int, float)):
                    if metric == 'params_m':
                        row_parts.append(f"{value:.1f}")
                    elif metric == 'flops_g':
                        row_parts.append(f"{value:.1f}")
                    elif metric == 'memory_gb':
                        row_parts.append(f"{value:.1f}")
                    elif metric == 'latency_ms':
                        row_parts.append(f"{value:.1f}")
                    else:
                        row_parts.append(f"{value:.2f}")
                else:
                    row_parts.append(str(value))
            
            lines.append(" & ".join(row_parts) + " \\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def _generate_model_ranking_table(self, aggregated_results: Dict) -> str:
        """生成模型排名表"""
        if not aggregated_results:
            return "% 没有可汇总的结果"
        
        # 计算综合排名
        rankings = {}
        
        for method_name, metrics in aggregated_results.items():
            rankings[method_name] = {}
            
            for metric_name in self.comparison_config['main_metrics']:
                if metric_name in metrics:
                    value = metrics[metric_name]['mean']
                    rankings[method_name][metric_name] = value
        
        # 计算排名（考虑不同指标的最优方向）
        ranking_scores = {}
        for method_name, values in rankings.items():
            score = 0
            
            # Rel-L2: 越小越好
            if 'rel_l2' in values:
                rel_l2_rank = self._get_rank([v['rel_l2'] for v in rankings.values()], values['rel_l2'], ascending=True)
                score += rel_l2_rank
            
            # MAE: 越小越好
            if 'mae' in values:
                mae_rank = self._get_rank([v['mae'] for v in rankings.values()], values['mae'], ascending=True)
                score += mae_rank
            
            # PSNR: 越大越好
            if 'psnr' in values:
                psnr_rank = self._get_rank([v['psnr'] for v in rankings.values()], values['psnr'], ascending=False)
                score += psnr_rank
            
            # SSIM: 越大越好
            if 'ssim' in values:
                ssim_rank = self._get_rank([v['ssim'] for v in rankings.values()], values['ssim'], ascending=False)
                score += ssim_rank
            
            ranking_scores[method_name] = score
        
        # 按综合得分排序（越小越好）
        sorted_methods = sorted(ranking_scores.items(), key=lambda x: x[1])
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{模型综合排名}",
            "\\label{tab:model_ranking}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{排名} & \\textbf{模型} & \\textbf{Rel-L2} & \\textbf{MAE} & \\textbf{PSNR} & \\textbf{SSIM} \\\\",
            "\\midrule"
        ]
        
        for rank, (method_name, score) in enumerate(sorted_methods, 1):
            values = rankings[method_name]
            
            row_parts = [str(rank), method_name.replace('_', ' ').title()]
            
            for metric in self.comparison_config['main_metrics']:
                if metric in values:
                    value = values[metric]
                    if metric in ['rel_l2', 'mae']:
                        row_parts.append(f"{value:.4f}")
                    elif metric == 'psnr':
                        row_parts.append(f"{value:.2f}")
                    elif metric == 'ssim':
                        row_parts.append(f"{value:.3f}")
                else:
                    row_parts.append("--")
            
            lines.append(" & ".join(row_parts) + " \\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def _get_rank(self, all_values: List[float], target_value: float, ascending: bool = True) -> int:
        """计算排名"""
        sorted_values = sorted(all_values, reverse=not ascending)
        try:
            rank = sorted_values.index(target_value) + 1
        except ValueError:
            rank = len(all_values)
        return rank
    
    def _generate_significance_summary_table(self, significance_results: Dict) -> str:
        """生成显著性检验汇总表"""
        if not significance_results:
            return "% 没有显著性检验结果"
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{显著性检验结果汇总 (vs 基线)}",
            "\\label{tab:significance_summary}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{模型} & \\textbf{指标} & \\textbf{改进率(\\%)} & \\textbf{Cohen's d} & \\textbf{显著性} \\\\",
            "\\midrule"
        ]
        
        for method_name, method_sig in significance_results.items():
            first_row = True
            for metric_name, sig_result in method_sig.items():
                if 'error' in sig_result:
                    continue
                
                row_parts = []
                if first_row:
                    row_parts.append(method_name.replace('_', ' ').title())
                    first_row = False
                else:
                    row_parts.append("")
                
                row_parts.extend([
                    metric_name.upper(),
                    f"{sig_result['improvement']:.1f}",
                    f"{sig_result['cohens_d']:.2f}",
                    sig_result['significance'].replace('_', ' ').title()
                ])
                
                lines.append(" & ".join(row_parts) + " \\\")
            
            # 添加空行分隔不同模型
            lines.append("")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def save_enhanced_results(self, aggregated_results: Dict, 
                            significance_results: Dict = None,
                            tables: Dict[str, str] = None) -> None:
        """保存增强版结果"""
        self.logger.info("保存增强版结果")
        
        # 保存聚合结果
        with open(self.output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        # 保存显著性检验结果
        if significance_results:
            with open(self.output_dir / 'significance_results.json', 'w') as f:
                json.dump(significance_results, f, indent=2, default=str)
        
        # 保存表格
        if tables:
            for table_name, table_content in tables.items():
                table_file = self.output_dir / f'{table_name}_table.tex'
                with open(table_file, 'w') as f:
                    f.write(table_content)
                self.logger.info(f"保存表格: {table_file}")
        
        # 生成综合报告
        self._generate_comprehensive_report(aggregated_results, significance_results, tables)
        
        # 保存为CSV格式
        self._save_csv_results(aggregated_results, significance_results)
    
    def _generate_comprehensive_report(self, aggregated_results: Dict, 
                                   significance_results: Dict = None,
                                   tables: Dict[str, str] = None) -> None:
        """生成综合报告"""
        report_lines = [
            "# 模型横向对比实验报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"实验总数: {len(self.all_results)} 个模型",
            f"总实验次数: {sum(len(seeds) for seeds in self.all_results.values())} 次",
            "\n## 实验设置",
            f"- 基线模型: {self.baseline_method or '自动选择'}",
            f"- 最少种子数: {self.comparison_config['min_seeds_for_stats']}",
            f"- 显著性水平: {self.comparison_config['significance_level']}",
            "\n## 主要结果",
        ]
        
        # 最佳模型
        if aggregated_results:
            best_model = min(aggregated_results.items(), 
                           key=lambda x: x[1].get('rel_l2', {}).get('mean', float('inf')))
            
            report_lines.extend([
                f"\n### 最佳模型 (Rel-L2 指标)",
                f"- 模型: {best_model[0]}",
                f"- Rel-L2: {best_model[1]['rel_l2']['mean']:.4f} ± {best_model[1]['rel_l2']['std']:.4f}",
                f"- 种子数: {best_model[1]['rel_l2']['count']}",
            ])
        
        # 模型排名
        if aggregated_results:
            report_lines.extend([
                "\n### 模型排名 (按 Rel-L2 排序)",
                "",
                "| 排名 | 模型 | Rel-L2 (均值±标准差) | 种子数 |",
                "|------|------|---------------------|--------|",
            ])
            
            sorted_methods = sorted(aggregated_results.items(), 
                                  key=lambda x: x[1].get('rel_l2', {}).get('mean', float('inf')))
            
            for rank, (method_name, metrics) in enumerate(sorted_methods, 1):
                rel_l2_stats = metrics.get('rel_l2', {})
                mean_val = rel_l2_stats.get('mean', 0)
                std_val = rel_l2_stats.get('std', 0)
                count = rel_l2_stats.get('count', 0)
                
                report_lines.append(
                    f"| {rank} | {method_name} | {mean_val:.4f}±{std_val:.4f} | {count} |"
                )
        
        # 显著性检验总结
        if significance_results:
            report_lines.extend([
                "\n## 显著性检验结果",
                "",
                "| 模型 | 指标 | 改进率(%) | Cohen's d | 显著性 |",
                "|------|------|-----------|-----------|--------|",
            ])
            
            for method_name, method_sig in significance_results.items():
                for metric_name, sig_result in method_sig.items():
                    if 'error' in sig_result:
                        continue
                    
                    report_lines.append(
                        f"| {method_name} | {metric_name} | "
                        f"{sig_result['improvement']:.1f} | "
                        f"{sig_result['cohens_d']:.2f} | "
                        f"{sig_result['significance']} |"
                    )
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / 'comparison_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"生成综合报告: {report_file}")
    
    def _save_csv_results(self, aggregated_results: Dict, 
                         significance_results: Dict = None) -> None:
        """保存CSV格式结果"""
        if not aggregated_results:
            return
        
        # 主要结果CSV
        rows = []
        for method_name, metrics in aggregated_results.items():
            row = {'method': method_name}
            
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    row[f'{metric_name}_mean'] = stats['mean']
                    row[f'{metric_name}_std'] = stats['std']
                    row[f'{metric_name}_count'] = stats['count']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / 'comparison_results.csv'
        df.to_csv(csv_file, index=False)
        self.logger.info(f"保存CSV结果: {csv_file}")
    
    def run_enhanced_comparison(self) -> Dict:
        """运行完整的增强版对比汇总"""
        try:
            self.logger.info("开始增强版对比实验汇总")
            
            # 1. 收集结果
            self.collect_comparison_results()
            
            if not self.all_results:
                self.logger.warning("没有找到可汇总的结果")
                return {'status': 'no_results'}
            
            self.logger.info(f"成功收集 {len(self.all_results)} 个模型的结果")
            
            # 2. 聚合结果
            aggregated_results = self.aggregate_comparison_results()
            
            if not aggregated_results:
                self.logger.warning("聚合结果为空")
                return {'status': 'aggregation_failed'}
            
            # 3. 显著性检验
            significance_results = self.compute_significance_tests(aggregated_results)
            
            # 4. 生成表格
            tables = self.generate_enhanced_comparison_tables(aggregated_results, significance_results)
            
            # 5. 保存结果
            self.save_enhanced_results(aggregated_results, significance_results, tables)
            
            self.logger.info("增强版对比汇总完成")
            
            return {
                'status': 'success',
                'num_models': len(aggregated_results),
                'num_experiments': sum(len(seeds) for seeds in self.all_results.values()),
                'output_files': {
                    'report': str(self.output_dir / 'comparison_report.md'),
                    'csv': str(self.output_dir / 'comparison_results.csv'),
                    'tables': list(tables.keys()) if tables else []
                }
            }
            
        except Exception as e:
            self.logger.error(f"增强版对比汇总失败: {e}")
            return {'status': 'error', 'error': str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版实验结果汇总脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本使用
    python tools/enhanced_summarize.py --runs_dir runs/ --output paper_package/comparison/
    
    # 指定基线模型
    python tools/enhanced_summarize.py --runs_dir runs/ --baseline_method unet --output paper_package/comparison/
    
    # 详细输出
    python tools/enhanced_summarize.py --runs_dir runs/ --output paper_package/comparison/ --verbose
        """
    )
    
    parser.add_argument("--runs_dir", type=str, default="runs/",
                       help="实验结果目录 (默认: runs/)")
    parser.add_argument("--output", type=str, default="paper_package/comparison/",
                       help="输出目录 (默认: paper_package/comparison/)")
    parser.add_argument("--baseline_method", type=str, default=None,
                       help="基线方法名称 (默认: 自动选择)")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print(f"\n🚀 开始增强版对比实验汇总")
    print(f"实验目录: {args.runs_dir}")
    print(f"输出目录: {args.output}")
    print(f"基线模型: {args.baseline_method or '自动选择'}")
    print("="*60 + "\n")
    
    # 创建汇总器
    summarizer = EnhancedComparisonSummarizer(
        runs_dir=args.runs_dir,
        output_dir=args.output,
        baseline_method=args.baseline_method
    )
    
    # 运行汇总
    result = summarizer.run_enhanced_comparison()
    
    if result['status'] == 'success':
        print(f"\n✅ 对比汇总完成!")
        print(f"模型数量: {result['num_models']}")
        print(f"实验总数: {result['num_experiments']}")
        print(f"报告文件: {result['output_files']['report']}")
        print(f"CSV文件: {result['output_files']['csv']}")
        if result['output_files']['tables']:
            print(f"生成表格: {len(result['output_files']['tables'])} 个")
    else:
        print(f"\n❌ 对比汇总失败: {result.get('error', '未知错误')}")
        print(f"请查看日志: {args.output}/comparison_summary.log")


if __name__ == "__main__":
    main()