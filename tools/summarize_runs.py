#!/usr/bin/env python3
"""实验结果汇总脚本

自动汇总多次实验的结果，生成标准化的论文表格和显著性分析报告

使用方法:
    python tools/summarize_runs.py --runs_dir runs/ --output paper_package/metrics/
    python tools/summarize_runs.py --runs_dir runs/ --baseline_method unet --output paper_package/metrics/
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
from collections import defaultdict
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import StatisticalAnalyzer


class RunsSummarizer:
    """实验结果汇总器
    
    负责收集、分析和汇总多次实验的结果
    生成论文级别的表格和统计分析
    """
    
    def __init__(self, runs_dir: str, output_dir: str):
        """
        Args:
            runs_dir: 实验结果目录
            output_dir: 输出目录
        """
        self.runs_dir = Path(runs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.all_results = {}  # {method_name: {seed: metrics}}
        self.method_configs = {}  # {method_name: config}
        
        # 统计分析器
        self.analyzer = StatisticalAnalyzer()
    
    def collect_results(self) -> None:
        """收集所有实验结果"""
        print(f"Collecting results from {self.runs_dir}")
        
        # 遍历所有实验目录
        for exp_dir in self.runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 解析实验名称
            exp_name = exp_dir.name
            method_name, seed = self._parse_experiment_name(exp_name)
            
            if method_name is None:
                print(f"Skipping {exp_name}: cannot parse method name")
                continue
            
            # 查找结果文件
            results_file = exp_dir / "metrics_summary.json"
            config_file = exp_dir / "config_merged.yaml"
            
            if not results_file.exists():
                print(f"Skipping {exp_name}: no metrics_summary.json found")
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
                
                # 存储结果
                if method_name not in self.all_results:
                    self.all_results[method_name] = {}
                    self.method_configs[method_name] = config
                
                self.all_results[method_name][seed] = metrics
                
                print(f"Loaded {exp_name}: {method_name} (seed {seed})")
                
            except Exception as e:
                print(f"Error loading {exp_name}: {e}")
                continue
        
        print(f"Collected results for {len(self.all_results)} methods")
        for method_name, seeds in self.all_results.items():
            print(f"  {method_name}: {len(seeds)} seeds")
    
    def _parse_experiment_name(self, exp_name: str) -> Tuple[Optional[str], Optional[int]]:
        """解析实验名称
        
        期望格式: <task>-<data>-<res>-<model>-<keyhyper>-<seed>-<date>
        例如: SRx4-DR2D-256-SwinFNO_w8d2262_m16-s2025-20251011
        
        Args:
            exp_name: 实验名称
            
        Returns:
            method_name: 方法名称
            seed: 随机种子
        """
        parts = exp_name.split('-')
        
        if len(parts) < 6:
            return None, None
        
        try:
            # 查找种子部分（格式为 s<数字>）
            seed_part = None
            seed_idx = None
            for i, part in enumerate(parts):
                if part.startswith('s') and part[1:].isdigit():
                    seed_part = part
                    seed_idx = i
                    break
            
            if seed_part is None:
                return None, None
            
            seed = int(seed_part[1:])
            
            # 方法名称是除了种子和日期之外的部分
            method_parts = parts[:seed_idx] + parts[seed_idx+1:-1]  # 排除日期
            method_name = '-'.join(method_parts)
            
            return method_name, seed
            
        except (ValueError, IndexError):
            return None, None
    
    def aggregate_results(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """聚合所有方法的结果
        
        Returns:
            aggregated: {method_name: {metric_name: {mean, std, min, max, count}}}
        """
        aggregated = {}
        
        for method_name, seeds_results in self.all_results.items():
            # 转换为指标列表格式
            metrics_list = []
            for seed, metrics in seeds_results.items():
                # 将聚合指标转换为tensor格式（模拟）
                tensor_metrics = {}
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        # 已经是聚合格式，直接使用mean值
                        tensor_metrics[metric_name] = torch.tensor([stats['mean']])
                    else:
                        # 原始值
                        tensor_metrics[metric_name] = torch.tensor([stats])
                
                metrics_list.append(tensor_metrics)
            
            # 使用统计分析器聚合
            method_aggregated = self.analyzer.aggregate_metrics(metrics_list)
            aggregated[method_name] = method_aggregated
        
        return aggregated
    
    def compute_significance_tests(self, aggregated_results: Dict, 
                                 baseline_method: str = None) -> Dict:
        """计算显著性检验
        
        Args:
            aggregated_results: 聚合结果
            baseline_method: 基线方法名称
            
        Returns:
            significance_results: 显著性检验结果
        """
        if baseline_method is None or baseline_method not in self.all_results:
            # 自动选择基线（样本数最多的方法）
            baseline_method = max(self.all_results.keys(), 
                                key=lambda x: len(self.all_results[x]))
            print(f"Auto-selected baseline method: {baseline_method}")
        
        significance_results = {}
        
        # 准备基线数据
        baseline_metrics_list = []
        for seed, metrics in self.all_results[baseline_method].items():
            tensor_metrics = {}
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    tensor_metrics[metric_name] = torch.tensor([stats['mean']])
                else:
                    tensor_metrics[metric_name] = torch.tensor([stats])
            baseline_metrics_list.append(tensor_metrics)
        
        # 对每个方法进行显著性检验
        for method_name, seeds_results in self.all_results.items():
            if method_name == baseline_method:
                continue
            
            # 准备方法数据
            method_metrics_list = []
            for seed, metrics in seeds_results.items():
                tensor_metrics = {}
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        tensor_metrics[metric_name] = torch.tensor([stats['mean']])
                    else:
                        tensor_metrics[metric_name] = torch.tensor([stats])
                method_metrics_list.append(tensor_metrics)
            
            # 计算显著性检验
            method_significance = {}
            for metric_name in ['rel_l2', 'mae', 'psnr', 'ssim']:
                try:
                    test_result = self.analyzer.compute_significance_test(
                        baseline_metrics_list, method_metrics_list, metric_name
                    )
                    method_significance[metric_name] = test_result
                except Exception as e:
                    print(f"Significance test failed for {method_name}-{metric_name}: {e}")
                    method_significance[metric_name] = {'error': str(e)}
            
            significance_results[method_name] = method_significance
        
        return significance_results
    
    def generate_main_table(self, aggregated_results: Dict, 
                          significance_results: Dict = None) -> str:
        """生成主要结果表格
        
        Args:
            aggregated_results: 聚合结果
            significance_results: 显著性检验结果
            
        Returns:
            table_latex: LaTeX格式的表格
        """
        # 主要指标
        main_metrics = ['rel_l2', 'mae', 'psnr', 'ssim', 'dc_error']
        
        # 表头
        header = "Method"
        for metric in main_metrics:
            header += f" & {metric.upper()}"
        header += " \\\\"
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Main Results on PDEBench Sparse Observation Reconstruction}",
            "\\label{tab:main_results}",
            f"\\begin{{tabular}}{{l{'c' * len(main_metrics)}}}",
            "\\toprule",
            header,
            "\\midrule"
        ]
        
        # 数据行
        for method_name in sorted(aggregated_results.keys()):
            method_results = aggregated_results[method_name]
            
            # 清理方法名称（用于显示）
            display_name = method_name.replace('_', '\\_')
            row = display_name
            
            for metric in main_metrics:
                if metric in method_results:
                    stats = method_results[metric]
                    mean = stats['mean']
                    std = stats['std']
                    
                    # 格式化数值
                    if metric in ['rel_l2', 'mae', 'dc_error']:
                        value_str = f"{mean:.4f}±{std:.4f}"
                    elif metric == 'psnr':
                        value_str = f"{mean:.2f}±{std:.2f}"
                    else:  # ssim
                        value_str = f"{mean:.3f}±{std:.3f}"
                    
                    # 添加显著性标记
                    if (significance_results and method_name in significance_results and
                        metric in significance_results[method_name]):
                        sig_result = significance_results[method_name][metric]
                        if isinstance(sig_result, dict) and sig_result.get('significant', False):
                            if sig_result.get('p_value', 1.0) < 0.001:
                                value_str += "***"
                            elif sig_result.get('p_value', 1.0) < 0.01:
                                value_str += "**"
                            elif sig_result.get('p_value', 1.0) < 0.05:
                                value_str += "*"
                    
                    row += f" & {value_str}"
                else:
                    row += " & -"
            
            row += " \\\\"
            lines.append(row)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item Note: Values are reported as mean±std over multiple seeds.",
            "\\item Significance levels: *** p<0.001, ** p<0.01, * p<0.05",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_resource_table(self, aggregated_results: Dict) -> str:
        """生成资源消耗表格
        
        Args:
            aggregated_results: 聚合结果
            
        Returns:
            table_latex: LaTeX格式的资源表格
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Resource Consumption Comparison}",
            "\\label{tab:resources}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & Params (M) & FLOPs (G) & Memory (GB) & Latency (ms) \\\\",
            "\\midrule"
        ]
        
        for method_name in sorted(aggregated_results.keys()):
            method_results = aggregated_results[method_name]
            
            display_name = method_name.replace('_', '\\_')
            row = display_name
            
            # 资源指标
            resource_metrics = ['params_m', 'flops_g', 'memory_gb', 'latency_ms']
            for metric in resource_metrics:
                if metric in method_results:
                    stats = method_results[metric]
                    mean = stats['mean']
                    
                    if metric == 'params_m':
                        value_str = f"{mean:.2f}"
                    elif metric == 'flops_g':
                        value_str = f"{mean:.1f}"
                    elif metric == 'memory_gb':
                        value_str = f"{mean:.1f}"
                    else:  # latency_ms
                        value_str = f"{mean:.1f}"
                    
                    row += f" & {value_str}"
                else:
                    row += " & -"
            
            row += " \\\\"
            lines.append(row)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_significance_report(self, significance_results: Dict, 
                                   baseline_method: str) -> str:
        """生成显著性检验报告
        
        Args:
            significance_results: 显著性检验结果
            baseline_method: 基线方法
            
        Returns:
            report: 显著性检验报告
        """
        report = []
        report.append("Statistical Significance Analysis")
        report.append("=" * 50)
        report.append(f"Baseline method: {baseline_method}")
        report.append("")
        
        for method_name, method_tests in significance_results.items():
            report.append(f"Method: {method_name}")
            report.append("-" * 30)
            
            for metric_name, test_result in method_tests.items():
                if 'error' in test_result:
                    report.append(f"  {metric_name}: Test failed ({test_result['error']})")
                    continue
                
                t_stat = test_result.get('t_stat', 0)
                p_value = test_result.get('p_value', 1)
                cohen_d = test_result.get('cohen_d', 0)
                significant = test_result.get('significant', False)
                
                sig_mark = "***" if significant else ""
                effect_size = "large" if abs(cohen_d) > 0.8 else "medium" if abs(cohen_d) > 0.5 else "small"
                
                report.append(f"  {metric_name:10s}: t={t_stat:6.3f}, p={p_value:.6f}, "
                            f"d={cohen_d:6.3f} ({effect_size}) {sig_mark}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, aggregated_results: Dict, significance_results: Dict = None,
                    baseline_method: str = None) -> None:
        """保存所有结果
        
        Args:
            aggregated_results: 聚合结果
            significance_results: 显著性检验结果
            baseline_method: 基线方法
        """
        # 保存原始数据
        with open(self.output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        if significance_results:
            with open(self.output_dir / 'significance_results.json', 'w') as f:
                json.dump(significance_results, f, indent=2, default=str)
        
        # 生成表格
        main_table = self.generate_main_table(aggregated_results, significance_results)
        with open(self.output_dir / 'main_table.tex', 'w') as f:
            f.write(main_table)
        
        resource_table = self.generate_resource_table(aggregated_results)
        with open(self.output_dir / 'resource_table.tex', 'w') as f:
            f.write(resource_table)
        
        # 生成报告
        if significance_results and baseline_method:
            sig_report = self.generate_significance_report(significance_results, baseline_method)
            with open(self.output_dir / 'significance_report.txt', 'w') as f:
                f.write(sig_report)
        
        # 生成CSV（便于进一步分析）
        self._save_csv(aggregated_results)
        
        print(f"Results saved to {self.output_dir}")
    
    def _save_csv(self, aggregated_results: Dict) -> None:
        """保存CSV格式的结果"""
        # 准备数据
        rows = []
        for method_name, method_results in aggregated_results.items():
            row = {'method': method_name}
            for metric_name, stats in method_results.items():
                row[f'{metric_name}_mean'] = stats['mean']
                row[f'{metric_name}_std'] = stats['std']
                row[f'{metric_name}_count'] = stats['count']
            rows.append(row)
        
        # 保存为CSV
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'results.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Summarize experimental runs')
    parser.add_argument('--runs_dir', type=str, default='runs/', help='Directory containing experimental runs')
    parser.add_argument('--output', type=str, default='paper_package/metrics/', help='Output directory')
    parser.add_argument('--baseline_method', type=str, default=None, help='Baseline method for significance tests')
    
    args = parser.parse_args()
    
    # 创建汇总器
    summarizer = RunsSummarizer(args.runs_dir, args.output)
    
    # 收集结果
    summarizer.collect_results()
    
    if not summarizer.all_results:
        print("No results found!")
        return
    
    # 聚合结果
    aggregated_results = summarizer.aggregate_results()
    
    # 计算显著性检验
    significance_results = summarizer.compute_significance_tests(
        aggregated_results, args.baseline_method
    )
    
    # 保存结果
    summarizer.save_results(
        aggregated_results, significance_results, 
        args.baseline_method or list(summarizer.all_results.keys())[0]
    )
    
    print("Summary completed!")


if __name__ == '__main__':
    # 添加torch导入（用于tensor操作）
    import torch
    main()