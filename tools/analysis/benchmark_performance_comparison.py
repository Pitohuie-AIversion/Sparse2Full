#!/usr/bin/env python3
"""
性能对比分析工具
对比原始版本和重构版本的性能差异
"""

import os
import sys
import json
import time
import psutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import subprocess
import tempfile
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

DEFAULT_ORIGINAL_SCRIPT = project_root / "tools" / "training" / "train_real_data_ar.py"
DEFAULT_REFACTORED_SCRIPT = project_root / "tools" / "training" / "train_real_data_ar_refactored.py"
DEFAULT_CONFIG = project_root / "configs" / "ar_training_refactored_config.yaml"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent: float
    gpu_memory_peak_mb: float
    gpu_memory_avg_mb: float
    gpu_utilization: float
    io_read_mb: float
    io_write_mb: float
    
@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    version: str  # 'original' or 'refactored'
    metrics: PerformanceMetrics
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""
    
@dataclass
class ComparisonResult:
    """对比结果"""
    benchmark_name: str
    original_result: Optional[BenchmarkResult]
    refactored_result: Optional[BenchmarkResult]
    speedup_ratio: Optional[float] = None
    memory_improvement_ratio: Optional[float] = None
    gpu_memory_improvement_ratio: Optional[float] = None
    
class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.io_start = self.process.io_counters()
        
        # GPU监控
        self.gpu_available = torch.cuda.is_available()
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        
        if self.gpu_available:
            self.gpu_start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
    
    def sample(self):
        """采样当前资源使用情况"""
        current_time = time.time() - self.start_time
        
        # 内存采样
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
        self.memory_samples.append((current_time, memory_mb))
        
        # CPU采样
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append((current_time, cpu_percent))
        
        # GPU采样
        if self.gpu_available:
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.gpu_memory_samples.append((current_time, gpu_memory_mb))
            
            # GPU利用率（简化实现）
            if hasattr(torch.cuda, 'utilization'):
                gpu_util = torch.cuda.utilization()
                self.gpu_util_samples.append((current_time, gpu_util))
    
    def get_metrics(self, execution_time: float) -> PerformanceMetrics:
        """获取性能指标"""
        # 内存统计
        memory_values = [m[1] for m in self.memory_samples]
        memory_peak = max(memory_values) if memory_values else 0
        memory_avg = np.mean(memory_values) if memory_values else 0
        
        # CPU统计
        cpu_values = [c[1] for c in self.cpu_samples if c[1] > 0]  # 过滤掉0值
        cpu_avg = np.mean(cpu_values) if cpu_values else 0
        
        # GPU统计
        gpu_memory_peak = 0
        gpu_memory_avg = 0
        gpu_util_avg = 0
        
        if self.gpu_available:
            gpu_memory_values = [m[1] for m in self.gpu_memory_samples]
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            gpu_memory_avg = np.mean(gpu_memory_values) if gpu_memory_values else 0
            
            gpu_util_values = [u[1] for u in self.gpu_util_samples if u[1] > 0]
            gpu_util_avg = np.mean(gpu_util_values) if gpu_util_values else 0
        
        # IO统计
        io_end = self.process.io_counters()
        io_read_mb = (io_end.read_bytes - self.io_start.read_bytes) / 1024 / 1024
        io_write_mb = (io_end.write_bytes - self.io_start.write_bytes) / 1024 / 1024
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_peak_mb=memory_peak,
            memory_avg_mb=memory_avg,
            cpu_percent=cpu_avg,
            gpu_memory_peak_mb=gpu_memory_peak,
            gpu_memory_avg_mb=gpu_memory_avg,
            gpu_utilization=gpu_util_avg,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb
        )

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, original_script: Path, refactored_script: Path, config_path: Path):
        self.original_script = original_script
        self.refactored_script = refactored_script
        self.config_path = config_path
        self.temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        
        # 验证文件存在
        if not self.original_script.exists():
            raise FileNotFoundError(f"原始脚本不存在: {self.original_script}")
        
        if not self.refactored_script.exists():
            raise FileNotFoundError(f"重构脚本不存在: {self.refactored_script}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
    
    def __del__(self):
        """清理临时目录"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, test_type: str) -> Dict:
        """创建测试配置"""
        # 加载基础配置
        with open(self.config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # 根据测试类型调整配置
        if test_type == "lightweight":
            config = self._create_lightweight_config(base_config)
        elif test_type == "medium":
            config = self._create_medium_config(base_config)
        elif test_type == "heavy":
            config = self._create_heavy_config(base_config)
        elif test_type == "memory_pressure":
            config = self._create_memory_pressure_config(base_config)
        elif test_type == "io_intensive":
            config = self._create_io_intensive_config(base_config)
        else:
            config = base_config
        
        return config
    
    def _create_lightweight_config(self, base_config: Dict) -> Dict:
        """创建轻量级测试配置"""
        config = base_config.copy()
        
        # 数据配置
        config['data']['image_size'] = [32, 32]
        config['data']['T_in'] = 2
        config['data']['T_out'] = 1
        config['data']['dataloader']['batch_size'] = 4
        
        # 模型配置
        config['model']['hidden_dim'] = 32
        config['model']['depths'] = [2, 2]
        config['model']['num_heads'] = [4, 8]
        
        # 训练配置
        config['training']['epochs'] = 2
        config['training']['log_interval'] = 1
        config['training']['val_interval'] = 1
        
        return config
    
    def _create_medium_config(self, base_config: Dict) -> Dict:
        """创建中等测试配置"""
        config = base_config.copy()
        
        # 数据配置
        config['data']['image_size'] = [64, 64]
        config['data']['T_in'] = 4
        config['data']['T_out'] = 1
        config['data']['dataloader']['batch_size'] = 8
        
        # 模型配置
        config['model']['hidden_dim'] = 64
        config['model']['depths'] = [2, 2, 2]
        config['model']['num_heads'] = [4, 8, 16]
        
        # 训练配置
        config['training']['epochs'] = 3
        config['training']['log_interval'] = 1
        config['training']['val_interval'] = 1
        
        return config
    
    def _create_heavy_config(self, base_config: Dict) -> Dict:
        """创建重量级测试配置"""
        config = base_config.copy()
        
        # 数据配置
        config['data']['image_size'] = [128, 128]
        config['data']['T_in'] = 8
        config['data']['T_out'] = 2
        config['data']['dataloader']['batch_size'] = 16
        
        # 模型配置
        config['model']['hidden_dim'] = 128
        config['model']['depths'] = [2, 2, 6, 2]
        config['model']['num_heads'] = [4, 8, 16, 32]
        
        # 训练配置
        config['training']['epochs'] = 2
        config['training']['log_interval'] = 1
        config['training']['val_interval'] = 1
        
        return config
    
    def _create_memory_pressure_config(self, base_config: Dict) -> Dict:
        """创建内存压力测试配置"""
        config = base_config.copy()
        
        # 大数据配置以产生内存压力
        config['data']['image_size'] = [256, 256]
        config['data']['T_in'] = 16
        config['data']['T_out'] = 4
        config['data']['dataloader']['batch_size'] = 32
        config['data']['dataloader']['num_workers'] = 4
        config['data']['dataloader']['prefetch_factor'] = 4
        
        # 大模型配置
        config['model']['hidden_dim'] = 256
        config['model']['depths'] = [2, 2, 18, 2]
        config['model']['num_heads'] = [4, 8, 16, 32]
        
        # 训练配置
        config['training']['epochs'] = 1
        config['training']['amp']['enabled'] = True
        
        return config
    
    def _create_io_intensive_config(self, base_config: Dict) -> Dict:
        """创建IO密集型测试配置"""
        config = base_config.copy()
        
        # 配置大量数据加载
        config['data']['image_size'] = [64, 64]
        config['data']['T_in'] = 4
        config['data']['T_out'] = 1
        config['data']['dataloader']['batch_size'] = 4
        config['data']['dataloader']['num_workers'] = 8
        config['data']['dataloader']['prefetch_factor'] = 8
        
        # 频繁保存检查点
        config['training']['checkpoint_interval'] = 1
        config['training']['val_interval'] = 1
        config['training']['log_interval'] = 1
        config['training']['epochs'] = 5
        
        return config
    
    def run_single_benchmark(self, script_path: Path, config: Dict, test_name: str, version: str) -> BenchmarkResult:
        """运行单个基准测试"""
        logger.info(f"运行 {test_name} - {version} 版本...")
        
        # 创建临时配置文件
        config_file = self.temp_dir / f"{test_name}_{version}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        # 准备命令
        cmd = [
            sys.executable, str(script_path),
            "--config", str(config_file),
            "--test-mode", "--max-epochs", str(config['training']['epochs'])
        ]
        
        # 开始监控
        monitor = ResourceMonitor()
        start_time = time.time()
        
        try:
            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )
            
            # 监控资源使用
            while process.poll() is None:
                monitor.sample()
                time.sleep(0.1)  # 100ms采样间隔
            
            # 获取执行时间
            execution_time = time.time() - start_time
            
            # 获取性能指标
            metrics = monitor.get_metrics(execution_time)
            
            # 检查执行结果
            success = process.returncode == 0
            stdout, stderr = process.communicate()
            
            if not success:
                error_msg = stderr.decode('utf-8')[-500:]  # 取最后500字符
                logger.warning(f"{test_name} - {version} 执行失败: {error_msg}")
            else:
                logger.info(f"✓ {test_name} - {version} 完成 ({execution_time:.1f}s)")
            
            return BenchmarkResult(
                name=test_name,
                version=version,
                metrics=metrics,
                success=success,
                error_message=error_msg if not success else None,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"{test_name} - {version} 执行错误: {e}")
            return BenchmarkResult(
                name=test_name,
                version=version,
                metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0),
                success=False,
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def run_comparison_benchmarks(self, test_suites: List[str]) -> List[ComparisonResult]:
        """运行对比基准测试"""
        results = []
        
        for test_suite in test_suites:
            logger.info(f"\n运行测试套件: {test_suite}")
            
            # 创建测试配置
            config = self.create_test_config(test_suite)
            
            # 运行原始版本
            original_result = self.run_single_benchmark(
                self.original_script, config, test_suite, "original"
            )
            
            # 运行重构版本
            refactored_result = self.run_single_benchmark(
                self.refactored_script, config, test_suite, "refactored"
            )
            
            # 计算对比结果
            comparison = self._calculate_comparison(original_result, refactored_result)
            results.append(comparison)
            
            # 短暂休息以避免系统过载
            time.sleep(2)
        
        return results
    
    def _calculate_comparison(self, original: BenchmarkResult, refactored: BenchmarkResult) -> ComparisonResult:
        """计算对比结果"""
        speedup_ratio = None
        memory_improvement_ratio = None
        gpu_memory_improvement_ratio = None
        
        if original.success and refactored.success:
            # 计算加速比
            if original.metrics.execution_time > 0:
                speedup_ratio = original.metrics.execution_time / refactored.metrics.execution_time
            
            # 计算内存改进比
            if original.metrics.memory_peak_mb > 0:
                memory_improvement_ratio = (original.metrics.memory_peak_mb - refactored.metrics.memory_peak_mb) / original.metrics.memory_peak_mb
            
            # 计算GPU内存改进比
            if original.metrics.gpu_memory_peak_mb > 0:
                gpu_memory_improvement_ratio = (original.metrics.gpu_memory_peak_mb - refactored.metrics.gpu_memory_peak_mb) / original.metrics.gpu_memory_peak_mb
        
        return ComparisonResult(
            benchmark_name=original.name,
            original_result=original,
            refactored_result=refactored,
            speedup_ratio=speedup_ratio,
            memory_improvement_ratio=memory_improvement_ratio,
            gpu_memory_improvement_ratio=gpu_memory_improvement_ratio
        )

class BenchmarkReporter:
    """基准测试报告生成器"""
    
    def __init__(self, results: List[ComparisonResult]):
        self.results = results
    
    def generate_reports(self, output_dir: Path):
        """生成所有报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        self._generate_text_report(output_dir / "performance_comparison_report.txt")
        
        # 生成JSON报告
        self._generate_json_report(output_dir / "performance_comparison_results.json")
        
        # 生成CSV数据
        self._generate_csv_data(output_dir / "performance_metrics.csv")
        
        # 生成可视化图表
        self._generate_visualizations(output_dir)
        
        # 生成性能分析文档
        self._generate_analysis_document(output_dir / "performance_analysis.md")
        
        logger.info(f"性能对比报告已生成: {output_dir}")
    
    def _generate_text_report(self, output_file: Path):
        """生成文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("性能对比分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总测试数: {len(self.results)}\n\n")
            
            # 总体统计
            successful_comparisons = [r for r in self.results if r.original_result.success and r.refactored_result.success]
            
            if successful_comparisons:
                avg_speedup = np.mean([r.speedup_ratio for r in successful_comparisons if r.speedup_ratio is not None])
                avg_memory_improvement = np.mean([r.memory_improvement_ratio for r in successful_comparisons if r.memory_improvement_ratio is not None])
                
                f.write(f"成功对比数: {len(successful_comparisons)}\n")
                f.write(f"平均加速比: {avg_speedup:.2f}x\n")
                f.write(f"平均内存改进: {avg_memory_improvement*100:.1f}%\n\n")
            
            # 详细结果
            f.write("详细对比结果:\n")
            f.write("-" * 60 + "\n\n")
            
            for result in self.results:
                f.write(f"测试: {result.benchmark_name}\n")
                f.write(f"原始版本: {'✓ 成功' if result.original_result.success else '✗ 失败'}\n")
                f.write(f"重构版本: {'✓ 成功' if result.refactored_result.success else '✗ 失败'}\n")
                
                if result.original_result.success and result.refactored_result.success:
                    orig_metrics = result.original_result.metrics
                    ref_metrics = result.refactored_result.metrics
                    
                    f.write(f"执行时间: {orig_metrics.execution_time:.1f}s → {ref_metrics.execution_time:.1f}s")
                    if result.speedup_ratio:
                        f.write(f" (加速 {result.speedup_ratio:.2f}x)")
                    f.write("\n")
                    
                    f.write(f"内存峰值: {orig_metrics.memory_peak_mb:.0f}MB → {ref_metrics.memory_peak_mb:.0f}MB")
                    if result.memory_improvement_ratio:
                        f.write(f" (改进 {result.memory_improvement_ratio*100:.1f}%)")
                    f.write("\n")
                    
                    if orig_metrics.gpu_memory_peak_mb > 0:
                        f.write(f"GPU内存峰值: {orig_metrics.gpu_memory_peak_mb:.0f}MB → {ref_metrics.gpu_memory_peak_mb:.0f}MB")
                        if result.gpu_memory_improvement_ratio:
                            f.write(f" (改进 {result.gpu_memory_improvement_ratio*100:.1f}%)")
                        f.write("\n")
                
                f.write("\n")
    
    def _generate_json_report(self, output_file: Path):
        """生成JSON报告"""
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'successful_comparisons': len([r for r in self.results if r.original_result.success and r.refactored_result.success])
            },
            'results': [asdict(result) for result in self.results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_csv_data(self, output_file: Path):
        """生成CSV数据"""
        data = []
        
        for result in self.results:
            if result.original_result.success and result.refactored_result.success:
                orig_metrics = result.original_result.metrics
                ref_metrics = result.refactored_result.metrics
                
                data.append({
                    'benchmark_name': result.benchmark_name,
                    'original_execution_time': orig_metrics.execution_time,
                    'refactored_execution_time': ref_metrics.execution_time,
                    'speedup_ratio': result.speedup_ratio,
                    'original_memory_peak_mb': orig_metrics.memory_peak_mb,
                    'refactored_memory_peak_mb': ref_metrics.memory_peak_mb,
                    'memory_improvement_ratio': result.memory_improvement_ratio,
                    'original_gpu_memory_peak_mb': orig_metrics.gpu_memory_peak_mb,
                    'refactored_gpu_memory_peak_mb': ref_metrics.gpu_memory_peak_mb,
                    'gpu_memory_improvement_ratio': result.gpu_memory_improvement_ratio,
                    'original_cpu_percent': orig_metrics.cpu_percent,
                    'refactored_cpu_percent': ref_metrics.cpu_percent,
                    'original_gpu_utilization': orig_metrics.gpu_utilization,
                    'refactored_gpu_utilization': ref_metrics.gpu_utilization
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    def _generate_visualizations(self, output_dir: Path):
        """生成可视化图表"""
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        
        # 准备数据
        successful_results = [r for r in self.results if r.original_result.success and r.refactored_result.success]
        
        if not successful_results:
            logger.warning("没有成功的对比结果，跳过可视化生成")
            return
        
        # 1. 执行时间对比图
        self._plot_execution_time_comparison(successful_results, output_dir)
        
        # 2. 内存使用对比图
        self._plot_memory_usage_comparison(successful_results, output_dir)
        
        # 3. 性能改进散点图
        self._plot_performance_improvement_scatter(successful_results, output_dir)
        
        # 4. 性能雷达图
        self._plot_performance_radar(successful_results, output_dir)
        
        # 5. 综合性能热力图
        self._plot_performance_heatmap(successful_results, output_dir)
    
    def _plot_execution_time_comparison(self, results: List[ComparisonResult], output_dir: Path):
        """绘制执行时间对比图"""
        plt.figure(figsize=(12, 6))
        
        benchmark_names = [r.benchmark_name for r in results]
        original_times = [r.original_result.metrics.execution_time for r in results]
        refactored_times = [r.refactored_result.metrics.execution_time for r in results]
        
        x = np.arange(len(benchmark_names))
        width = 0.35
        
        plt.bar(x - width/2, original_times, width, label='原始版本', color='lightcoral', alpha=0.8)
        plt.bar(x + width/2, refactored_times, width, label='重构版本', color='skyblue', alpha=0.8)
        
        plt.xlabel('测试套件')
        plt.ylabel('执行时间 (秒)')
        plt.title('执行时间对比')
        plt.xticks(x, benchmark_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage_comparison(self, results: List[ComparisonResult], output_dir: Path):
        """绘制内存使用对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        benchmark_names = [r.benchmark_name for r in results]
        original_memory = [r.original_result.metrics.memory_peak_mb for r in results]
        refactored_memory = [r.refactored_result.metrics.memory_peak_mb for r in results]
        
        # CPU内存对比
        x = np.arange(len(benchmark_names))
        width = 0.35
        
        ax1.bar(x - width/2, original_memory, width, label='原始版本', color='lightcoral', alpha=0.8)
        ax1.bar(x + width/2, refactored_memory, width, label='重构版本', color='skyblue', alpha=0.8)
        
        ax1.set_xlabel('测试套件')
        ax1.set_ylabel('内存峰值 (MB)')
        ax1.set_title('CPU内存使用对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmark_names, rotation=45, ha='right')
        ax1.legend()
        
        # GPU内存对比（如果有数据）
        gpu_original = [r.original_result.metrics.gpu_memory_peak_mb for r in results]
        gpu_refactored = [r.refactored_result.metrics.gpu_memory_peak_mb for r in results]
        
        if any(gpu > 0 for gpu in gpu_original):
            ax2.bar(x - width/2, gpu_original, width, label='原始版本', color='lightgreen', alpha=0.8)
            ax2.bar(x + width/2, gpu_refactored, width, label='重构版本', color='lightblue', alpha=0.8)
            
            ax2.set_xlabel('测试套件')
            ax2.set_ylabel('GPU内存峰值 (MB)')
            ax2.set_title('GPU内存使用对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(benchmark_names, rotation=45, ha='right')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无GPU数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('GPU内存使用对比')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'memory_usage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_improvement_scatter(self, results: List[ComparisonResult], output_dir: Path):
        """绘制性能改进散点图"""
        plt.figure(figsize=(10, 8))
        
        speedup_ratios = [r.speedup_ratio for r in results if r.speedup_ratio is not None]
        memory_improvements = [r.memory_improvement_ratio * 100 for r in results if r.memory_improvement_ratio is not None]
        benchmark_names = [r.benchmark_name for r in results if r.speedup_ratio is not None]
        
        scatter = plt.scatter(speedup_ratios, memory_improvements, 
                            s=100, alpha=0.7, c=range(len(speedup_ratios)), cmap='viridis')
        
        # 添加标签
        for i, name in enumerate(benchmark_names):
            plt.annotate(name, (speedup_ratios[i], memory_improvements[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='无内存改进')
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='无速度改进')
        
        plt.xlabel('加速比 (倍)')
        plt.ylabel('内存改进 (%)')
        plt.title('性能改进散点图')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加象限说明
        plt.text(0.1, 0.9, '象限1\n速度↓ 内存↑', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        plt.text(1.5, 0.9, '象限2\n速度↑ 内存↑', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        plt.text(0.1, 0.1, '象限3\n速度↓ 内存↓', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        plt.text(1.5, 0.1, '象限4\n速度↑ 内存↓', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_improvement_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, results: List[ComparisonResult], output_dir: Path):
        """绘制性能雷达图"""
        if len(results) < 2:
            return
        
        # 选择前几个测试进行雷达图对比
        selected_results = results[:min(4, len(results))]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        metrics = ['执行时间', '内存峰值', 'CPU使用率', 'GPU内存', 'GPU利用率']
        
        for i, result in enumerate(selected_results):
            ax = axes[i]
            
            # 标准化指标到0-1范围
            orig_metrics = result.original_result.metrics
            ref_metrics = result.refactored_result.metrics
            
            # 注意：时间和内存是越小越好，所以使用倒数
            orig_values = [
                1.0 / (orig_metrics.execution_time + 1),  # 执行时间倒数
                1.0 / (orig_metrics.memory_peak_mb + 1),    # 内存倒数
                orig_metrics.cpu_percent / 100.0,           # CPU使用率
                1.0 / (orig_metrics.gpu_memory_peak_mb + 1) if orig_metrics.gpu_memory_peak_mb > 0 else 0.5,  # GPU内存倒数
                orig_metrics.gpu_utilization / 100.0        # GPU利用率
            ]
            
            ref_values = [
                1.0 / (ref_metrics.execution_time + 1),
                1.0 / (ref_metrics.memory_peak_mb + 1),
                ref_metrics.cpu_percent / 100.0,
                1.0 / (ref_metrics.gpu_memory_peak_mb + 1) if ref_metrics.gpu_memory_peak_mb > 0 else 0.5,
                ref_metrics.gpu_utilization / 100.0
            ]
            
            # 创建角度
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            orig_values += orig_values[:1]
            ref_values += ref_values[:1]
            
            # 绘制雷达图
            ax.plot(angles, orig_values, 'o-', linewidth=2, label='原始版本', color='lightcoral')
            ax.fill(angles, orig_values, alpha=0.25, color='lightcoral')
            
            ax.plot(angles, ref_values, 'o-', linewidth=2, label='重构版本', color='skyblue')
            ax.fill(angles, ref_values, alpha=0.25, color='skyblue')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title(f'{result.benchmark_name}', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
        
        # 隐藏未使用的子图
        for i in range(len(selected_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('性能雷达图对比', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, results: List[ComparisonResult], output_dir: Path):
        """绘制性能热力图"""
        # 准备数据
        benchmark_names = [r.benchmark_name for r in results]
        
        # 创建性能矩阵
        performance_matrix = []
        for result in results:
            orig_metrics = result.original_result.metrics
            ref_metrics = result.refactored_result.metrics
            
            row = [
                orig_metrics.execution_time,
                ref_metrics.execution_time,
                orig_metrics.memory_peak_mb,
                ref_metrics.memory_peak_mb,
                orig_metrics.cpu_percent,
                ref_metrics.cpu_percent
            ]
            
            if orig_metrics.gpu_memory_peak_mb > 0:
                row.extend([
                    orig_metrics.gpu_memory_peak_mb,
                    ref_metrics.gpu_memory_peak_mb,
                    orig_metrics.gpu_utilization,
                    ref_metrics.gpu_utilization
                ])
            else:
                row.extend([0, 0, 0, 0])
            
            performance_matrix.append(row)
        
        # 创建DataFrame
        columns = ['原始时间', '重构时间', '原始内存', '重构内存', '原始CPU', '重构CPU', 
                  '原始GPU内存', '重构GPU内存', '原始GPU利用率', '重构GPU利用率']
        
        df = pd.DataFrame(performance_matrix, index=benchmark_names, columns=columns)
        
        # 标准化数据用于热力图（按列标准化）
        df_normalized = df.copy()
        for col in df.columns:
            if df[col].max() != df[col].min():
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': '标准化性能值'})
        plt.title('性能指标热力图（标准化）')
        plt.xlabel('性能指标')
        plt.ylabel('测试套件')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_document(self, output_file: Path):
        """生成性能分析文档"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 性能对比分析文档\n\n")
            
            # 执行摘要
            f.write("## 执行摘要\n\n")
            
            successful_results = [r for r in self.results if r.original_result.success and r.refactored_result.success]
            
            if successful_results:
                avg_speedup = np.mean([r.speedup_ratio for r in successful_results if r.speedup_ratio is not None])
                avg_memory_improvement = np.mean([r.memory_improvement_ratio for r in successful_results if r.memory_improvement_ratio is not None])
                
                f.write(f"重构版本相比原始版本平均加速 **{avg_speedup:.2f}倍**，")
                f.write(f"内存使用平均改进 **{avg_memory_improvement*100:.1f}%**。\n\n")
                
                # 性能分类
                speedup_categories = {
                    '显著加速 (>2x)': sum(1 for r in successful_results if r.speedup_ratio and r.speedup_ratio > 2),
                    '中等加速 (1.2-2x)': sum(1 for r in successful_results if r.speedup_ratio and 1.2 < r.speedup_ratio <= 2),
                    '轻微加速 (1-1.2x)': sum(1 for r in successful_results if r.speedup_ratio and 1 <= r.speedup_ratio <= 1.2),
                    '性能下降 (<1x)': sum(1 for r in successful_results if r.speedup_ratio and r.speedup_ratio < 1)
                }
                
                f.write("### 加速比分布\n\n")
                for category, count in speedup_categories.items():
                    f.write(f"- {category}: {count} 个测试\n")
                f.write("\n")
            
            # 详细分析
            f.write("## 详细分析\n\n")
            
            for result in self.results:
                f.write(f"### {result.benchmark_name}\n\n")
                
                if not (result.original_result.success and result.refactored_result.success):
                    f.write("**状态**: 测试执行失败\n\n")
                    if result.original_result.error_message:
                        f.write(f"原始版本错误: {result.original_result.error_message[:200]}...\n\n")
                    if result.refactored_result.error_message:
                        f.write(f"重构版本错误: {result.refactored_result.error_message[:200]}...\n\n")
                    continue
                
                orig_metrics = result.original_result.metrics
                ref_metrics = result.refactored_result.metrics
                
                f.write("**性能对比**:\n\n")
                f.write(f"- 执行时间: {orig_metrics.execution_time:.1f}s → {ref_metrics.execution_time:.1f}s")
                if result.speedup_ratio:
                    f.write(f" (加速 {result.speedup_ratio:.2f}x)")
                f.write("\n")
                
                f.write(f"- 内存峰值: {orig_metrics.memory_peak_mb:.0f}MB → {ref_metrics.memory_peak_mb:.0f}MB")
                if result.memory_improvement_ratio:
                    f.write(f" (改进 {result.memory_improvement_ratio*100:.1f}%)")
                f.write("\n")
                
                if orig_metrics.gpu_memory_peak_mb > 0:
                    f.write(f"- GPU内存峰值: {orig_metrics.gpu_memory_peak_mb:.0f}MB → {ref_metrics.gpu_memory_peak_mb:.0f}MB")
                    if result.gpu_memory_improvement_ratio:
                        f.write(f" (改进 {result.gpu_memory_improvement_ratio*100:.1f}%)")
                    f.write("\n")
                
                f.write(f"- CPU使用率: {orig_metrics.cpu_percent:.1f}% → {ref_metrics.cpu_percent:.1f}%\n")
                
                # 性能分析
                f.write("\n**分析**:\n\n")
                
                if result.speedup_ratio and result.speedup_ratio > 1.5:
                    f.write("- ✅ **显著性能提升**: 重构版本在该测试上表现优异\n")
                elif result.speedup_ratio and result.speedup_ratio > 1.1:
                    f.write("- 🟡 **中等性能提升**: 重构版本有一定性能改进\n")
                elif result.speedup_ratio and result.speedup_ratio < 0.9:
                    f.write("- 🔴 **性能下降**: 重构版本在该测试上性能下降，需要优化\n")
                else:
                    f.write("- ⚪ **性能相当**: 两个版本性能相近\n")
                
                if result.memory_improvement_ratio and result.memory_improvement_ratio > 0.2:
                    f.write("- ✅ **内存使用优化**: 重构版本显著减少了内存使用\n")
                elif result.memory_improvement_ratio and result.memory_improvement_ratio < -0.1:
                    f.write("- 🔴 **内存使用增加**: 重构版本内存使用增加，需要关注\n")
                
                f.write("\n")
            
            # 建议
            f.write("## 优化建议\n\n")
            
            # 性能下降的测试
            slow_tests = [r for r in successful_results if r.speedup_ratio and r.speedup_ratio < 1]
            if slow_tests:
                f.write("### 需要性能优化的测试\n\n")
                f.write("以下测试显示重构版本性能下降，建议进行优化:\n\n")
                for test in slow_tests:
                    f.write(f"- **{test.benchmark_name}**: 速度下降 {1/test.speedup_ratio:.2f}x\n")
                f.write("\n")
            
            # 内存使用增加的测试
            memory_increased_tests = [r for r in successful_results if r.memory_improvement_ratio and r.memory_improvement_ratio < -0.1]
            if memory_increased_tests:
                f.write("### 需要内存优化的测试\n\n")
                f.write("以下测试显示重构版本内存使用增加，建议进行内存优化:\n\n")
                for test in memory_increased_tests:
                    f.write(f"- **{test.benchmark_name}**: 内存增加 {abs(test.memory_improvement_ratio)*100:.1f}%\n")
                f.write("\n")
            
            f.write("### 通用优化建议\n\n")
            f.write("1. **代码优化**: 检查重构代码中的性能瓶颈\n")
            f.write("2. **内存管理**: 优化内存分配和释放策略\n")
            f.write("3. **并行化**: 考虑使用更多的并行化技术\n")
            f.write("4. **缓存优化**: 优化数据访问模式和缓存使用\n")
            f.write("5. **算法优化**: 检查是否有更高效的算法实现\n")
            
            f.write("\n---\n")
            f.write(f"分析生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="性能对比分析工具")
    parser.add_argument("--original-script", type=str,
                       default=str(DEFAULT_ORIGINAL_SCRIPT),
                       help="原始脚本路径")
    parser.add_argument("--refactored-script", type=str,
                       default=str(DEFAULT_REFACTORED_SCRIPT),
                       help="重构脚本路径")
    parser.add_argument("--config", type=str,
                       default=str(DEFAULT_CONFIG),
                       help="配置文件路径")
    parser.add_argument("--output", type=str, default="performance_comparison_results",
                       help="输出目录")
    parser.add_argument("--test-suites", nargs='+', 
                       default=['lightweight', 'medium', 'heavy', 'memory_pressure', 'io_intensive'],
                       help="测试套件列表")
    parser.add_argument("--parallel", action='store_true',
                       help="并行运行测试")
    parser.add_argument("--timeout", type=int, default=1800,
                       help="单个测试超时时间（秒）")
    
    args = parser.parse_args()
    
    logger.info("开始性能对比分析...")
    
    # 创建测试运行器
    try:
        runner = BenchmarkRunner(
            Path(args.original_script),
            Path(args.refactored_script),
            Path(args.config)
        )
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"创建测试运行器失败: {e}")
        sys.exit(1)
    
    # 运行对比测试
    logger.info(f"运行测试套件: {args.test_suites}")
    results = runner.run_comparison_benchmarks(args.test_suites)
    
    # 生成报告
    output_dir = Path(args.output)
    reporter = BenchmarkReporter(results)
    reporter.generate_reports(output_dir)
    
    # 打印总结
    logger.info(f"\n性能对比分析完成!")
    
    successful_results = [r for r in results if r.original_result.success and r.refactored_result.success]
    if successful_results:
        avg_speedup = np.mean([r.speedup_ratio for r in successful_results if r.speedup_ratio is not None])
        avg_memory_improvement = np.mean([r.memory_improvement_ratio for r in successful_results if r.memory_improvement_ratio is not None])
        
        logger.info(f"成功对比数: {len(successful_results)}/{len(results)}")
        logger.info(f"平均加速比: {avg_speedup:.2f}x")
        logger.info(f"平均内存改进: {avg_memory_improvement*100:.1f}%")
    
    # 显示失败测试
    failed_results = [r for r in results if not (r.original_result.success and r.refactored_result.success)]
    if failed_results:
        logger.warning(f"失败测试数: {len(failed_results)}")
        for result in failed_results:
            logger.warning(f"  - {result.benchmark_name}")
    
    logger.info(f"详细报告已保存到: {output_dir}")
    
    # 返回适当的退出码
    sys.exit(0 if len(failed_results) == 0 else 1)

if __name__ == "__main__":
    main()
