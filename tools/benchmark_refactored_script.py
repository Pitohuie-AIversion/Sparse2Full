#!/usr/bin/env python3
"""
重构脚本性能基准测试工具
对train_real_data_ar_refactored.py进行全面的性能基准测试
"""

import os
import sys
import time
import json
import psutil
import tracemalloc
import tempfile
import subprocess
import threading
import queue
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

DEFAULT_REFACTORED_SCRIPT = project_root / "tools" / "training" / "train_real_data_ar_refactored.py"
DEFAULT_CONFIG = project_root / "configs" / "ar_training_refactored_config.yaml"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_percent_avg: float
    cpu_percent_peak: float
    gpu_memory_peak_mb: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    timestamp: str
    metrics: PerformanceMetrics
    config: Dict[str, Any]
    status: str
    error_message: Optional[str] = None

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        
        # GPU监控
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pass
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU和内存
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                # GPU信息（如果可用）
                gpu_memory = None
                gpu_util = None
                if self.gpu_available:
                    try:
                        import pynvml
                        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1024 / 1024  # MB
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    except:
                        pass
                
                # 磁盘I/O
                io_counters = self.process.io_counters()
                disk_read = io_counters.read_bytes / 1024 / 1024  # MB
                disk_write = io_counters.write_bytes / 1024 / 1024  # MB
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / 1024 / 1024,  # MB
                    'gpu_memory_mb': gpu_memory,
                    'gpu_utilization': gpu_util,
                    'disk_read_mb': disk_read,
                    'disk_write_mb': disk_write
                }
                
                self.metrics_queue.put(metrics)
                
            except Exception as e:
                print(f"监控错误: {e}")
            
            time.sleep(0.1)  # 100ms间隔
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取监控指标"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, script_path: str, config_path: str):
        self.script_path = Path(script_path)
        self.config_path = Path(config_path)
        self.results: List[BenchmarkResult] = []
        
        # 创建临时目录用于测试
        self.temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        
        # 确保脚本可执行
        if not self.script_path.exists():
            raise FileNotFoundError(f"脚本文件不存在: {self.script_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
    
    def __del__(self):
        """清理临时目录"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        print("开始运行性能基准测试...")
        
        # 定义测试套件
        test_suites = [
            ("轻量级测试", self._create_lightweight_config),
            ("中等负载测试", self._create_medium_config),
            ("重负载测试", self._create_heavy_config),
            ("内存压力测试", self._create_memory_pressure_config),
            ("I/O压力测试", self._create_io_pressure_config),
            ("并发测试", self._create_concurrent_config)
        ]
        
        for test_name, config_creator in test_suites:
            print(f"\n运行 {test_name}...")
            try:
                result = self._run_single_benchmark(test_name, config_creator)
                self.results.append(result)
                print(f"✓ {test_name} 完成")
            except Exception as e:
                print(f"✗ {test_name} 失败: {e}")
                # 记录失败结果
                failed_result = BenchmarkResult(
                    test_name=test_name,
                    timestamp=datetime.now().isoformat(),
                    metrics=PerformanceMetrics(0, 0, 0, 0, 0),
                    config={},
                    status="FAILED",
                    error_message=str(e)
                )
                self.results.append(failed_result)
        
        return self.results
    
    def _create_lightweight_config(self) -> Dict[str, Any]:
        """创建轻量级测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_lightweight',
                'seed': 42
            },
            'data': {
                'T_in': 2,
                'T_out': 1,
                'dataloader': {
                    'batch_size': 2,
                    'val_batch_size': 2,
                    'num_workers': 1,
                    'pin_memory': False
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 16,
                'depths': [1, 1],
                'num_heads': [2, 4]
            },
            'training': {
                'epochs': 1,
                'optimizer': {
                    'lr': 0.001
                },
                'scheduler': {
                    'name': 'cosine',
                    'T_max': 10
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': True,
                'max_batches': 5
            }
        }
    
    def _create_medium_config(self) -> Dict[str, Any]:
        """创建中等负载测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_medium',
                'seed': 42
            },
            'data': {
                'T_in': 4,
                'T_out': 2,
                'dataloader': {
                    'batch_size': 8,
                    'val_batch_size': 8,
                    'num_workers': 2,
                    'pin_memory': True
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 64,
                'depths': [2, 2, 2],
                'num_heads': [4, 8, 16]
            },
            'training': {
                'epochs': 2,
                'optimizer': {
                    'lr': 0.001
                },
                'scheduler': {
                    'name': 'cosine',
                    'T_max': 20
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': False,
                'max_batches': 10
            }
        }
    
    def _create_heavy_config(self) -> Dict[str, Any]:
        """创建重负载测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_heavy',
                'seed': 42
            },
            'data': {
                'T_in': 8,
                'T_out': 4,
                'dataloader': {
                    'batch_size': 16,
                    'val_batch_size': 16,
                    'num_workers': 4,
                    'pin_memory': True,
                    'persistent_workers': True
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 128,
                'depths': [2, 2, 6, 2],
                'num_heads': [4, 8, 16, 32]
            },
            'training': {
                'epochs': 3,
                'optimizer': {
                    'lr': 0.001
                },
                'scheduler': {
                    'name': 'cosine',
                    'T_max': 50
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': False,
                'max_batches': 20
            }
        }
    
    def _create_memory_pressure_config(self) -> Dict[str, Any]:
        """创建内存压力测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_memory_pressure',
                'seed': 42
            },
            'data': {
                'T_in': 16,
                'T_out': 8,
                'dataloader': {
                    'batch_size': 32,
                    'val_batch_size': 32,
                    'num_workers': 4,
                    'pin_memory': True,
                    'prefetch_factor': 4
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 256,
                'depths': [2, 2, 18, 2],
                'num_heads': [8, 16, 32, 64]
            },
            'training': {
                'epochs': 1,
                'optimizer': {
                    'lr': 0.001
                },
                'amp': {
                    'enabled': True,
                    'opt_level': 'O1'
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': False,
                'max_batches': 5
            }
        }
    
    def _create_io_pressure_config(self) -> Dict[str, Any]:
        """创建I/O压力测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_io_pressure',
                'seed': 42
            },
            'data': {
                'T_in': 4,
                'T_out': 2,
                'dataloader': {
                    'batch_size': 4,
                    'val_batch_size': 4,
                    'num_workers': 8,  # 高并发I/O
                    'pin_memory': False,  # 强制磁盘I/O
                    'persistent_workers': False
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 64,
                'depths': [2, 2, 2],
                'num_heads': [4, 8, 16]
            },
            'training': {
                'epochs': 2,
                'optimizer': {
                    'lr': 0.001
                },
                'checkpointing': {
                    'enabled': True,
                    'save_every': 1,  # 频繁保存
                    'keep_last': 10
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': False,
                'max_batches': 15
            }
        }
    
    def _create_concurrent_config(self) -> Dict[str, Any]:
        """创建并发测试配置"""
        return {
            'experiment': {
                'name': 'benchmark_concurrent',
                'seed': 42
            },
            'data': {
                'T_in': 4,
                'T_out': 2,
                'dataloader': {
                    'batch_size': 8,
                    'val_batch_size': 8,
                    'num_workers': 0,  # 将在测试中动态设置
                    'pin_memory': True
                }
            },
            'model': {
                'name': 'SwinUNet',
                'hidden_dim': 64,
                'depths': [2, 2, 2],
                'num_heads': [4, 8, 16]
            },
            'training': {
                'epochs': 1,
                'optimizer': {
                    'lr': 0.001
                }
            },
            'testing': {
                'enabled': True,
                'test_mode': True,
                'skip_validation': False,
                'max_batches': 10
            }
        }
    
    def _run_single_benchmark(self, test_name: str, config_creator) -> BenchmarkResult:
        """运行单个基准测试"""
        # 创建测试配置
        test_config = config_creator()
        config_file = self.temp_dir / f"{test_name.lower().replace(' ', '_')}_config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        
        # 开始监控
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # 开始内存跟踪
        tracemalloc.start()
        
        start_time = time.time()
        status = "SUCCESS"
        error_message = None
        
        try:
            # 运行脚本
            cmd = [
                sys.executable, str(self.script_path),
                "--config", str(config_file),
                "--test-mode"  # 确保在测试模式下运行
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.temp_dir),
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode != 0:
                status = "FAILED"
                error_message = result.stderr[:500]  # 限制错误消息长度
                
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            error_message = "测试超时 (30分钟)"
        except Exception as e:
            status = "ERROR"
            error_message = str(e)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 获取内存统计
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 处理监控数据
        metrics_data = monitor.get_metrics()
        
        if metrics_data:
            # 计算统计信息
            memory_values = [m['memory_mb'] for m in metrics_data if m['memory_mb'] is not None]
            cpu_values = [m['cpu_percent'] for m in metrics_data if m['cpu_percent'] is not None]
            gpu_memory_values = [m['gpu_memory_mb'] for m in metrics_data if m['gpu_memory_mb'] is not None]
            gpu_util_values = [m['gpu_utilization'] for m in metrics_data if m['gpu_utilization'] is not None]
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_peak_mb=max(memory_values) if memory_values else peak / 1024 / 1024,
                memory_avg_mb=np.mean(memory_values) if memory_values else peak / 1024 / 1024,
                cpu_percent_avg=np.mean(cpu_values) if cpu_values else 0,
                cpu_percent_peak=max(cpu_values) if cpu_values else 0,
                gpu_memory_peak_mb=max(gpu_memory_values) if gpu_memory_values else None,
                gpu_utilization_avg=np.mean(gpu_util_values) if gpu_util_values else None
            )
        else:
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_peak_mb=peak / 1024 / 1024,
                memory_avg_mb=current / 1024 / 1024,
                cpu_percent_avg=0,
                cpu_percent_peak=0
            )
        
        return BenchmarkResult(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=test_config,
            status=status,
            error_message=error_message
        )

class BenchmarkReporter:
    """基准测试报告生成器"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
    
    def generate_report(self, output_dir: Path):
        """生成完整的基准测试报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        self._generate_text_report(output_dir / "benchmark_report.txt")
        
        # 生成JSON报告
        self._generate_json_report(output_dir / "benchmark_results.json")
        
        # 生成可视化报告
        self._generate_visualization_report(output_dir)
        
        # 生成性能分析
        self._generate_performance_analysis(output_dir / "performance_analysis.md")
        
        print(f"基准测试报告已生成: {output_dir}")
    
    def _generate_text_report(self, output_file: Path):
        """生成文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("重构脚本性能基准测试报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试总数: {len(self.results)}\n")
            f.write(f"成功测试: {sum(1 for r in self.results if r.status == 'SUCCESS')}\n")
            f.write(f"失败测试: {sum(1 for r in self.results if r.status != 'SUCCESS')}\n\n")
            
            for result in self.results:
                f.write(f"测试: {result.test_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"状态: {result.status}\n")
                
                if result.status == "SUCCESS":
                    metrics = result.metrics
                    f.write(f"执行时间: {metrics.execution_time:.2f} 秒\n")
                    f.write(f"内存峰值: {metrics.memory_peak_mb:.1f} MB\n")
                    f.write(f"内存平均: {metrics.memory_avg_mb:.1f} MB\n")
                    f.write(f"CPU平均使用率: {metrics.cpu_percent_avg:.1f}%\n")
                    f.write(f"CPU峰值使用率: {metrics.cpu_percent_peak:.1f}%\n")
                    
                    if metrics.gpu_memory_peak_mb:
                        f.write(f"GPU内存峰值: {metrics.gpu_memory_peak_mb:.1f} MB\n")
                    if metrics.gpu_utilization_avg:
                        f.write(f"GPU平均使用率: {metrics.gpu_utilization_avg:.1f}%\n")
                else:
                    f.write(f"错误信息: {result.error_message}\n")
                
                f.write("\n")
    
    def _generate_json_report(self, output_file: Path):
        """生成JSON报告"""
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r.status == 'SUCCESS'),
                'failed_tests': sum(1 for r in self.results if r.status != 'SUCCESS')
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'timestamp': r.timestamp,
                    'status': r.status,
                    'error_message': r.error_message,
                    'metrics': asdict(r.metrics) if r.status == 'SUCCESS' else None,
                    'config': r.config
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_visualization_report(self, output_dir: Path):
        """生成可视化报告"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 筛选成功的测试
        successful_results = [r for r in self.results if r.status == "SUCCESS"]
        
        if not successful_results:
            print("没有成功的测试可用于生成可视化报告")
            return
        
        # 1. 执行时间对比图
        plt.figure(figsize=(12, 6))
        test_names = [r.test_name for r in successful_results]
        execution_times = [r.metrics.execution_time for r in successful_results]
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(test_names)), execution_times)
        plt.xlabel('测试')
        plt.ylabel('执行时间 (秒)')
        plt.title('执行时间对比')
        plt.xticks(range(len(test_names)), test_names, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, execution_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.1f}s', ha='center', va='bottom')
        
        # 2. 内存使用对比图
        plt.subplot(1, 2, 2)
        memory_peaks = [r.metrics.memory_peak_mb for r in successful_results]
        memory_avgs = [r.metrics.memory_avg_mb for r in successful_results]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        plt.bar(x - width/2, memory_peaks, width, label='峰值内存', alpha=0.8)
        plt.bar(x + width/2, memory_avgs, width, label='平均内存', alpha=0.8)
        
        plt.xlabel('测试')
        plt.ylabel('内存使用 (MB)')
        plt.title('内存使用对比')
        plt.xticks(x, test_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. CPU使用率图
        plt.figure(figsize=(10, 6))
        cpu_avgs = [r.metrics.cpu_percent_avg for r in successful_results]
        cpu_peaks = [r.metrics.cpu_percent_peak for r in successful_results]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        plt.bar(x - width/2, cpu_avgs, width, label='平均CPU使用率', alpha=0.8)
        plt.bar(x + width/2, cpu_peaks, width, label='峰值CPU使用率', alpha=0.8)
        
        plt.xlabel('测试')
        plt.ylabel('CPU使用率 (%)')
        plt.title('CPU使用率对比')
        plt.xticks(x, test_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cpu_usage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 性能散点图
        plt.figure(figsize=(10, 8))
        
        # 执行时间 vs 内存峰值
        plt.subplot(2, 2, 1)
        execution_times = [r.metrics.execution_time for r in successful_results]
        memory_peaks = [r.metrics.memory_peak_mb for r in successful_results]
        
        plt.scatter(execution_times, memory_peaks, s=100, alpha=0.7)
        for i, name in enumerate(test_names):
            plt.annotate(name, (execution_times[i], memory_peaks[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('执行时间 (秒)')
        plt.ylabel('内存峰值 (MB)')
        plt.title('执行时间 vs 内存峰值')
        
        # CPU平均 vs 内存平均
        plt.subplot(2, 2, 2)
        cpu_avgs = [r.metrics.cpu_percent_avg for r in successful_results]
        memory_avgs = [r.metrics.memory_avg_mb for r in successful_results]
        
        plt.scatter(cpu_avgs, memory_avgs, s=100, alpha=0.7, color='orange')
        for i, name in enumerate(test_names):
            plt.annotate(name, (cpu_avgs[i], memory_avgs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('平均CPU使用率 (%)')
        plt.ylabel('平均内存 (MB)')
        plt.title('CPU使用率 vs 内存使用')
        
        # 性能效率图
        plt.subplot(2, 2, 3)
        efficiency_scores = []
        for r in successful_results:
            # 简单的效率分数：1/(时间*内存)
            efficiency = 1.0 / (r.metrics.execution_time * r.metrics.memory_peak_mb)
            efficiency_scores.append(efficiency)
        
        bars = plt.bar(range(len(test_names)), efficiency_scores)
        plt.xlabel('测试')
        plt.ylabel('效率分数')
        plt.title('性能效率对比')
        plt.xticks(range(len(test_names)), test_names, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, efficiency_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 资源使用热力图
        plt.subplot(2, 2, 4)
        
        # 创建资源使用矩阵
        resource_matrix = []
        for r in successful_results:
            row = [
                r.metrics.execution_time / max(execution_times),
                r.metrics.memory_peak_mb / max(memory_peaks),
                r.metrics.cpu_percent_avg / 100.0,
                efficiency_scores[successful_results.index(r)] / max(efficiency_scores)
            ]
            resource_matrix.append(row)
        
        resource_matrix = np.array(resource_matrix)
        
        sns.heatmap(resource_matrix, 
                   xticklabels=['时间', '内存', 'CPU', '效率'],
                   yticklabels=test_names,
                   annot=True, fmt='.2f', cmap='RdYlGn')
        plt.title('资源使用热力图')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_analysis(self, output_file: Path):
        """生成性能分析文档"""
        successful_results = [r for r in self.results if r.status == "SUCCESS"]
        
        if not successful_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# 性能分析\n\n")
                f.write("没有成功的测试可用于性能分析。\n")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 重构脚本性能分析\n\n")
            
            # 执行时间分析
            f.write("## 执行时间分析\n\n")
            execution_times = [(r.test_name, r.metrics.execution_time) for r in successful_results]
            execution_times.sort(key=lambda x: x[1])
            
            f.write("执行时间排名（从快到慢）：\n")
            for i, (name, time) in enumerate(execution_times, 1):
                f.write(f"{i}. **{name}**: {time:.2f}秒\n")
            
            fastest = execution_times[0]
            slowest = execution_times[-1]
            f.write(f"\n最快测试 **{fastest[0]}** 比最慢测试 **{slowest[0]}** 快 {slowest[1]/fastest[1]:.1f} 倍。\n\n")
            
            # 内存使用分析
            f.write("## 内存使用分析\n\n")
            memory_peaks = [(r.test_name, r.metrics.memory_peak_mb) for r in successful_results]
            memory_peaks.sort(key=lambda x: x[1])
            
            f.write("内存使用排名（从少到多）：\n")
            for i, (name, memory) in enumerate(memory_peaks, 1):
                f.write(f"{i}. **{name}**: {memory:.1f}MB\n")
            
            lowest = memory_peaks[0]
            highest = memory_peaks[-1]
            f.write(f"\n最低内存使用 **{lowest[0]}** 比最高 **{highest[0]}** 少 {(highest[1]-lowest[1]):.1f}MB ({(highest[1]/lowest[1]-1)*100:.1f}% 节省)。\n\n")
            
            # CPU使用率分析
            f.write("## CPU使用率分析\n\n")
            cpu_avgs = [(r.test_name, r.metrics.cpu_percent_avg) for r in successful_results]
            cpu_avgs.sort(key=lambda x: x[1], reverse=True)
            
            f.write("CPU使用率排名（从高到低）：\n")
            for i, (name, cpu) in enumerate(cpu_avgs, 1):
                f.write(f"{i}. **{name}**: {cpu:.1f}%\n")
            
            # 性能效率分析
            f.write("## 性能效率分析\n\n")
            
            efficiency_scores = []
            for r in successful_results:
                # 效率分数：1/(时间*内存)
                efficiency = 1.0 / (r.metrics.execution_time * r.metrics.memory_peak_mb)
                efficiency_scores.append((r.test_name, efficiency))
            
            efficiency_scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write("性能效率排名（时间*内存的倒数，越高越好）：\n")
            for i, (name, efficiency) in enumerate(efficiency_scores, 1):
                f.write(f"{i}. **{name}**: {efficiency:.6f}\n")
            
            # 优化建议
            f.write("## 优化建议\n\n")
            
            best_efficiency = efficiency_scores[0]
            worst_efficiency = efficiency_scores[-1]
            
            f.write(f"基于效率分析，建议：\n")
            f.write(f"1. **参考 {best_efficiency[0]} 的配置**: 该测试在时间和内存使用上达到了最佳平衡。\n")
            f.write(f"2. **优化 {worst_efficiency[0]} 的配置**: 该测试在效率上有较大改进空间。\n")
            
            # 检查资源瓶颈
            high_memory_users = [name for name, memory in memory_peaks if memory > 1000]
            if high_memory_users:
                f.write(f"3. **内存优化**: 以下测试内存使用超过1GB，建议优化：{', '.join(high_memory_users)}\n")
            
            long_running_tests = [name for name, exec_time in execution_times if exec_time > 300]
            if long_running_tests:
                f.write(f"4. **时间优化**: 以下测试执行时间超过5分钟，建议优化：{', '.join(long_running_tests)}\n")
            
            # GPU分析（如果有GPU数据）
            gpu_results = [r for r in successful_results if r.metrics.gpu_memory_peak_mb is not None]
            if gpu_results:
                f.write("## GPU使用分析\n\n")
                gpu_memory_peaks = [(r.test_name, r.metrics.gpu_memory_peak_mb) for r in gpu_results]
                gpu_memory_peaks.sort(key=lambda x: x[1])
                
                f.write("GPU内存使用排名（从少到多）：\n")
                for i, (name, memory) in enumerate(gpu_memory_peaks, 1):
                    f.write(f"{i}. **{name}**: {memory:.1f}MB\n")
            
            f.write("\n---\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="重构脚本性能基准测试")
    parser.add_argument("--script", type=str,
                       default=str(DEFAULT_REFACTORED_SCRIPT),
                       help="重构脚本路径")
    parser.add_argument("--config", type=str,
                       default=str(DEFAULT_CONFIG),
                       help="配置文件路径")
    parser.add_argument("--output", type=str,
                       default="benchmark_results",
                       help="输出目录")
    parser.add_argument("--format", choices=["text", "json", "all"], default="all",
                       help="输出格式")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式（只运行轻量级测试）")
    
    args = parser.parse_args()
    
    print("开始重构脚本性能基准测试...")
    
    # 创建基准测试运行器
    runner = BenchmarkRunner(args.script, args.config)
    
    # 运行基准测试
    results = runner.run_all_benchmarks()
    
    # 生成报告
    output_dir = Path(args.output)
    reporter = BenchmarkReporter(results)
    reporter.generate_report(output_dir)
    
    # 打印总结
    successful_count = sum(1 for r in results if r.status == "SUCCESS")
    total_count = len(results)
    
    print(f"\n基准测试完成!")
    print(f"成功测试: {successful_count}/{total_count}")
    print(f"结果保存在: {output_dir}")
    
    # 返回适当的退出码
    sys.exit(0 if successful_count == total_count else 1)
