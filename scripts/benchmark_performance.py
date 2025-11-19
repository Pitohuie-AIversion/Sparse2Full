"""
性能基准测试和验证脚本
用于评估训练脚本和模型在各种配置下的性能表现
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import psutil

# 导入项目模块
from src.models.swin_temporal_nar import SwinTemporalNAR, SwinTemporalConfig
from src.data.pdebench_dataset import PDEBenchDataset, PDEBenchDataModule
from src.optimizers.hardware_profiler import HardwareProfiler
from src.optimizers.mixed_precision_trainer import MixedPrecisionTrainer
from src.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 模型配置
    model_name: str = "SwinTemporalNAR"
    input_channels: int = 1
    hidden_dim: int = 96
    num_layers: int = 4
    num_heads: int = 8
    window_size: int = 7
    
    # 数据配置
    batch_size: int = 32
    time_steps: int = 10
    prediction_steps: int = 5
    spatial_resolution: Tuple[int, int] = (64, 64)
    
    # 训练配置
    num_epochs: int = 3
    learning_rate: float = 1e-3
    mixed_precision: bool = True
    compile_model: bool = False
    
    # 硬件配置
    device: str = "auto"  # auto, cpu, cuda, multi_gpu
    num_workers: int = 4
    pin_memory: bool = True
    
    # 测试配置
    warmup_steps: int = 10
    benchmark_steps: int = 100
    profile_memory: bool = True
    profile_throughput: bool = True
    profile_latency: bool = True

class PerformanceBenchmark:
    """
    性能基准测试器
    用于评估模型和训练流程的性能表现
    """
    
    def __init__(self, config: BenchmarkConfig, output_dir: str = "benchmark_results"):
        """
        初始化基准测试器
        
        Args:
            config: 基准测试配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.device = self._setup_device()
        self.model = None
        self.data_loader = None
        self.optimizer = None
        self.criterion = None
        
        # 性能监控
        self.monitor = PerformanceMonitor(
            log_dir=str(self.output_dir / "logs"),
            monitoring_interval=0.5
        )
        
        # 硬件分析器
        self.hardware_profiler = HardwareProfiler()
        
        # 测试结果
        self.results = {
            'config': asdict(config),
            'system_info': self._get_system_info(),
            'hardware_profile': self.hardware_profiler.get_optimal_config(),
            'benchmarks': {}
        }
        
        logger.info(f"性能基准测试器初始化完成: output_dir={self.output_dir}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("使用CPU设备")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import cpuinfo
            import platform
            
            cpu_info = cpuinfo.get_cpu_info()
            memory = psutil.virtual_memory()
            
            system_info = {
                'platform': platform.platform(),
                'processor': cpu_info.get('brand_raw', 'Unknown'),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'total_memory_gb': memory.total / (1024**3),
                'python_version': platform.python_version(),
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            if torch.cuda.is_available():
                system_info['gpu_name'] = torch.cuda.get_device_name()
                system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return system_info
            
        except Exception as e:
            logger.warning(f"获取系统信息失败: {e}")
            return {}
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        if self.config.model_name == "SwinTemporalNAR":
            model_config = SwinTemporalConfig(
                input_channels=self.config.input_channels,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                window_size=self.config.window_size,
                time_steps=self.config.time_steps,
                prediction_steps=self.config.prediction_steps,
                spatial_resolution=self.config.spatial_resolution
            )
            model = SwinTemporalNAR(model_config)
        else:
            raise ValueError(f"不支持的模型: {self.config.model_name}")
        
        return model.to(self.device)
    
    def _create_synthetic_data(self) -> torch.utils.data.DataLoader:
        """创建合成数据用于基准测试"""
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, config):
                self.config = config
                self.length = 1000
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                # 创建合成数据
                input_data = torch.randn(
                    self.config.time_steps,
                    self.config.input_channels,
                    *self.config.spatial_resolution
                )
                target_data = torch.randn(
                    self.config.prediction_steps,
                    self.config.input_channels,
                    *self.config.spatial_resolution
                )
                return input_data, target_data
        
        dataset = SyntheticDataset(self.config)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        return data_loader
    
    def benchmark_model_forward(self) -> Dict[str, Any]:
        """基准测试模型前向传播"""
        logger.info("开始模型前向传播基准测试")
        
        if self.model is None:
            self.model = self._create_model()
        
        # 创建测试数据
        batch_size = self.config.batch_size
        time_steps = self.config.time_steps
        channels = self.config.input_channels
        height, width = self.config.spatial_resolution
        
        test_input = torch.randn(
            batch_size, time_steps, channels, height, width
        ).to(self.device)
        
        # 预热
        logger.info(f"预热 {self.config.warmup_steps} 步...")
        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(self.config.warmup_steps), desc="Warmup"):
                _ = self.model(test_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # 基准测试
        logger.info(f"基准测试 {self.config.benchmark_steps} 步...")
        forward_times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in tqdm(range(self.config.benchmark_steps), desc="Benchmark Forward"):
                # 记录内存使用
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_before = torch.cuda.memory_allocated()
                
                start_time = time.time()
                output = self.model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage.append(memory_after - memory_before)
                
                end_time = time.time()
                forward_times.append(end_time - start_time)
        
        # 计算统计信息
        forward_times = np.array(forward_times)
        memory_usage = np.array(memory_usage) if memory_usage else np.array([0])
        
        results = {
            'test_type': 'model_forward',
            'batch_size': batch_size,
            'input_shape': test_input.shape,
            'output_shape': output.shape if 'output' in locals() else None,
            'forward_times': {
                'mean': float(np.mean(forward_times)),
                'std': float(np.std(forward_times)),
                'min': float(np.min(forward_times)),
                'max': float(np.max(forward_times)),
                'median': float(np.median(forward_times)),
                'percentile_95': float(np.percentile(forward_times, 95)),
                'percentile_99': float(np.percentile(forward_times, 99))
            },
            'throughput': {
                'samples_per_second': batch_size / np.mean(forward_times),
                'inference_per_second': 1.0 / np.mean(forward_times)
            },
            'memory_usage': {
                'mean_mb': float(np.mean(memory_usage) / (1024**2)),
                'peak_mb': float(np.max(memory_usage) / (1024**2))
            },
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        }
        
        logger.info(f"模型前向传播基准测试完成: "
                   f"平均时间={results['forward_times']['mean']*1000:.2f}ms, "
                   f"吞吐量={results['throughput']['samples_per_second']:.1f} samples/s")
        
        return results
    
    def benchmark_training_step(self) -> Dict[str, Any]:
        """基准测试训练步骤"""
        logger.info("开始训练步骤基准测试")
        
        if self.model is None:
            self.model = self._create_model()
        
        if self.data_loader is None:
            self.data_loader = self._create_synthetic_data()
        
        # 创建优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # 混合精度训练
        if self.config.mixed_precision and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        # 模型编译（如果支持）
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("模型编译已启用")
            except Exception as e:
                logger.warning(f"模型编译失败: {e}")
        
        # 预热
        logger.info(f"预热 {self.config.warmup_steps} 步...")
        self.model.train()
        
        data_iter = iter(self.data_loader)
        for _ in tqdm(range(self.config.warmup_steps), desc="Warmup"):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                inputs, targets = next(data_iter)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        # 基准测试
        logger.info(f"基准测试 {self.config.benchmark_steps} 步...")
        step_times = []
        loss_values = []
        memory_usage = []
        
        data_iter = iter(self.data_loader)
        for _ in tqdm(range(self.config.benchmark_steps), desc="Benchmark Training"):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                inputs, targets = next(data_iter)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 记录内存使用
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            start_time = time.time()
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
            
            end_time = time.time()
            step_times.append(end_time - start_time)
            loss_values.append(loss.item())
        
        # 计算统计信息
        step_times = np.array(step_times)
        loss_values = np.array(loss_values)
        memory_usage = np.array(memory_usage) if memory_usage else np.array([0])
        
        results = {
            'test_type': 'training_step',
            'batch_size': self.config.batch_size,
            'mixed_precision': self.config.mixed_precision,
            'compile_model': self.config.compile_model,
            'step_times': {
                'mean': float(np.mean(step_times)),
                'std': float(np.std(step_times)),
                'min': float(np.min(step_times)),
                'max': float(np.max(step_times)),
                'median': float(np.median(step_times)),
                'percentile_95': float(np.percentile(step_times, 95)),
                'percentile_99': float(np.percentile(step_times, 99))
            },
            'loss_values': {
                'mean': float(np.mean(loss_values)),
                'std': float(np.std(loss_values)),
                'min': float(np.min(loss_values)),
                'max': float(np.max(loss_values))
            },
            'throughput': {
                'samples_per_second': self.config.batch_size / np.mean(step_times),
                'steps_per_second': 1.0 / np.mean(step_times)
            },
            'memory_usage': {
                'mean_mb': float(np.mean(memory_usage) / (1024**2)),
                'peak_mb': float(np.max(memory_usage) / (1024**2))
            }
        }
        
        logger.info(f"训练步骤基准测试完成: "
                   f"平均时间={results['step_times']['mean']*1000:.2f}ms, "
                   f"吞吐量={results['throughput']['samples_per_second']:.1f} samples/s, "
                   f"平均损失={results['loss_values']['mean']:.6f}")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """基准测试内存使用情况"""
        logger.info("开始内存使用基准测试")
        
        if self.model is None:
            self.model = self._create_model()
        
        # 记录初始内存状态
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            initial_memory_reserved = torch.cuda.memory_reserved()
        else:
            initial_memory = 0
            initial_memory_reserved = 0
        
        # 模型内存使用
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        # 创建不同批大小的测试数据
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        memory_results = []
        
        for batch_size in tqdm(batch_sizes, desc="Memory Benchmark"):
            try:
                # 创建测试输入
                test_input = torch.randn(
                    batch_size, 
                    self.config.time_steps,
                    self.config.input_channels,
                    *self.config.spatial_resolution
                ).to(self.device)
                
                # 记录内存使用
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_before_forward = torch.cuda.memory_allocated()
                
                # 前向传播
                with torch.no_grad():
                    output = self.model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_after_forward = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    
                    memory_results.append({
                        'batch_size': batch_size,
                        'input_shape': list(test_input.shape),
                        'output_shape': list(output.shape),
                        'memory_before_mb': memory_before_forward / (1024**2),
                        'memory_after_mb': memory_after_forward / (1024**2),
                        'memory_increase_mb': (memory_after_forward - memory_before_forward) / (1024**2),
                        'peak_memory_mb': peak_memory / (1024**2)
                    })
                else:
                    memory_results.append({
                        'batch_size': batch_size,
                        'input_shape': list(test_input.shape),
                        'output_shape': list(output.shape),
                        'memory_before_mb': 0,
                        'memory_after_mb': 0,
                        'memory_increase_mb': 0,
                        'peak_memory_mb': 0
                    })
                
                # 清理
                del test_input, output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"批大小 {batch_size} 超出内存限制")
                    break
                else:
                    raise
        
        # 计算内存增长趋势
        if len(memory_results) > 1:
            batch_sizes_tested = [r['batch_size'] for r in memory_results]
            memory_increases = [r['memory_increase_mb'] for r in memory_results]
            
            # 线性拟合
            coeffs = np.polyfit(batch_sizes_tested, memory_increases, 1)
            memory_per_sample_mb = coeffs[0]
            base_memory_mb = coeffs[1]
        else:
            memory_per_sample_mb = 0
            base_memory_mb = 0
        
        results = {
            'test_type': 'memory_usage',
            'initial_memory_mb': initial_memory / (1024**2),
            'initial_memory_reserved_mb': initial_memory_reserved / (1024**2),
            'model_memory_mb': model_memory / (1024**2),
            'memory_results': memory_results,
            'memory_analysis': {
                'memory_per_sample_mb': memory_per_sample_mb,
                'base_memory_mb': base_memory_mb,
                'max_tested_batch_size': max([r['batch_size'] for r in memory_results]) if memory_results else 0
            }
        }
        
        logger.info(f"内存使用基准测试完成: "
                   f"模型内存={results['model_memory_mb']:.1f}MB, "
                   f"每样本内存={memory_per_sample_mb:.2f}MB")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合基准测试"""
        logger.info("开始综合基准测试")
        
        # 启动性能监控
        self.monitor.start_monitoring()
        
        try:
            # 运行各个基准测试
            benchmarks = {}
            
            # 1. 模型前向传播基准测试
            logger.info("=== 模型前向传播基准测试 ===")
            benchmarks['model_forward'] = self.benchmark_model_forward()
            
            # 2. 训练步骤基准测试
            logger.info("=== 训练步骤基准测试 ===")
            benchmarks['training_step'] = self.benchmark_training_step()
            
            # 3. 内存使用基准测试
            logger.info("=== 内存使用基准测试 ===")
            benchmarks['memory_usage'] = self.benchmark_memory_usage()
            
            # 4. 获取性能监控摘要
            performance_summary = self.monitor.get_performance_summary()
            
            # 综合结果
            comprehensive_results = {
                'timestamp': time.time(),
                'config': asdict(self.config),
                'system_info': self.results['system_info'],
                'hardware_profile': self.results['hardware_profile'],
                'benchmarks': benchmarks,
                'performance_summary': performance_summary
            }
            
            # 保存结果
            self.save_benchmark_results(comprehensive_results)
            
            # 生成报告
            self.generate_benchmark_report(comprehensive_results)
            
            logger.info("综合基准测试完成")
            return comprehensive_results
            
        finally:
            # 停止性能监控
            self.monitor.stop_monitoring()
            self.monitor.cleanup()
    
    def save_benchmark_results(self, results: Dict[str, Any]):
        """保存基准测试结果"""
        results_file = self.output_dir / 'benchmark_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"基准测试结果已保存: {results_file}")
    
    def generate_benchmark_report(self, results: Dict[str, Any]):
        """生成基准测试报告"""
        report_file = self.output_dir / 'benchmark_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# PDEBench 性能基准测试报告\n\n")
            f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 系统信息
            f.write("## 系统信息\n\n")
            system_info = results['system_info']
            f.write(f"- **平台**: {system_info.get('platform', 'Unknown')}\n")
            f.write(f"- **处理器**: {system_info.get('processor', 'Unknown')}\n")
            f.write(f"- **CPU核心数**: {system_info.get('cpu_cores', 0)}\n")
            f.write(f"- **CPU线程数**: {system_info.get('cpu_threads', 0)}\n")
            f.write(f"- **总内存**: {system_info.get('total_memory_gb', 0):.1f} GB\n")
            f.write(f"- **PyTorch版本**: {system_info.get('pytorch_version', 'Unknown')}\n")
            f.write(f"- **CUDA可用**: {system_info.get('cuda_available', False)}\n")
            
            if system_info.get('cuda_available'):
                f.write(f"- **GPU名称**: {system_info.get('gpu_name', 'Unknown')}\n")
                f.write(f"- **GPU内存**: {system_info.get('gpu_memory_gb', 0):.1f} GB\n")
            
            f.write("\n")
            
            # 模型前向传播结果
            f.write("## 模型前向传播性能\n\n")
            forward_results = results['benchmarks']['model_forward']
            times = forward_results['forward_times']
            throughput = forward_results['throughput']
            
            f.write(f"- **平均前向时间**: {times['mean']*1000:.2f} ms\n")
            f.write(f"- **前向时间标准差**: {times['std']*1000:.2f} ms\n")
            f.write(f"- **95百分位数**: {times['percentile_95']*1000:.2f} ms\n")
            f.write(f"- **99百分位数**: {times['percentile_99']*1000:.2f} ms\n")
            f.write(f"- **样本吞吐量**: {throughput['samples_per_second']:.1f} samples/s\n")
            f.write(f"- **模型参数量**: {forward_results['model_parameters']:,}\n")
            f.write(f"- **模型大小**: {forward_results['model_size_mb']:.1f} MB\n\n")
            
            # 训练步骤结果
            f.write("## 训练步骤性能\n\n")
            training_results = results['benchmarks']['training_step']
            step_times = training_results['step_times']
            training_throughput = training_results['throughput']
            
            f.write(f"- **平均训练时间**: {step_times['mean']*1000:.2f} ms\n")
            f.write(f"- **训练时间标准差**: {step_times['std']*1000:.2f} ms\n")
            f.write(f"- **95百分位数**: {step_times['percentile_95']*1000:.2f} ms\n")
            f.write(f"- **99百分位数**: {step_times['percentile_99']*1000:.2f} ms\n")
            f.write(f"- **训练吞吐量**: {training_throughput['samples_per_second']:.1f} samples/s\n")
            f.write(f"- **平均损失**: {training_results['loss_values']['mean']:.6f}\n\n")
            
            # 内存使用结果
            f.write("## 内存使用分析\n\n")
            memory_results = results['benchmarks']['memory_usage']
            memory_analysis = memory_results['memory_analysis']
            
            f.write(f"- **模型内存占用**: {memory_results['model_memory_mb']:.1f} MB\n")
            f.write(f"- **每样本内存增长**: {memory_analysis['memory_per_sample_mb']:.2f} MB\n")
            f.write(f"- **基础内存占用**: {memory_analysis['base_memory_mb']:.1f} MB\n")
            f.write(f"- **最大测试批大小**: {memory_analysis['max_tested_batch_size']}\n\n")
            
            # 性能摘要
            f.write("## 性能摘要\n\n")
            perf_summary = results['performance_summary']
            
            if 'avg_loss' in perf_summary:
                f.write(f"- **平均损失**: {perf_summary['avg_loss']:.6f}\n")
            if 'avg_throughput' in perf_summary:
                f.write(f"- **平均吞吐量**: {perf_summary['avg_throughput']:.1f} samples/s\n")
            if 'avg_cpu_usage' in perf_summary:
                f.write(f"- **平均CPU使用率**: {perf_summary['avg_cpu_usage']:.1f}%\n")
            if 'avg_gpu_usage' in perf_summary:
                f.write(f"- **平均GPU使用率**: {perf_summary['avg_gpu_usage']:.1f}%\n")
            
            f.write("\n")
            
            # 优化建议
            f.write("## 优化建议\n\n")
            f.write("基于基准测试结果，建议以下优化策略：\n\n")
            
            if step_times['mean'] > 0.1:  # 训练时间超过100ms
                f.write("1. **训练时间过长**: 考虑减少模型复杂度或增加批大小\n")
            
            if memory_analysis['memory_per_sample_mb'] > 100:  # 每样本内存超过100MB
                f.write("2. **内存使用较高**: 考虑使用梯度累积或模型分片技术\n")
            
            if training_throughput['samples_per_second'] < 100:  # 吞吐量低于100 samples/s
                f.write("3. **吞吐量较低**: 考虑使用混合精度训练或数据加载优化\n")
            
            f.write("\n")
        
        logger.info(f"基准测试报告已生成: {report_file}")
    
    def cleanup(self):
        """清理资源"""
        if self.monitor:
            self.monitor.cleanup()
        
        # 清理GPU内存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 垃圾回收
        gc.collect()
        
        logger.info("基准测试器清理完成")

def run_benchmark_suite(
    output_dir: str = "benchmark_results",
    batch_sizes: List[int] = [8, 16, 32, 64],
    mixed_precision_options: List[bool] = [False, True],
    model_configs: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """运行基准测试套件"""
    
    if model_configs is None:
        model_configs = [
            {'hidden_dim': 64, 'num_layers': 2},
            {'hidden_dim': 96, 'num_layers': 4},
            {'hidden_dim': 128, 'num_layers': 6}
        ]
    
    all_results = []
    
    for batch_size in batch_sizes:
        for mixed_precision in mixed_precision_options:
            for model_config in model_configs:
                logger.info(f"运行基准测试: batch_size={batch_size}, "
                           f"mixed_precision={mixed_precision}, "
                           f"model_config={model_config}")
                
                # 创建配置
                config = BenchmarkConfig(
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    **model_config
                )
                
                # 运行基准测试
                benchmark = PerformanceBenchmark(config, output_dir)
                results = benchmark.run_comprehensive_benchmark()
                all_results.append(results)
                
                # 清理资源
                benchmark.cleanup()
                
                # 短暂休息避免过热
                time.sleep(10)
    
    # 生成汇总报告
    summary_file = Path(output_dir) / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_benchmarks': len(all_results),
            'configurations': [
                {
                    'batch_size': r['config']['batch_size'],
                    'mixed_precision': r['config']['mixed_precision'],
                    'hidden_dim': r['config']['hidden_dim'],
                    'num_layers': r['config']['num_layers']
                }
                for r in all_results
            ],
            'results': all_results
        }, f, indent=2, default=str)
    
    logger.info(f"基准测试套件完成: {len(all_results)} 个配置已测试")
    return all_results

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行基准测试
    results = run_benchmark_suite(
        output_dir="benchmark_results",
        batch_sizes=[16, 32],
        mixed_precision_options=[True],
        model_configs=[
            {'hidden_dim': 96, 'num_layers': 4}
        ]
    )
    
    logger.info(f"基准测试完成，共测试了 {len(results)} 个配置")