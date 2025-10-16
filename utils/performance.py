"""性能分析模块

提供模型性能分析功能：
- 参数量统计
- FLOPs计算
- 显存使用分析
- 推理延迟测试
- 资源使用报告
"""

import time
import psutil
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
from contextlib import contextmanager
import gc


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """重置统计信息"""
        self.stats = {
            'memory': [],
            'timing': [],
            'flops': {},
            'params': {}
        }
    
    @contextmanager
    def profile_memory(self, tag: str = "default"):
        """内存使用分析上下文管理器"""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
            
            yield
            
            peak_memory = torch.cuda.max_memory_allocated()
            end_memory = torch.cuda.memory_allocated()
            
            self.stats['memory'].append({
                'tag': tag,
                'start_mb': start_memory / 1024**2,
                'peak_mb': peak_memory / 1024**2,
                'end_mb': end_memory / 1024**2,
                'allocated_mb': (end_memory - start_memory) / 1024**2
            })
        else:
            # CPU内存监控
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2
            
            yield
            
            end_memory = process.memory_info().rss / 1024**2
            
            self.stats['memory'].append({
                'tag': tag,
                'start_mb': start_memory,
                'peak_mb': end_memory,  # 简化处理
                'end_mb': end_memory,
                'allocated_mb': end_memory - start_memory
            })
    
    @contextmanager
    def profile_time(self, tag: str = "default"):
        """时间分析上下文管理器"""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        yield
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        self.stats['timing'].append({
            'tag': tag,
            'duration_ms': (end_time - start_time) * 1000
        })
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                     num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Any]:
        """全面的模型性能分析
        
        Args:
            model: 待分析的模型
            input_shape: 输入形状 (B, C, H, W)
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            性能分析报告
        """
        model.eval()
        device = next(model.parameters()).device
        
        # 创建测试输入
        dummy_input = torch.randn(input_shape, device=device)
        
        # 参数量统计
        params_info = self.count_parameters(model)
        
        # FLOPs计算
        flops_info = self.calculate_flops(model, dummy_input)
        
        # 内存使用分析
        memory_info = self.analyze_memory_usage(model, dummy_input)
        
        # 推理延迟测试
        latency_info = self.measure_inference_latency(
            model, dummy_input, num_runs, warmup_runs
        )
        
        return {
            'parameters': params_info,
            'flops': flops_info,
            'memory': memory_info,
            'latency': latency_info,
            'input_shape': input_shape,
            'device': str(device)
        }
    
    def count_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """统计模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 按模块统计
        module_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    module_params[name] = params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params,
            'total_mb': total_params * 4 / 1024**2,  # 假设float32
            'by_module': module_params
        }
    
    def calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """计算模型FLOPs"""
        try:
            from fvcore.nn import FlopCountMode, flop_count
            
            # 使用fvcore计算FLOPs
            flops_dict, _ = flop_count(model, (input_tensor,), 
                                     supported_ops=None)
            
            total_flops = sum(flops_dict.values())
            
            return {
                'total': total_flops,
                'total_gflops': total_flops / 1e9,
                'by_operation': flops_dict
            }
        except ImportError:
            # 简化的FLOPs估算
            return self._estimate_flops_simple(model, input_tensor)
    
    def _estimate_flops_simple(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """简化的FLOPs估算"""
        total_flops = 0
        B, C, H, W = input_tensor.shape
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 卷积层FLOPs
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = H * W * module.out_channels  # 简化假设
                flops = kernel_flops * output_elements * B
                total_flops += flops
            
            elif isinstance(module, nn.Linear):
                # 全连接层FLOPs
                flops = module.in_features * module.out_features * B
                total_flops += flops
        
        return {
            'total': total_flops,
            'total_gflops': total_flops / 1e9,
            'estimation_method': 'simplified'
        }
    
    def analyze_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """分析内存使用"""
        with self.profile_memory("model_forward"):
            with torch.no_grad():
                _ = model(input_tensor)
        
        memory_stats = self.stats['memory'][-1]
        
        # 模型参数内存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # 输入内存
        input_memory = input_tensor.numel() * input_tensor.element_size() / 1024**2
        
        return {
            'forward_pass_mb': memory_stats['allocated_mb'],
            'peak_memory_mb': memory_stats['peak_mb'],
            'model_parameters_mb': param_memory,
            'input_tensor_mb': input_memory,
            'total_estimated_mb': param_memory + input_memory + memory_stats['allocated_mb']
        }
    
    def measure_inference_latency(self, model: nn.Module, input_tensor: torch.Tensor,
                                 num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Any]:
        """测量推理延迟"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # 测量延迟
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                with self.profile_time("inference"):
                    _ = model(input_tensor)
                latencies.append(self.stats['timing'][-1]['duration_ms'])
        
        latencies = np.array(latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'num_runs': num_runs,
            'warmup_runs': warmup_runs
        }
    
    def generate_report(self, results: Dict[str, Any], save_path: Optional[Path] = None) -> str:
        """生成性能分析报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MODEL PERFORMANCE ANALYSIS REPORT")
        report_lines.append("=" * 60)
        
        # 基本信息
        report_lines.append(f"\nInput Shape: {results['input_shape']}")
        report_lines.append(f"Device: {results['device']}")
        
        # 参数量信息
        params = results['parameters']
        report_lines.append(f"\n📊 PARAMETERS:")
        report_lines.append(f"  Total Parameters: {params['total']:,} ({params['total_mb']:.2f} MB)")
        report_lines.append(f"  Trainable: {params['trainable']:,}")
        report_lines.append(f"  Non-trainable: {params['non_trainable']:,}")
        
        # FLOPs信息
        flops = results['flops']
        report_lines.append(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
        report_lines.append(f"  Total FLOPs: {flops['total']:,} ({flops['total_gflops']:.2f} GFLOPs)")
        
        # 内存使用
        memory = results['memory']
        report_lines.append(f"\n💾 MEMORY USAGE:")
        report_lines.append(f"  Model Parameters: {memory['model_parameters_mb']:.2f} MB")
        report_lines.append(f"  Forward Pass: {memory['forward_pass_mb']:.2f} MB")
        report_lines.append(f"  Peak Memory: {memory['peak_memory_mb']:.2f} MB")
        report_lines.append(f"  Total Estimated: {memory['total_estimated_mb']:.2f} MB")
        
        # 延迟信息
        latency = results['latency']
        report_lines.append(f"\n⏱️  INFERENCE LATENCY:")
        report_lines.append(f"  Mean: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
        report_lines.append(f"  Median: {latency['median_ms']:.2f} ms")
        report_lines.append(f"  Min/Max: {latency['min_ms']:.2f} / {latency['max_ms']:.2f} ms")
        report_lines.append(f"  P95/P99: {latency['p95_ms']:.2f} / {latency['p99_ms']:.2f} ms")
        report_lines.append(f"  Runs: {latency['num_runs']} (warmup: {latency['warmup_runs']})")
        
        # 效率指标
        throughput = 1000 / latency['mean_ms']  # samples per second
        efficiency = flops['total_gflops'] / latency['mean_ms'] * 1000  # GFLOPs per second
        
        report_lines.append(f"\n📈 EFFICIENCY METRICS:")
        report_lines.append(f"  Throughput: {throughput:.2f} samples/sec")
        report_lines.append(f"  Computational Efficiency: {efficiency:.2f} GFLOPs/sec")
        report_lines.append(f"  Memory Efficiency: {params['total'] / memory['peak_memory_mb'] / 1024**2:.2f} params/MB")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def save_detailed_results(self, results: Dict[str, Any], save_path: Path):
        """保存详细的性能分析结果"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式的详细结果
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存文本报告
        report = self.generate_report(results)
        with open(save_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write(report)


def benchmark_models(models: Dict[str, nn.Module], input_shape: Tuple[int, ...],
                    save_dir: Path, device: str = 'cuda') -> Dict[str, Dict]:
    """批量测试多个模型的性能
    
    Args:
        models: 模型字典 {name: model}
        input_shape: 输入形状
        save_dir: 保存目录
        device: 设备
        
    Returns:
        所有模型的性能结果
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    profiler = PerformanceProfiler(device)
    
    all_results = {}
    
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        
        # 移动模型到指定设备
        model = model.to(device)
        
        # 性能分析
        results = profiler.profile_model(model, input_shape)
        all_results[name] = results
        
        # 保存单个模型结果
        model_save_path = save_dir / f"{name}_performance"
        profiler.save_detailed_results(results, model_save_path)
        
        # 清理内存
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 生成对比报告
    comparison_report = generate_comparison_report(all_results)
    with open(save_dir / "comparison_report.txt", 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    # 保存汇总结果
    with open(save_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def generate_comparison_report(results: Dict[str, Dict]) -> str:
    """生成模型对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 80)
    
    # 创建对比表格
    models = list(results.keys())
    
    # 参数量对比
    report_lines.append(f"\n📊 PARAMETERS COMPARISON:")
    report_lines.append(f"{'Model':<20} {'Total Params':<15} {'Size (MB)':<12} {'Trainable':<15}")
    report_lines.append("-" * 65)
    
    for model in models:
        params = results[model]['parameters']
        report_lines.append(
            f"{model:<20} {params['total']:>14,} {params['total_mb']:>11.2f} "
            f"{params['trainable']:>14,}"
        )
    
    # FLOPs对比
    report_lines.append(f"\n⚡ COMPUTATIONAL COMPLEXITY COMPARISON:")
    report_lines.append(f"{'Model':<20} {'GFLOPs':<12} {'Efficiency':<15}")
    report_lines.append("-" * 50)
    
    for model in models:
        flops = results[model]['flops']
        latency = results[model]['latency']
        efficiency = flops['total_gflops'] / latency['mean_ms'] * 1000
        report_lines.append(
            f"{model:<20} {flops['total_gflops']:>11.2f} {efficiency:>14.2f}"
        )
    
    # 内存使用对比
    report_lines.append(f"\n💾 MEMORY USAGE COMPARISON:")
    report_lines.append(f"{'Model':<20} {'Peak (MB)':<12} {'Forward (MB)':<15} {'Total (MB)':<12}")
    report_lines.append("-" * 65)
    
    for model in models:
        memory = results[model]['memory']
        report_lines.append(
            f"{model:<20} {memory['peak_memory_mb']:>11.2f} "
            f"{memory['forward_pass_mb']:>14.2f} {memory['total_estimated_mb']:>11.2f}"
        )
    
    # 延迟对比
    report_lines.append(f"\n⏱️  INFERENCE LATENCY COMPARISON:")
    report_lines.append(f"{'Model':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput':<15}")
    report_lines.append("-" * 65)
    
    for model in models:
        latency = results[model]['latency']
        throughput = 1000 / latency['mean_ms']
        report_lines.append(
            f"{model:<20} {latency['mean_ms']:>11.2f} {latency['std_ms']:>11.2f} "
            f"{throughput:>14.2f}"
        )
    
    # 综合排名
    report_lines.append(f"\n🏆 OVERALL RANKING:")
    
    # 计算综合得分（简化版本）
    scores = {}
    for model in models:
        params_score = 1 / (results[model]['parameters']['total'] / 1e6)  # 参数越少越好
        flops_score = 1 / results[model]['flops']['total_gflops']  # FLOPs越少越好
        latency_score = 1 / results[model]['latency']['mean_ms']  # 延迟越低越好
        memory_score = 1 / results[model]['memory']['peak_memory_mb']  # 内存越少越好
        
        # 加权平均（可调整权重）
        total_score = (params_score * 0.2 + flops_score * 0.3 + 
                      latency_score * 0.3 + memory_score * 0.2)
        scores[model] = total_score
    
    # 按得分排序
    ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(ranked_models, 1):
        report_lines.append(f"  {i}. {model} (Score: {score:.4f})")
    
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def profile_training_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                         loss_fn: callable, input_batch: torch.Tensor,
                         target_batch: torch.Tensor) -> Dict[str, float]:
    """分析单个训练步骤的性能"""
    profiler = PerformanceProfiler()
    
    # 前向传播
    with profiler.profile_time("forward"):
        with profiler.profile_memory("forward"):
            output = model(input_batch)
            loss = loss_fn(output, target_batch)
    
    # 反向传播
    with profiler.profile_time("backward"):
        with profiler.profile_memory("backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 汇总结果
    timing_stats = profiler.stats['timing']
    memory_stats = profiler.stats['memory']
    
    return {
        'forward_time_ms': timing_stats[0]['duration_ms'],
        'backward_time_ms': timing_stats[1]['duration_ms'],
        'total_time_ms': sum(t['duration_ms'] for t in timing_stats),
        'forward_memory_mb': memory_stats[0]['allocated_mb'],
        'backward_memory_mb': memory_stats[1]['allocated_mb'],
        'peak_memory_mb': max(m['peak_mb'] for m in memory_stats),
        'loss_value': loss.item()
    }