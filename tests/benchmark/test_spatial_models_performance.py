"""
Performance benchmarking suite for spatial prediction models.
Measures inference speed, memory usage, and computational efficiency.
"""

import pytest
import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Import spatial models
from models import (
    UNet, UNetPlusPlus, FNO2d, UFNOUNet,
    SegFormer, UNetFormer, SegFormerUNetFormer,
    VisionTransformer, SwinTransformerTiny, Transformer,
    SwinUNet, HybridModel, MLPModel, MLPMixer, LIIFModel,
    SparseAttentionEncoder, SparseSwinUNet
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    model_name: str
    input_shape: Tuple[int, int, int, int]
    inference_time_ms: float
    memory_usage_mb: float
    throughput_fps: float
    flops_g: float
    params_m: float
    efficiency_score: float


class SpatialModelBenchmark:
    """Benchmark suite for spatial models"""
    
    def __init__(self, device: str = 'auto', warmup_runs: int = 5, benchmark_runs: int = 20):
        self.device = self._get_device(device)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[PerformanceMetrics] = []
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for benchmarking"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def get_model_configs(self) -> Dict[type, Dict]:
        """Get optimized configurations for each model type"""
        configs = {
            UNet: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'features': [32, 64, 128, 256]},
            UNetPlusPlus: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'features': [32, 64, 128, 256]},
            FNO2d: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'modes': 16, 'width': 32, 'layers': 4},
            UFNOUNet: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'modes': 16, 'width': 32, 'layers': 4},
            SwinUNet: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'depths': [2, 2, 2, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 8},
            HybridModel: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'backbone': 'swin', 'fusion': 'concat'},
            MLPModel: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'hidden_dim': 256, 'num_layers': 4},
            MLPMixer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'patch_size': 16, 'hidden_dim': 512, 'num_blocks': 8},
            LIIFModel: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'hidden_dim': 256, 'num_layers': 4},
            SegFormer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'backbone': 'b0'},
            UNetFormer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'backbone': 'resnet50'},
            SegFormerUNetFormer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'segformer_backbone': 'b0', 'unetformer_backbone': 'resnet50'},
            VisionTransformer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'patch_size': 16, 'embed_dim': 768},
            SwinTransformerTiny: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
            Transformer: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'd_model': 256, 'nhead': 8},
            SparseAttentionEncoder: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'embed_dim': 256, 'num_heads': 8, 'sparse_ratio': 0.5},
            SparseSwinUNet: {'in_ch': 3, 'out_ch': 3, 'img_size': 256, 'depths': [2, 2, 2, 2], 'num_heads': [3, 6, 12, 24], 'sparse_ratio': 0.5},
        }
        return configs
    
    def create_model(self, model_class: type, config: Dict) -> nn.Module:
        """Create model instance with given configuration"""
        try:
            model = model_class(**config)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to create {model_class.__name__}: {str(e)}")
            return None
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measure average inference time"""
        # Warm up
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / self.benchmark_runs * 1000  # Convert to ms
        
        return avg_time
    
    def measure_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measure peak memory usage during inference"""
        if self.device.type != 'cuda':
            return 0.0  # CPU memory measurement is more complex
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        
        return peak_memory
    
    def estimate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Estimate FLOPs for the model (simplified estimation)"""
        # This is a simplified FLOP estimation
        # In practice, you'd use specialized tools like thop or ptflops
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Estimate operations per parameter (very rough)
        # This assumes each parameter contributes to roughly 100 operations
        # for a typical forward pass through multiple layers
        estimated_ops = param_count * 100
        
        # Convert to GFLOPs
        gflops = estimated_ops / 1e9
        
        return gflops
    
    def calculate_throughput(self, inference_time_ms: float, batch_size: int) -> float:
        """Calculate throughput in FPS"""
        if inference_time_ms <= 0:
            return 0.0
        
        fps = (batch_size * 1000.0) / inference_time_ms
        return fps
    
    def calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall efficiency score"""
        # Weighted combination of different metrics
        # Higher is better
        
        # Normalize metrics (higher is better)
        throughput_score = min(metrics.throughput_fps / 100.0, 1.0)  # Cap at 100 FPS
        memory_score = max(0.0, 1.0 - (metrics.memory_usage_mb / 8192.0))  # 8GB is max reasonable
        flops_score = min(metrics.flops_g / 50.0, 1.0)  # Cap at 50 GFLOPs
        
        # Weighted average (throughput is most important)
        efficiency = (0.5 * throughput_score + 
                     0.3 * memory_score + 
                     0.2 * flops_score)
        
        return efficiency
    
    def benchmark_model(self, model_class: type, input_shape: Tuple[int, int, int, int], 
                       config: Dict) -> Optional[PerformanceMetrics]:
        """Benchmark a single model configuration"""
        print(f"Benchmarking {model_class.__name__} with input {input_shape}...")
        
        # Create model
        model = self.create_model(model_class, config)
        if model is None:
            return None
        
        try:
            # Create input tensor
            input_tensor = torch.randn(input_shape, device=self.device)
            
            # Measure inference time
            inference_time = self.measure_inference_time(model, input_tensor)
            
            # Measure memory usage
            memory_usage = self.measure_memory_usage(model, input_tensor)
            
            # Calculate throughput
            batch_size = input_shape[0]
            throughput = self.calculate_throughput(inference_time, batch_size)
            
            # Estimate FLOPs
            flops = self.estimate_flops(model, input_tensor)
            
            # Get parameter count
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            
            # Create metrics
            metrics = PerformanceMetrics(
                model_name=model_class.__name__,
                input_shape=input_shape,
                inference_time_ms=inference_time,
                memory_usage_mb=memory_usage,
                throughput_fps=throughput,
                flops_g=flops,
                params_m=params_m,
                efficiency_score=0.0  # Will be calculated later
            )
            
            # Calculate efficiency score
            metrics.efficiency_score = self.calculate_efficiency_score(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error benchmarking {model_class.__name__}: {str(e)}")
            return None
        
        finally:
            # Cleanup
            del model, input_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    def run_full_benchmark(self, input_shapes: List[Tuple[int, int, int, int]] = None) -> Dict:
        """Run complete benchmark suite"""
        if input_shapes is None:
            input_shapes = [
                (1, 3, 128, 128),   # Small images
                (1, 3, 256, 256),   # Medium images
                (1, 3, 512, 512),   # Large images
                (4, 3, 256, 256),   # Batch processing
            ]
        
        model_configs = self.get_model_configs()
        all_results = []
        
        print(f"Running benchmark suite on {self.device}")
        print(f"Testing {len(model_configs)} models with {len(input_shapes)} input shapes")
        print("-" * 80)
        
        for model_class, config in model_configs.items():
            for input_shape in input_shapes:
                # Skip if input size is too small for the model
                if self._should_skip_model_input(model_class, input_shape):
                    continue
                
                metrics = self.benchmark_model(model_class, input_shape, config)
                if metrics is not None:
                    all_results.append(metrics)
                    self.results.append(metrics)
        
        # Generate summary report
        summary = self.generate_summary_report(all_results)
        
        return summary
    
    def _should_skip_model_input(self, model_class: type, input_shape: Tuple[int, int, int, int]) -> bool:
        """Check if model should be skipped for given input size"""
        _, _, height, width = input_shape
        min_size = 32  # Minimum reasonable size for most models
        
        # Special cases
        if 'Swin' in model_class.__name__ and (height < 32 or width < 32):
            return True
        if 'ViT' in model_class.__name__ and (height < 32 or width < 32):
            return True
        if 'SegFormer' in model_class.__name__ and (height < 32 or width < 32):
            return True
        
        return height < min_size or width < min_size
    
    def generate_summary_report(self, results: List[PerformanceMetrics]) -> Dict:
        """Generate comprehensive benchmark summary"""
        if not results:
            return {"error": "No benchmark results available"}
        
        # Group by model
        model_groups = {}
        for result in results:
            if result.model_name not in model_groups:
                model_groups[result.model_name] = []
            model_groups[result.model_name].append(result)
        
        # Calculate statistics for each model
        model_stats = {}
        for model_name, model_results in model_groups.items():
            inference_times = [r.inference_time_ms for r in model_results]
            throughputs = [r.throughput_fps for r in model_results]
            memory_usages = [r.memory_usage_mb for r in model_results]
            efficiencies = [r.efficiency_score for r in model_results]
            
            model_stats[model_name] = {
                'avg_inference_time_ms': np.mean(inference_times),
                'std_inference_time_ms': np.std(inference_times),
                'avg_throughput_fps': np.mean(throughputs),
                'std_throughput_fps': np.std(throughputs),
                'avg_memory_mb': np.mean(memory_usages),
                'max_memory_mb': np.max(memory_usages),
                'avg_efficiency': np.mean(efficiencies),
                'best_efficiency': np.max(efficiencies),
                'params_m': model_results[0].params_m,
                'flops_g': model_results[0].flops_g,
            }
        
        # Find best performers
        best_speed = min(results, key=lambda x: x.inference_time_ms)
        best_throughput = max(results, key=lambda x: x.throughput_fps)
        best_efficiency = max(results, key=lambda x: x.efficiency_score)
        most_efficient_memory = min(results, key=lambda x: x.memory_usage_mb)
        
        summary = {
            'total_models_tested': len(model_groups),
            'total_configurations': len(results),
            'device': str(self.device),
            'model_statistics': model_stats,
            'best_performers': {
                'fastest_inference': {
                    'model': best_speed.model_name,
                    'input_shape': best_speed.input_shape,
                    'time_ms': best_speed.inference_time_ms,
                    'throughput_fps': best_speed.throughput_fps
                },
                'highest_throughput': {
                    'model': best_throughput.model_name,
                    'input_shape': best_throughput.input_shape,
                    'throughput_fps': best_throughput.throughput_fps,
                    'time_ms': best_throughput.inference_time_ms
                },
                'best_efficiency': {
                    'model': best_efficiency.model_name,
                    'input_shape': best_efficiency.input_shape,
                    'efficiency_score': best_efficiency.efficiency_score,
                    'throughput_fps': best_efficiency.throughput_fps,
                    'memory_mb': best_efficiency.memory_usage_mb
                },
                'most_memory_efficient': {
                    'model': most_efficient_memory.model_name,
                    'input_shape': most_efficient_memory.input_shape,
                    'memory_mb': most_efficient_memory.memory_usage_mb,
                    'throughput_fps': most_efficient_memory.throughput_fps
                }
            },
            'performance_tiers': self._categorize_performance_tiers(model_stats)
        }
        
        return summary
    
    def _categorize_performance_tiers(self, model_stats: Dict) -> Dict:
        """Categorize models into performance tiers"""
        if not model_stats:
            return {}
        
        # Extract efficiency scores
        efficiencies = {name: stats['avg_efficiency'] for name, stats in model_stats.items()}
        
        # Sort by efficiency
        sorted_models = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)
        
        n_models = len(sorted_models)
        
        # Define tiers (top 25%, next 25%, etc.)
        tier_size = max(1, n_models // 4)
        
        tiers = {
            'tier_1_excellent': sorted_models[:tier_size],
            'tier_2_good': sorted_models[tier_size:2*tier_size],
            'tier_3_average': sorted_models[2*tier_size:3*tier_size],
            'tier_4_below_average': sorted_models[3*tier_size:],
        }
        
        return tiers
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file"""
        if not self.results:
            print("No results to save")
            return
        
        # Convert to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                'model_name': result.model_name,
                'input_shape': result.input_shape,
                'inference_time_ms': result.inference_time_ms,
                'memory_usage_mb': result.memory_usage_mb,
                'throughput_fps': result.throughput_fps,
                'flops_g': result.flops_g,
                'params_m': result.params_m,
                'efficiency_score': result.efficiency_score,
            })
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_summary(self, summary: Dict):
        """Print formatted benchmark summary"""
        print("\n" + "="*80)
        print("SPATIAL MODEL PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nDevice: {summary['device']}")
        print(f"Models tested: {summary['total_models_tested']}")
        print(f"Total configurations: {summary['total_configurations']}")
        
        print("\n" + "-"*60)
        print("BEST PERFORMERS")
        print("-"*60)
        
        best = summary['best_performers']
        print(f"Fastest Inference: {best['fastest_inference']['model']} "
              f"({best['fastest_inference']['time_ms']:.2f}ms)")
        print(f"Highest Throughput: {best['highest_throughput']['model']} "
              f"({best['highest_throughput']['throughput_fps']:.2f} FPS)")
        print(f"Best Efficiency: {best['best_efficiency']['model']} "
              f"(score: {best['best_efficiency']['efficiency_score']:.3f})")
        print(f"Most Memory Efficient: {best['most_memory_efficient']['model']} "
              f"({best['most_memory_efficient']['memory_mb']:.1f}MB)")
        
        print("\n" + "-"*60)
        print("PERFORMANCE TIERS")
        print("-"*60)
        
        tiers = summary['performance_tiers']
        for tier_name, models in tiers.items():
            if models:
                tier_name_formatted = tier_name.replace('_', ' ').title()
                print(f"\n{tier_name_formatted}:")
                for model_name, efficiency in models:
                    stats = summary['model_statistics'][model_name]
                    print(f"  - {model_name}: {efficiency:.3f} "
                          f"({stats['avg_throughput_fps']:.1f} FPS, "
                          f"{stats['avg_memory_mb']:.1f}MB)")


# Test functions for pytest integration
@pytest.mark.benchmark
class TestSpatialModelPerformance:
    """Performance tests for spatial models"""
    
    @pytest.fixture(scope="class")
    def benchmark_suite(self):
        """Create benchmark suite"""
        return SpatialModelBenchmark(warmup_runs=3, benchmark_runs=10)
    
    def test_basic_performance_benchmark(self, benchmark_suite):
        """Run basic performance benchmark"""
        # Quick benchmark with smaller input sizes
        input_shapes = [(1, 3, 128, 128), (2, 3, 128, 128)]
        summary = benchmark_suite.run_full_benchmark(input_shapes)
        
        assert 'total_models_tested' in summary
        assert summary['total_models_tested'] > 0
        assert 'best_performers' in summary
        
        # Save results
        results_dir = Path('tests/results')
        results_dir.mkdir(exist_ok=True)
        benchmark_suite.save_results(str(results_dir / 'spatial_models_benchmark.json'))
        benchmark_suite.print_summary(summary)
    
    def test_memory_efficiency(self, benchmark_suite):
        """Test memory efficiency of models"""
        # Test with larger input to stress memory
        input_shapes = [(1, 3, 256, 256)]
        summary = benchmark_suite.run_full_benchmark(input_shapes)
        
        # Check that memory usage is reasonable
        for model_name, stats in summary['model_statistics'].items():
            assert stats['max_memory_mb'] < 4096, f"{model_name} uses too much memory: {stats['max_memory_mb']}MB"
    
    def test_inference_speed_requirements(self, benchmark_suite):
        """Test that models meet inference speed requirements"""
        input_shapes = [(1, 3, 128, 128)]
        summary = benchmark_suite.run_full_benchmark(input_shapes)
        
        # Check that models meet minimum speed requirements
        for model_name, stats in summary['model_statistics'].items():
            assert stats['avg_throughput_fps'] > 1.0, f"{model_name} too slow: {stats['avg_throughput_fps']:.2f} FPS"


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark = SpatialModelBenchmark(warmup_runs=5, benchmark_runs=20)
    
    # Test with different input sizes
    input_shapes = [
        (1, 3, 128, 128),   # Small
        (1, 3, 256, 256),   # Medium
        (1, 3, 512, 512),   # Large
        (4, 3, 256, 256),   # Batch
    ]
    
    summary = benchmark.run_full_benchmark(input_shapes)
    benchmark.print_summary(summary)
    
    # Save detailed results
    results_dir = Path('tests/results')
    results_dir.mkdir(exist_ok=True)
    benchmark.save_results(str(results_dir / 'full_spatial_models_benchmark.json'))
    
    print(f"\nBenchmark completed. Results saved to {results_dir}")