"""
测试资源监控功能

验证系统资源监控的正确实现：
- Params: 模型参数数量（百万）
- FLOPs: 浮点运算次数（十亿@256²）
- 显存: GPU内存使用峰值（GB）
- 延迟: 推理延迟（毫秒）
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Dict, Tuple
import numpy as np


class SimpleTestModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()
    
    def reset(self):
        """重置监控状态"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def count_parameters(self, model: nn.Module) -> Dict[str, float]:
        """计算模型参数数量
        
        Returns:
            params_info: {'total_params': int, 'trainable_params': int, 'params_M': float}
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'params_M': total_params / 1e6  # 百万参数
        }
    
    def calculate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """计算FLOPs
        
        Args:
            model: 模型
            input_shape: 输入形状 (B, C, H, W)
            
        Returns:
            flops_info: {'total_flops': int, 'flops_G': float}
        """
        model = model.to(self.device)  # 确保模型在正确设备上
        model.eval()
        
        # 创建输入张量
        x = torch.randn(input_shape).to(self.device)
        
        # 使用钩子函数计算FLOPs
        flops_count = [0]
        
        def conv_flop_count(module, input, output):
            if isinstance(module, nn.Conv2d):
                # 卷积层FLOPs = 输出元素数 × (输入通道数 × 核大小 + 1)
                output_dims = output.shape
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                
                filters_per_channel = output_dims[1]
                output_elements = output_dims[0] * output_dims[2] * output_dims[3]
                
                # 每个输出元素的计算量
                kernel_flops = kernel_dims[0] * kernel_dims[1] * in_channels
                if module.bias is not None:
                    kernel_flops += 1
                
                flops = kernel_flops * filters_per_channel * output_elements
                flops_count[0] += flops
        
        # 注册钩子
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(conv_flop_count)
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = model(x)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        total_flops = flops_count[0]
        
        return {
            'total_flops': total_flops,
            'flops_G': total_flops / 1e9  # 十亿FLOPs
        }
    
    def measure_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """测量显存使用
        
        Returns:
            memory_info: {'peak_memory_MB': float, 'peak_memory_GB': float}
        """
        if not torch.cuda.is_available():
            return {'peak_memory_MB': 0.0, 'peak_memory_GB': 0.0}
        
        model = model.to(self.device)
        model.eval()
        
        # 重置显存统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建输入并进行前向传播
        x = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = model(x)
        
        # 获取峰值显存使用
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        
        return {
            'peak_memory_MB': peak_memory_mb,
            'peak_memory_GB': peak_memory_gb
        }
    
    def measure_inference_latency(self, model: nn.Module, input_shape: Tuple[int, ...], 
                                 num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """测量推理延迟
        
        Args:
            model: 模型
            input_shape: 输入形状
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            latency_info: {'mean_latency_ms': float, 'std_latency_ms': float, 'min_latency_ms': float, 'max_latency_ms': float}
        """
        model = model.to(self.device)
        model.eval()
        
        # 创建输入
        x = torch.randn(input_shape).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量延迟
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies))
        }
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Dict]:
        """完整的模型性能分析
        
        Returns:
            profile_results: {
                'params': {...},
                'flops': {...},
                'memory': {...},
                'latency': {...}
            }
        """
        print(f"正在分析模型性能 (输入形状: {input_shape})...")
        
        # 参数统计
        params_info = self.count_parameters(model)
        print(f"参数数量: {params_info['params_M']:.2f}M")
        
        # FLOPs计算
        flops_info = self.calculate_flops(model, input_shape)
        print(f"FLOPs: {flops_info['flops_G']:.2f}G")
        
        # 显存使用
        memory_info = self.measure_memory_usage(model, input_shape)
        print(f"显存使用: {memory_info['peak_memory_GB']:.3f}GB")
        
        # 推理延迟
        latency_info = self.measure_inference_latency(model, input_shape)
        print(f"推理延迟: {latency_info['mean_latency_ms']:.2f}±{latency_info['std_latency_ms']:.2f}ms")
        
        return {
            'params': params_info,
            'flops': flops_info,
            'memory': memory_info,
            'latency': latency_info
        }


def test_parameter_counting():
    """测试参数计数"""
    print("测试参数计数...")
    
    monitor = ResourceMonitor()
    
    # 创建测试模型
    model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=64)
    
    # 计算参数
    params_info = monitor.count_parameters(model)
    
    # 验证结果
    assert params_info['total_params'] > 0, "总参数数应该大于0"
    assert params_info['trainable_params'] > 0, "可训练参数数应该大于0"
    assert params_info['params_M'] > 0, "参数数量（百万）应该大于0"
    
    # 手动验证参数数量
    expected_params = (
        3 * 64 * 3 * 3 + 64 +  # conv1: 3->64, 3x3 + bias
        64 * 64 * 3 * 3 + 64 + # conv2: 64->64, 3x3 + bias  
        64 * 3 * 3 * 3 + 3     # conv3: 64->3, 3x3 + bias
    )
    
    assert params_info['total_params'] == expected_params, \
        f"参数数量不匹配: {params_info['total_params']} vs {expected_params}"
    
    print(f"✓ 参数计数正确: {params_info['params_M']:.3f}M 参数")


def test_flops_calculation():
    """测试FLOPs计算"""
    print("测试FLOPs计算...")
    
    monitor = ResourceMonitor()
    model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=64)
    
    # 计算FLOPs
    input_shape = (1, 3, 256, 256)
    flops_info = monitor.calculate_flops(model, input_shape)
    
    # 验证结果
    assert flops_info['total_flops'] > 0, "总FLOPs应该大于0"
    assert flops_info['flops_G'] > 0, "FLOPs（十亿）应该大于0"
    
    # 估算FLOPs（粗略验证）
    H, W = 256, 256
    estimated_flops = (
        3 * 64 * 3 * 3 * H * W +  # conv1
        64 * 64 * 3 * 3 * H * W + # conv2
        64 * 3 * 3 * 3 * H * W    # conv3
    )
    
    # 允许一定误差
    ratio = flops_info['total_flops'] / estimated_flops
    assert 0.8 <= ratio <= 1.2, f"FLOPs计算可能有误: {ratio:.2f}"
    
    print(f"✓ FLOPs计算正确: {flops_info['flops_G']:.2f}G FLOPs")


def test_memory_monitoring():
    """测试显存监控"""
    print("测试显存监控...")
    
    monitor = ResourceMonitor()
    model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=64)
    
    # 测量显存使用
    input_shape = (4, 3, 256, 256)  # 更大的batch size
    memory_info = monitor.measure_memory_usage(model, input_shape)
    
    # 验证结果
    if torch.cuda.is_available():
        assert memory_info['peak_memory_MB'] > 0, "显存使用应该大于0"
        assert memory_info['peak_memory_GB'] > 0, "显存使用（GB）应该大于0"
        print(f"✓ 显存监控正确: {memory_info['peak_memory_GB']:.3f}GB")
    else:
        assert memory_info['peak_memory_MB'] == 0, "CPU模式下显存使用应该为0"
        print("✓ CPU模式下显存监控正确")


def test_latency_measurement():
    """测试延迟测量"""
    print("测试延迟测量...")
    
    monitor = ResourceMonitor()
    model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=64)
    
    # 测量延迟
    input_shape = (1, 3, 256, 256)
    latency_info = monitor.measure_inference_latency(model, input_shape, num_runs=20)
    
    # 验证结果
    assert latency_info['mean_latency_ms'] > 0, "平均延迟应该大于0"
    assert latency_info['std_latency_ms'] >= 0, "延迟标准差应该非负"
    assert latency_info['min_latency_ms'] > 0, "最小延迟应该大于0"
    assert latency_info['max_latency_ms'] >= latency_info['min_latency_ms'], "最大延迟应该不小于最小延迟"
    
    print(f"✓ 延迟测量正确: {latency_info['mean_latency_ms']:.2f}±{latency_info['std_latency_ms']:.2f}ms")


def test_complete_profiling():
    """测试完整性能分析"""
    print("测试完整性能分析...")
    
    monitor = ResourceMonitor()
    model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=128)  # 更大的模型
    
    # 完整性能分析
    input_shape = (2, 3, 256, 256)
    profile_results = monitor.profile_model(model, input_shape)
    
    # 验证结果结构
    expected_keys = ['params', 'flops', 'memory', 'latency']
    for key in expected_keys:
        assert key in profile_results, f"缺少性能指标: {key}"
    
    # 验证各项指标
    params = profile_results['params']
    assert params['params_M'] > 0, "参数数量应该大于0"
    
    flops = profile_results['flops']
    assert flops['flops_G'] > 0, "FLOPs应该大于0"
    
    memory = profile_results['memory']
    if torch.cuda.is_available():
        assert memory['peak_memory_GB'] > 0, "显存使用应该大于0"
    
    latency = profile_results['latency']
    assert latency['mean_latency_ms'] > 0, "延迟应该大于0"
    
    print("✓ 完整性能分析正确")


def test_resource_comparison():
    """测试资源对比"""
    print("测试资源对比...")
    
    monitor = ResourceMonitor()
    
    # 创建不同大小的模型
    small_model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=32)
    large_model = SimpleTestModel(in_channels=3, out_channels=3, hidden_dim=128)
    
    input_shape = (1, 3, 256, 256)
    
    # 分析两个模型
    small_profile = monitor.profile_model(small_model, input_shape)
    large_profile = monitor.profile_model(large_model, input_shape)
    
    # 验证大模型的资源消耗更多
    assert large_profile['params']['params_M'] > small_profile['params']['params_M'], \
        "大模型参数数量应该更多"
    
    assert large_profile['flops']['flops_G'] > small_profile['flops']['flops_G'], \
        "大模型FLOPs应该更多"
    
    if torch.cuda.is_available():
        assert large_profile['memory']['peak_memory_GB'] >= small_profile['memory']['peak_memory_GB'], \
            "大模型显存使用应该不少于小模型"
    
    print("✓ 资源对比测试正确")


def main():
    """主测试函数"""
    print("PDEBench稀疏观测重建系统 - 资源监控测试")
    print("=" * 60)
    
    try:
        test_parameter_counting()
        test_flops_calculation()
        test_memory_monitoring()
        test_latency_measurement()
        test_complete_profiling()
        test_resource_comparison()
        
        print("\n" + "=" * 60)
        print("✓ 所有资源监控测试通过！")
        print("✓ 参数计数功能正确")
        print("✓ FLOPs计算功能正确")
        print("✓ 显存监控功能正确")
        print("✓ 延迟测量功能正确")
        print("✓ 完整性能分析功能正确")
        print("✓ 资源对比功能正确")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()