#!/usr/bin/env python3
"""
计算模型资源统计：参数量、FLOPs、内存使用、推理性能
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from models import *
from utils.config import load_config
from utils.performance import PerformanceProfiler

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """计算模型参数量
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_model_size(model: nn.Module) -> float:
    """估算模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def calculate_flops_simple(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    """简化的FLOPs计算（基于参数量估算）"""
    total_params, _ = count_parameters(model)
    h, w = input_shape[2], input_shape[3]
    
    # 基于模型类型的经验公式
    model_name = model.__class__.__name__.lower()
    
    if 'fno' in model_name:
        # FNO模型主要是频域操作，相对较少
        flops = total_params * 2 * (h * w) / 1e9
    elif 'transformer' in model_name or 'swin' in model_name or 'former' in model_name:
        # Transformer类模型，注意力机制计算量大
        flops = total_params * 4 * (h * w) / 1e9
    elif 'unet' in model_name:
        # UNet类模型，卷积操作为主
        flops = total_params * 2 * (h * w) / 1e9
    elif 'mlp' in model_name:
        # MLP模型，全连接操作
        flops = total_params * 2 / 1e9
    else:
        # 默认估算
        flops = total_params * 2 * (h * w) / 1e9
    
    return flops

def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> Dict[str, float]:
    """估算内存使用（GB）"""
    total_params, _ = count_parameters(model)
    model_size_mb = estimate_model_size(model)
    
    # 输入数据大小
    input_size_mb = np.prod(input_shape) * batch_size * 4 / 1024 / 1024  # float32
    
    # 估算中间激活值大小（经验公式）
    model_name = model.__class__.__name__.lower()
    if 'transformer' in model_name or 'swin' in model_name:
        activation_multiplier = 8  # Transformer需要更多中间激活值
    elif 'unet' in model_name:
        activation_multiplier = 6  # UNet有跳跃连接
    else:
        activation_multiplier = 4  # 默认
    
    activation_size_mb = input_size_mb * activation_multiplier
    
    # 训练时需要额外的梯度和优化器状态
    training_memory_gb = (model_size_mb + input_size_mb + activation_size_mb + model_size_mb * 2) / 1024
    inference_memory_gb = (model_size_mb + input_size_mb + activation_size_mb * 0.5) / 1024
    
    return {
        'training_gb': training_memory_gb,
        'inference_gb': inference_memory_gb
    }

def benchmark_inference_speed(model: nn.Module, input_shape: Tuple[int, ...], 
                            device: str = 'cpu', num_runs: int = 100) -> Dict[str, float]:
    """基准测试推理速度"""
    model.eval()
    model = model.to(device)
    
    # 预热
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 同步GPU（如果使用）
    if device != 'cpu':
        torch.cuda.synchronize()
    
    # 测试推理时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device != 'cpu':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / mean_time  # 每秒帧数
    
    return {
        'latency_ms': mean_time,
        'latency_std_ms': std_time,
        'fps': fps
    }

def create_model_from_config(model_name: str, config_path: str = "configs/train.yaml") -> nn.Module:
    """根据配置创建模型"""
    
    # 模型配置映射 - 使用正确的类名和参数
    model_configs = {
        'FNO2D': {'in_channels': 1, 'out_channels': 1, 'img_size': 256, 'modes1': 16, 'modes2': 16, 'width': 64},
        'UNet': {'in_channels': 1, 'out_channels': 1, 'img_size': 256},
        'SwinUNet': {'in_channels': 1, 'out_channels': 1, 'img_size': 256, 'embed_dim': 96, 'depths': [2, 2, 6, 2]},
        'MLP': {'in_channels': 1, 'out_channels': 1, 'img_size': 256},
        'MLP_Mixer': {'in_channels': 1, 'out_channels': 1, 'img_size': 256, 'patch_size': 16, 'dim': 512, 'depth': 8},
        'Hybrid': {'in_channels': 1, 'out_channels': 1, 'img_size': 256},
        'UNetPlusPlus': {'in_channels': 1, 'out_channels': 1, 'img_size': 256},
        'UFNO_UNet': {'in_channels': 1, 'out_channels': 1, 'img_size': 256, 'modes1': 16, 'modes2': 16, 'width': 64}
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = model_configs[model_name]
    
    # 创建模型 - 使用正确的类名
    if model_name == 'FNO2D':
        model = FNO2d(**model_config)
    elif model_name == 'UNet':
        model = UNet(**model_config)
    elif model_name == 'SwinUNet':
        model = SwinUNet(**model_config)
    elif model_name == 'MLP':
        model = MLPModel(**model_config)
    elif model_name == 'MLP_Mixer':
        model = MLPMixer(**model_config)
    elif model_name == 'Hybrid':
        model = HybridModel(**model_config)
    elif model_name == 'UNetPlusPlus':
        model = UNetPlusPlus(**model_config)
    elif model_name == 'UFNO_UNet':
        model = UFNOUNet(**model_config)
    
    return model

def calculate_all_model_resources():
    """计算所有模型的资源统计"""
    
    # 从现有的model_ranking.csv获取模型列表
    ranking_file = "paper_package/metrics/original_model_ranking.csv"
    if not os.path.exists(ranking_file):
        print(f"错误：找不到文件 {ranking_file}")
        return None
    
    df = pd.read_csv(ranking_file)
    model_names = df['模型'].tolist()
    
    print(f"开始计算 {len(model_names)} 个模型的资源统计...")
    
    input_shape = (1, 1, 256, 256)  # [B, C, H, W]
    device = 'cpu'  # 使用CPU避免GPU内存问题
    
    results = {}
    
    for model_name in model_names:
        print(f"\n正在处理模型: {model_name}")
        
        try:
            # 创建模型
            model = create_model_from_config(model_name)
            
            # 计算参数量
            total_params, trainable_params = count_parameters(model)
            
            # 计算模型大小
            model_size_mb = estimate_model_size(model)
            
            # 计算FLOPs
            flops_g = calculate_flops_simple(model, input_shape)
            
            # 估算内存使用
            memory_usage = estimate_memory_usage(model, input_shape)
            
            # 基准测试推理速度
            speed_results = benchmark_inference_speed(model, input_shape, device, num_runs=50)
            
            # 计算效率指标
            gflops_per_sec = flops_g / (speed_results['latency_ms'] / 1000) if speed_results['latency_ms'] > 0 else 0
            
            results[model_name] = {
                'total_params_M': total_params / 1e6,
                'trainable_params_M': trainable_params / 1e6,
                'model_size_MB': model_size_mb,
                'flops_G': flops_g,
                'gflops_per_sec': gflops_per_sec,
                'training_memory_GB': memory_usage['training_gb'],
                'inference_memory_GB': memory_usage['inference_gb'],
                'latency_ms': speed_results['latency_ms'],
                'latency_std_ms': speed_results['latency_std_ms'],
                'fps': speed_results['fps']
            }
            
            print(f"  参数量: {total_params/1e6:.2f}M")
            print(f"  FLOPs: {flops_g:.2f}G")
            print(f"  推理延迟: {speed_results['latency_ms']:.2f}ms")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            # 使用默认值
            results[model_name] = {
                'total_params_M': 0.0,
                'trainable_params_M': 0.0,
                'model_size_MB': 0.0,
                'flops_G': 0.0,
                'gflops_per_sec': 0.0,
                'training_memory_GB': 0.0,
                'inference_memory_GB': 0.0,
                'latency_ms': 0.0,
                'latency_std_ms': 0.0,
                'fps': 0.0
            }
    
    # 保存结果
    output_file = "paper_package/metrics/model_resources.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n资源统计结果已保存到: {output_file}")
    return results

if __name__ == "__main__":
    results = calculate_all_model_resources()
    if results:
        print("\n=== 资源统计完成 ===")
        for model_name, stats in results.items():
            print(f"{model_name}: {stats['total_params_M']:.2f}M参数, {stats['flops_G']:.2f}G FLOPs, {stats['latency_ms']:.2f}ms延迟")