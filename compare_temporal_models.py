#!/usr/bin/env python3
"""
对比分析不同时间预测模型的性能
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple
from models.temporal.factory import create_model

def benchmark_model(model, input_data, num_iterations=10, warmup_iterations=3):
    """基准测试模型性能"""
    
    # 预热
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(input_data)
    
    # 同步CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时测试
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            output = model(input_data)
    
    # 同步CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    return avg_time

def compute_metrics(prediction, ground_truth):
    """计算评估指标"""
    
    # 相对L2误差
    rel_l2 = torch.norm(prediction - ground_truth) / torch.norm(ground_truth)
    
    # MAE
    mae = torch.mean(torch.abs(prediction - ground_truth))
    
    # RMSE
    rmse = torch.sqrt(torch.mean((prediction - ground_truth) ** 2))
    
    # 能量守恒检查
    energy_pred = torch.sum(prediction ** 2)
    energy_gt = torch.sum(ground_truth ** 2)
    energy_error = torch.abs(energy_pred - energy_gt) / energy_gt
    
    return {
        'rel_l2': rel_l2.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'energy_error': energy_error.item()
    }

def test_temporal_models():
    """测试各种时序模型"""
    print("=== 时序预测模型性能对比 ===\n")
    
    # 测试配置
    batch_size = 2
    T_in = 5
    T_out = 3
    H, W = 64, 64
    channels = 2
    
    # 创建测试数据
    print("创建测试数据...")
    input_data = torch.randn(batch_size, T_in, channels, H, W)
    ground_truth = torch.randn(batch_size, T_out, channels, H, W)
    
    print(f"输入形状: {input_data.shape}")
    print(f"真值形状: {ground_truth.shape}")
    
    # 模型配置
    model_configs = [
        {
            'name': 'PhysicsTransformer',
            'config': {
                'in_channels': channels,
                'out_channels': channels,
                'img_size': [H, W],
                'T_in': T_in,
                'T_out': T_out,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'pde_type': 'navier_stokes',
                'physics_weight': 0.1,
                'mode': 'ar'
            }
        },
        {
            'name': 'PhysicsTransformer_Small',
            'config': {
                'in_channels': channels,
                'out_channels': channels,
                'img_size': [H, W],
                'T_in': T_in,
                'T_out': T_out,
                'hidden_dim': 128,
                'num_heads': 4,
                'num_layers': 2,
                'pde_type': 'navier_stokes',
                'physics_weight': 0.1,
                'mode': 'ar'
            }
        },
        {
            'name': 'PhysicsTransformer_Large',
            'config': {
                'in_channels': channels,
                'out_channels': channels,
                'img_size': [H, W],
                'T_in': T_in,
                'T_out': T_out,
                'hidden_dim': 512,
                'num_heads': 16,
                'num_layers': 8,
                'pde_type': 'navier_stokes',
                'physics_weight': 0.1,
                'mode': 'ar'
            }
        }
    ]
    
    # 存储结果
    results = {}
    
    print("\n开始模型测试...\n")
    
    for model_info in model_configs:
        model_name = model_info['name']
        model_config = model_info['config']
        
        print(f"测试 {model_name}...")
        
        try:
            # 创建模型
            model = create_model("PhysicsTransformer", **model_config)
            model.eval()
            
            # 获取模型信息
            params = sum(p.numel() for p in model.parameters())
            
            # 基准测试
            print(f"  基准测试...")
            inference_time = benchmark_model(model, input_data)
            
            # 前向传播获取预测
            with torch.no_grad():
                if model_name.startswith('PhysicsTransformer'):
                    result = model(input_data, return_dict=True)
                    prediction = result['prediction']
                    physics_valid = result.get('physics_valid', {})
                else:
                    prediction = model(input_data)
                    physics_valid = {}
            
            # 计算指标
            metrics = compute_metrics(prediction, ground_truth)
            
            # 物理一致性检查
            if physics_valid:
                physics_score = sum(physics_valid.values()) / len(physics_valid) if physics_valid else 0.0
            else:
                physics_score = 0.0
            
            # 存储结果
            results[model_name] = {
                'params': params,
                'inference_time': inference_time,
                'output_shape': prediction.shape,
                'metrics': metrics,
                'physics_score': physics_score,
                'physics_valid': physics_valid
            }
            
            print(f"  ✅ 完成")
            print(f"    参数: {params:,}")
            print(f"    推理时间: {inference_time*1000:.2f}ms")
            print(f"    相对L2误差: {metrics['rel_l2']:.6f}")
            print(f"    MAE: {metrics['mae']:.6f}")
            print(f"    能量误差: {metrics['energy_error']:.6f}")
            if physics_score > 0:
                print(f"    物理一致性: {physics_score:.3f}")
            print()
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            results[model_name] = {'error': str(e)}
            print()
    
    return results

def analyze_scaling_behavior():
    """分析模型扩展性"""
    print("=== 模型扩展性分析 ===\n")
    
    # 不同规模的数据
    scales = [
        {'H': 32, 'W': 32, 'name': 'Small'},
        {'H': 64, 'W': 64, 'name': 'Medium'},
        {'H': 128, 'W': 128, 'name': 'Large'}
    ]
    
    batch_size = 2
    T_in = 5
    T_out = 3
    channels = 2
    
    scaling_results = {}
    
    for scale in scales:
        H, W = scale['H'], scale['W']
        scale_name = scale['name']
        
        print(f"测试 {scale_name} 规模 ({H}x{W})...")
        
        # 创建数据
        input_data = torch.randn(batch_size, T_in, channels, H, W)
        
        # 模型配置（固定架构，变化输入规模）
        model_config = {
            'in_channels': channels,
            'out_channels': channels,
            'img_size': [H, W],
            'T_in': T_in,
            'T_out': T_out,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'pde_type': 'navier_stokes',
            'physics_weight': 0.1,
            'mode': 'ar'
        }
        
        try:
            model = create_model("PhysicsTransformer", **model_config)
            model.eval()
            
            # 基准测试
            inference_time = benchmark_model(model, input_data)
            
            # 参数数量
            params = sum(p.numel() for p in model.parameters())
            
            scaling_results[scale_name] = {
                'input_size': (H, W),
                'params': params,
                'inference_time': inference_time,
                'input_memory': input_data.numel() * 4 / (1024**2),  # MB
                'model_memory': params * 4 / (1024**2)  # MB
            }
            
            print(f"  输入内存: {scaling_results[scale_name]['input_memory']:.2f}MB")
            print(f"  模型内存: {scaling_results[scale_name]['model_memory']:.2f}MB")
            print(f"  推理时间: {inference_time*1000:.2f}ms")
            print()
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            scaling_results[scale_name] = {'error': str(e)}
            print()
    
    return scaling_results

def generate_comparison_report():
    """生成对比报告"""
    print("=== 生成对比报告 ===\n")
    
    # 运行性能测试
    performance_results = test_temporal_models()
    
    # 运行扩展性分析
    scaling_results = analyze_scaling_behavior()
    
    print("=" * 80)
    print("时序预测模型性能对比报告")
    print("=" * 80)
    print()
    
    # 性能对比
    print("1. 模型性能对比")
    print("-" * 40)
    
    if performance_results:
        # 排序（按相对L2误差）
        sorted_models = sorted(
            [(name, data) for name, data in performance_results.items() if 'error' not in data],
            key=lambda x: x[1]['metrics']['rel_l2']
        )
        
        print(f"{'模型':<20} {'参数':<12} {'推理时间':<12} {'RelL2':<12} {'MAE':<12} {'能量误差':<12}")
        print("-" * 80)
        
        for model_name, data in sorted_models:
            print(f"{model_name:<20} {data['params']:<12,} "
                  f"{data['inference_time']*1000:<12.2f} {data['metrics']['rel_l2']:<12.6f} "
                  f"{data['metrics']['mae']:<12.6f} {data['metrics']['energy_error']:<12.6f}")
    
    print()
    
    # 扩展性分析
    print("2. 模型扩展性分析")
    print("-" * 40)
    
    if scaling_results:
        print(f"{'规模':<10} {'输入尺寸':<15} {'参数':<12} {'输入内存':<12} {'模型内存':<12} {'推理时间':<12}")
        print("-" * 80)
        
        for scale_name, data in scaling_results.items():
            if 'error' not in data:
                H, W = data['input_size']
                print(f"{scale_name:<10} {f'{H}x{W}':<15} {data['params']:<12,} "
                      f"{data['input_memory']:<12.2f} {data['model_memory']:<12.2f} "
                      f"{data['inference_time']*1000:<12.2f}")
    
    print()
    
    # 推荐建议
    print("3. 模型选择建议")
    print("-" * 40)
    
    if performance_results:
        best_accuracy = min(
            [(name, data['metrics']['rel_l2']) for name, data in performance_results.items() if 'error' not in data],
            key=lambda x: x[1]
        )
        
        best_efficiency = min(
            [(name, data['inference_time']) for name, data in performance_results.items() if 'error' not in data],
            key=lambda x: x[1]
        )
        
        smallest_model = min(
            [(name, data['params']) for name, data in performance_results.items() if 'error' not in data],
            key=lambda x: x[1]
        )
        
        print(f"• 最佳精度: {best_accuracy[0]} (RelL2: {best_accuracy[1]:.6f})")
        print(f"• 最高效率: {best_efficiency[0]} ({best_efficiency[1]*1000:.2f}ms)")
        print(f"• 最小模型: {smallest_model[0]} ({smallest_model[1]:,}参数)")
    
    print()
    print("4. 物理感知特性")
    print("-" * 40)
    
    physics_models = [(name, data) for name, data in performance_results.items() 
                     if 'error' not in data and data.get('physics_score', 0) > 0]
    
    if physics_models:
        print("物理约束检查:")
        for model_name, data in physics_models:
            print(f"• {model_name}: 物理一致性评分 {data['physics_score']:.3f}")
            if data['physics_valid']:
                print(f"  详细检查: {data['physics_valid']}")
    
    return performance_results, scaling_results

def main():
    """主函数"""
    print("开始时序预测模型性能对比分析...\n")
    
    try:
        # 生成对比报告
        performance_results, scaling_results = generate_comparison_report()
        
        print("\n" + "=" * 80)
        print("对比分析完成！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 对比分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)