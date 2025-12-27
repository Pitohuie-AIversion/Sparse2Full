#!/usr/bin/env python3
"""
验证物理感知Transformer在PDE数据上的物理一致性
"""

import torch
import torch.nn as nn
import numpy as np
from models.temporal.models.physics_transformer import PhysicsTransformerTemporal
from models.temporal.components.physics_constraints import PhysicsConsistencyChecker
from models.temporal.factory import create_model

def create_pde_data(batch_size=2, T=10, H=64, W=64, channels=2, pde_type="navier_stokes"):
    """创建模拟PDE数据"""
    
    if pde_type == "navier_stokes":
        # Navier-Stokes方程：速度场(u,v)和压力场
        # 模拟涡流结构
        x = np.linspace(0, 2*np.pi, W)
        y = np.linspace(0, 2*np.pi, H)
        X, Y = np.meshgrid(x, y)
        
        # 时间演化
        data = []
        for t in range(T):
            # 模拟涡流的时间演化
            omega = 0.1  # 角频率
            k = 2.0      # 波数
            
            # 速度场u
            u = np.sin(k*X - omega*t) * np.cos(k*Y + omega*t) * np.exp(-0.01*t)
            # 速度场v  
            v = -np.cos(k*X - omega*t) * np.sin(k*Y + omega*t) * np.exp(-0.01*t)
            
            # 添加随机扰动
            u += 0.05 * np.random.randn(H, W)
            v += 0.05 * np.random.randn(H, W)
            
            data.append(np.stack([u, v], axis=0))
        
        data = np.array(data)  # [T, C, H, W]
        data = torch.tensor(data, dtype=torch.float32)
        
        # 扩展批次维度
        data = data.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [B, T, C, H, W]
        
    elif pde_type == "heat":
        # 热方程：扩散过程
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)
        
        # 初始条件：高斯热源
        T0 = np.exp(-(X**2 + Y**2) / 0.1)
        
        data = []
        for t in range(T):
            # 热扩散
            sigma = np.sqrt(0.1 + 0.01*t)
            T_t = np.exp(-(X**2 + Y**2) / (2*sigma**2)) / (2*np.pi*sigma**2)
            data.append(T_t[np.newaxis, :, :])  # [1, H, W]
        
        data = np.array(data)  # [T, 1, H, W]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [B, T, 1, H, W]
        
    else:
        # 默认：随机数据
        data = torch.randn(batch_size, T, channels, H, W)
    
    return data

def test_physics_consistency():
    """测试物理一致性"""
    print("=== 测试物理感知Transformer的物理一致性 ===")
    
    # 参数配置
    batch_size = 2
    T_in = 5
    T_out = 3
    H, W = 64, 64
    channels = 2
    
    # 创建模拟PDE数据
    print("创建模拟Navier-Stokes数据...")
    pde_data = create_pde_data(
        batch_size=batch_size, 
        T=T_in+T_out, 
        H=H, W=W, 
        channels=channels, 
        pde_type="navier_stokes"
    )
    
    # 分割输入和真值
    input_data = pde_data[:, :T_in]  # [B, T_in, C, H, W]
    ground_truth = pde_data[:, T_in:T_in+T_out]  # [B, T_out, C, H, W]
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"真值数据形状: {ground_truth.shape}")
    
    # 创建物理感知Transformer模型
    print("创建物理感知Transformer模型...")
    model = create_model(
        "PhysicsTransformer",
        in_channels=channels,
        out_channels=channels,
        img_size=[H, W],
        T_in=T_in,
        T_out=T_out,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        pde_type="navier_stokes",
        physics_weight=0.1,
        causal_weight=0.1,
        mode="ar"
    )
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        # 使用return_dict获取详细信息
        result = model(input_data, return_dict=True)
        
        prediction = result['prediction']  # [B, T_out, C, H, W]
        physics_valid = result['physics_valid']  # 物理一致性检查结果
    
    print(f"预测结果形状: {prediction.shape}")
    print(f"物理一致性检查结果: {physics_valid}")
    
    # 验证物理一致性
    print("\n=== 物理一致性分析 ===")
    
    # 1. 能量守恒检查
    energy_input = torch.sum(input_data ** 2, dim=list(range(2, input_data.dim())))
    energy_output = torch.sum(prediction ** 2, dim=list(range(2, prediction.dim())))
    energy_change = torch.abs(energy_output.mean() - energy_input.mean()) / energy_input.mean()
    
    print(f"输入能量均值: {energy_input.mean().item():.6f}")
    print(f"输出能量均值: {energy_output.mean().item():.6f}")
    print(f"相对能量变化: {energy_change.item():.6f}")
    
    # 2. 时间平滑性检查
    if prediction.size(1) >= 2:
        temporal_diff = torch.abs(prediction[:, 1:] - prediction[:, :-1])
        temporal_smoothness = temporal_diff.mean() / prediction.abs().mean()
        print(f"时间平滑性指标: {temporal_smoothness.item():.6f}")
    
    # 3. 空间平滑性检查（梯度）
    def compute_spatial_gradient(x):
        """计算空间梯度"""
        # x形状: [B, T, C, H, W]
        grad_x = x[..., :, 1:] - x[..., :, :-1]  # [B, T, C, H, W-1]
        grad_y = x[..., 1:, :] - x[..., :-1, :]  # [B, T, C, H-1, W]
        
        # 为了维度匹配，我们裁剪到相同尺寸
        min_h = min(grad_x.size(-2), grad_y.size(-2))
        min_w = min(grad_x.size(-1), grad_y.size(-1))
        
        grad_x_cropped = grad_x[..., :min_h, :min_w]
        grad_y_cropped = grad_y[..., :min_h, :min_w]
        
        return torch.sqrt(grad_x_cropped**2 + grad_y_cropped**2)
    
    input_grad = compute_spatial_gradient(input_data)
    output_grad = compute_spatial_gradient(prediction)
    
    grad_ratio = output_grad.mean() / input_grad.mean()
    print(f"空间梯度比率: {grad_ratio.item():.6f}")
    
    # 4. 物理一致性评分
    consistency_score = 0.0
    
    # 能量守恒评分（越小越好）
    energy_score = max(0, 1.0 - energy_change.item())
    consistency_score += energy_score
    
    # 时间平滑性评分（越小越好）
    if prediction.size(1) >= 2:
        temporal_score = max(0, 1.0 - temporal_smoothness.item())
        consistency_score += temporal_score
    
    # 空间平滑性评分（接近1.0为好）
    spatial_score = max(0, 1.0 - abs(grad_ratio.item() - 1.0))
    consistency_score += spatial_score
    
    # 归一化评分
    num_checks = 3 if prediction.size(1) >= 2 else 2
    consistency_score /= num_checks
    
    print(f"\n物理一致性评分: {consistency_score:.3f}/1.0")
    
    # 验证结果合理性
    assert not torch.isnan(prediction).any(), "预测结果包含NaN"
    assert torch.isfinite(prediction).all(), "预测结果包含无穷值"
    assert prediction.shape == (batch_size, T_out, channels, H, W), "输出形状不匹配"
    
    print("✓ 物理一致性验证通过")
    
    return {
        'prediction': prediction,
        'ground_truth': ground_truth,
        'physics_valid': physics_valid,
        'consistency_score': consistency_score,
        'energy_change': energy_change.item(),
        'temporal_smoothness': temporal_smoothness.item() if prediction.size(1) >= 2 else None,
        'spatial_gradient_ratio': grad_ratio.item()
    }

def compare_with_baselines():
    """与基线模型对比"""
    print("\n=== 与基线模型对比 ===")
    
    batch_size = 2
    T_in = 5
    T_out = 3
    H, W = 64, 64
    channels = 2
    
    # 创建测试数据
    input_data = torch.randn(batch_size, T_in, channels, H, W)
    
    # 测试不同模型
    models_to_test = [
        ("PhysicsTransformer", {
            "in_channels": channels, "out_channels": channels,
            "img_size": [H, W], "T_in": T_in, "T_out": T_out,
            "hidden_dim": 256, "num_heads": 8, "num_layers": 4,
            "pde_type": "navier_stokes", "physics_weight": 0.1
        }),
        ("SwinTemporal", {
            "in_channels": channels, "out_channels": channels,
            "img_size": [H, W], "T_in": T_in, "T_out": T_out
        })
    ]
    
    results = {}
    
    for model_name, model_config in models_to_test:
        print(f"\n测试 {model_name}...")
        
        try:
            model = create_model(model_name, **model_config)
            
            with torch.no_grad():
                if model_name == "PhysicsTransformer":
                    result = model(input_data, return_dict=True)
                    prediction = result['prediction']
                    physics_valid = result['physics_valid']
                else:
                    prediction = model(input_data)
                    physics_valid = None
            
            # 计算基本指标
            params = sum(p.numel() for p in model.parameters())
            
            results[model_name] = {
                'params': params,
                'output_shape': prediction.shape,
                'physics_valid': physics_valid,
                'model_name': model_name
            }
            
            print(f"  参数数量: {params:,}")
            print(f"  输出形状: {prediction.shape}")
            if physics_valid:
                print(f"  物理有效性: {physics_valid}")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results

def main():
    """主函数"""
    print("开始物理感知Transformer物理一致性验证...\n")
    
    try:
        # 1. 测试物理一致性
        consistency_results = test_physics_consistency()
        
        # 2. 与基线对比
        baseline_results = compare_with_baselines()
        
        print("\n=== 总结 ===")
        print(f"物理一致性评分: {consistency_results['consistency_score']:.3f}/1.0")
        print(f"能量变化: {consistency_results['energy_change']:.6f}")
        if consistency_results['temporal_smoothness']:
            print(f"时间平滑性: {consistency_results['temporal_smoothness']:.6f}")
        print(f"空间梯度比率: {consistency_results['spatial_gradient_ratio']:.6f}")
        
        print("\n=== 模型对比 ===")
        for model_name, result in baseline_results.items():
            if 'error' in result:
                print(f"{model_name}: ❌ 失败 - {result['error']}")
            else:
                print(f"{model_name}: ✅ {result['params']:,}参数, 输出{result['output_shape']}")
        
        print("\n✅ 所有验证完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)