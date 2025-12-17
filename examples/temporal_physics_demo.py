"""
物理感知Transformer时序模型使用示例
展示如何使用新的时序预测模块
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入新的时序模块
from models.temporal.factory import create_model
from models.temporal.validation.temporal_consistency import TemporalConsistencyValidator


def generate_heat_equation_data(
    num_samples: int = 32,
    spatial_size: Tuple[int, int] = (64, 64),
    time_steps: int = 20,
    dt: float = 0.01,
    dx: float = 0.1,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成热方程数据用于测试"""
    H, W = spatial_size
    
    # 初始条件：高斯热源
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # 初始高斯分布
    initial_condition = np.exp(-10 * (X**2 + Y**2))
    
    # 数值求解热方程（简化版本）
    data = np.zeros((num_samples, time_steps, H, W))
    
    for b in range(num_samples):
        u = initial_condition.copy()
        
        # 添加一些随机性
        center_x = np.random.normal(0, 0.1)
        center_y = np.random.normal(0, 0.1)
        u = np.exp(-10 * ((X-center_x)**2 + (Y-center_y)**2))
        
        data[b, 0] = u
        
        # 简单的热方程数值解
        for t in range(1, time_steps):
            u_xx = (np.roll(u, 1, axis=1) - 2*u + np.roll(u, -1, axis=1)) / dx**2
            u_yy = (np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0)) / dx**2
            
            u_t = alpha * (u_xx + u_yy)
            u = u + dt * u_t
            
            data[b, t] = u
    
    return torch.FloatTensor(data[:, :-1]), torch.FloatTensor(data[:, 1:])


def test_physics_transformer():
    """测试物理感知Transformer模型"""
    print("Testing Physics-Aware Transformer Temporal Model")
    print("=" * 60)
    
    # 配置参数
    config = {
        'in_channels': 1,
        'out_channels': 1,
        'img_size': (64, 64),
        'T_in': 5,
        'T_out': 5,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'pde_type': 'heat',
        'constraint_weights': {
            'pde_residual': 1.0,
            'energy_conservation': 0.5,
            'boundary_condition': 0.3,
            'causality': 0.2,
            'smoothness': 0.1
        },
        'use_frequency_encoding': True,
        'mode': 'ar',
        'dropout': 0.1
    }
    
    # 创建模型
    print("Creating PhysicsTransformer model...")
    model = create_model('PhysicsTransformer', **config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 生成测试数据
    print("Generating test data...")
    input_data, target_data = generate_heat_equation_data(
        num_samples=8, spatial_size=(64, 64), time_steps=20
    )
    
    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target_data.shape}")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        # 准备输入（取前5个时间步）
        x_input = input_data[:, :5]  # [B, T_in, H, W]
        
        # 前向传播
        predictions = model(x_input, T_out=5, return_dict=False)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # 计算基础误差
        if predictions.shape == target_data[:, :5].shape:
            mse = nn.MSELoss()(predictions, target_data[:, :5])
            mae = nn.L1Loss()(predictions, target_data[:, :5])
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
    
    return model, input_data, predictions


def test_temporal_consistency():
    """测试时序一致性验证"""
    print("\nTesting Temporal Consistency Validation")
    print("=" * 50)
    
    # 创建验证器
    validator = TemporalConsistencyValidator(
        pde_type='heat',
        tolerance=1e-3,
        enable_visualization=True
    )
    
    # 生成测试预测
    B, T, H, W = 4, 10, 32, 32
    predictions = torch.randn(B, T, H, W) * 0.1
    
    # 添加一些物理合理的特性
    for b in range(B):
        for t in range(1, T):
            # 添加时间平滑性
            predictions[b, t] = 0.8 * predictions[b, t-1] + 0.2 * torch.randn(H, W) * 0.05
    
    # 验证物理一致性
    print("Validating physical consistency...")
    validation_results = validator.comprehensive_validation(
        predictions=predictions,
        dt=0.01,
        dx=0.1
    )
    
    # 打印关键结果
    print("\nKey Validation Results:")
    print(f"Physics Residual L2: {validation_results['residual_l2_mean']:.6f}")
    print(f"Energy Conservation: {validation_results['energy_change_mean']:.6f}")
    print(f"Causality Violation: {validation_results['causality_violation_mean']:.6f}")
    print(f"Boundary Consistency: {validation_results['boundary_consistency_error']:.6f}")
    print(f"Overall Physics Valid: {validation_results['overall_physics_valid']}")
    print(f"Validation Pass Rate: {validation_results['validation_pass_rate']:.2%}")
    
    # 生成验证报告
    report = validator.generate_validation_report()
    print(f"\nValidation Report:\n{report}")
    
    return validator, validation_results


def compare_temporal_models():
    """比较不同的时序模型"""
    print("\nComparing Different Temporal Models")
    print("=" * 50)
    
    # 基础配置
    base_config = {
        'in_channels': 1,
        'out_channels': 1, 
        'img_size': (32, 32),
        'T_in': 3,
        'T_out': 3,
        'hidden_dim': 64,
        'dropout': 0.1
    }
    
    # 测试不同的模型
    models_to_test = [
        ('SwinTemporal', {'mode': 'ar'}),
        ('PhysicsTransformer', {
            'pde_type': 'heat',
            'num_heads': 4,
            'num_layers': 2,
            'constraint_weights': {
                'pde_residual': 1.0,
                'energy_conservation': 0.5,
                'boundary_condition': 0.3
            },
            'mode': 'ar'
        })
    ]
    
    # 生成测试数据
    input_data, _ = generate_heat_equation_data(num_samples=4, spatial_size=(32, 32), time_steps=10)
    x_test = input_data[:, :3]
    
    results = {}
    
    for model_name, extra_config in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            
            # 合并配置
            config = {**base_config, **extra_config}
            
            # 创建模型
            model = create_model(model_name, **config)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                predictions = model(x_test, T_out=3, return_dict=False)
            
            # 计算基本指标
            mse = predictions.var().item()  # 简化指标
            param_count = sum(p.numel() for p in model.parameters())
            
            results[model_name] = {
                'predictions': predictions,
                'mse': mse,
                'params': param_count,
                'config': config
            }
            
            print(f"  Parameters: {param_count:,}")
            print(f"  Prediction variance: {mse:.6f}")
            print(f"  Prediction shape: {predictions.shape}")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def demonstrate_advanced_features():
    """演示高级功能"""
    print("\nDemonstrating Advanced Features")
    print("=" * 50)
    
    # 创建带物理约束的模型
    config = {
        'in_channels': 1,
        'out_channels': 1,
        'img_size': (32, 32),
        'T_in': 5,
        'T_out': 5,
        'hidden_dim': 96,
        'num_heads': 4,
        'num_layers': 3,
        'pde_type': 'heat',
        'constraint_weights': {
            'pde_residual': 1.0,
            'energy_conservation': 0.8,
            'boundary_condition': 0.5,
            'causality': 0.3,
            'smoothness': 0.2
        },
        'use_frequency_encoding': True,
        'mode': 'ar'
    }
    
    model = create_model('PhysicsTransformer', **config)
    
    # 生成测试数据
    input_data, target_data = generate_heat_equation_data(
        num_samples=2, spatial_size=(32, 32), time_steps=15
    )
    
    # 测试不同的推理模式
    print("Testing different inference modes...")
    
    model.eval()
    with torch.no_grad():
        x_input = input_data[:, :5]
        
        # 模式1：自回归预测
        print("1. Autoregressive prediction:")
        pred_ar = model(x_input, T_out=5, return_dict=True)
        print(f"   Output keys: {list(pred_ar.keys())}")
        
        # 模式2：单步预测
        print("2. Single-step prediction:")
        model_single = create_model('PhysicsTransformer', **{**config, 'mode': 'single'})
        pred_single = model_single(x_input, return_dict=True)
        print(f"   Single-step prediction shape: {pred_single['prediction'].shape}")
        
        # 模式3：带物理信息的预测
        print("3. Prediction with physical information:")
        physical_info = {
            'velocity': torch.randn(2, 5, 32*32) * 0.01,
            'diffusion_coeff': 0.1,
            'boundary_conditions': {'left': 0.0, 'right': 0.0}
        }
        pred_physics = model(x_input, T_out=5, physical_info=physical_info, return_dict=True)
        print(f"   Physics validation: {pred_physics['physics_valid']}")
    
    return model


def main():
    """主函数"""
    print("Physics-Aware Transformer Temporal Model Demo")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 测试基础功能
        model, input_data, predictions = test_physics_transformer()
        
        # 2. 测试一致性验证
        validator, validation_results = test_temporal_consistency()
        
        # 3. 比较不同模型
        comparison_results = compare_temporal_models()
        
        # 4. 演示高级功能
        advanced_model = demonstrate_advanced_features()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        # 总结
        print("\nSummary:")
        print(f"- PhysicsTransformer model created and tested")
        print(f"- Temporal consistency validation implemented")
        print(f"- Multiple temporal models compared")
        print(f"- Advanced features demonstrated")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()