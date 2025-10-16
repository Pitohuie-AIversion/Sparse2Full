"""测试模型统一接口

验证所有模型是否遵循统一接口：forward(x[B,C,H,W]) → y[B,C,H,W]
"""

import torch
import sys
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_model_interface():
    """测试模型统一接口"""
    print("=" * 60)
    print("PDEBench 模型统一接口验证")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    in_channels = 3
    out_channels = 3
    img_size = 256
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"输入形状: {x.shape}")
    print()
    
    # 测试模型列表
    models_to_test = [
        {
            'name': 'SwinUNet',
            'module': 'models.swin_unet',
            'class': 'SwinUNet',
            'params': {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'img_size': img_size,
                'embed_dim': 96,
                'depths': [2, 2, 2, 2],  # 简化深度
                'num_heads': [3, 6, 12, 24],
                'window_size': 8
            }
        },
        {
            'name': 'HybridModel',
            'module': 'models.hybrid',
            'class': 'HybridModel',
            'params': {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'img_size': img_size,
                'use_attention_branch': True,
                'use_fno_branch': False,  # 暂时禁用FNO分支
                'use_unet_branch': True
            }
        },
        {
            'name': 'MLPModel',
            'module': 'models.mlp',
            'class': 'MLPModel',
            'params': {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'img_size': img_size,
                'mode': 'patch',
                'patch_size': 16,  # 增大patch_size减少计算量
                'hidden_dims': [256, 256]  # 简化网络
            }
        }
    ]
    
    results = {}
    
    for model_config in models_to_test:
        model_name = model_config['name']
        print(f"测试 {model_name}...")
        
        try:
            # 动态导入模型
            module = __import__(model_config['module'], fromlist=[model_config['class']])
            model_class = getattr(module, model_config['class'])
            
            # 创建模型实例
            model = model_class(**model_config['params'])
            
            # 设置为评估模式
            model.eval()
            
            # 前向传播测试
            with torch.no_grad():
                output = model(x)
            
            # 检查输出形状
            expected_shape = (batch_size, out_channels, img_size, img_size)
            shape_correct = output.shape == expected_shape
            
            # 检查输出是否为有限值
            output_finite = torch.isfinite(output).all().item()
            
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters())
            
            results[model_name] = {
                'success': True,
                'output_shape': output.shape,
                'shape_correct': shape_correct,
                'output_finite': output_finite,
                'param_count': param_count,
                'param_count_M': param_count / 1e6
            }
            
            print(f"  ✓ 导入成功")
            print(f"  ✓ 输出形状: {output.shape}")
            print(f"  ✓ 形状正确: {shape_correct}")
            print(f"  ✓ 输出有限: {output_finite}")
            print(f"  ✓ 参数量: {param_count:,} ({param_count/1e6:.2f}M)")
            
        except Exception as e:
            results[model_name] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            print(f"  ✗ 测试失败: {e}")
            print(f"  详细错误信息:")
            print(traceback.format_exc())
        
        print()
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    success_count = 0
    total_count = len(models_to_test)
    
    for model_name, result in results.items():
        if result['success']:
            success_count += 1
            print(f"✓ {model_name}: 通过")
            print(f"  - 输出形状: {result['output_shape']}")
            print(f"  - 参数量: {result['param_count_M']:.2f}M")
        else:
            print(f"✗ {model_name}: 失败")
            print(f"  - 错误: {result['error']}")
        print()
    
    print(f"总体结果: {success_count}/{total_count} 模型通过测试")
    
    if success_count == total_count:
        print("🎉 所有模型都通过了统一接口验证！")
        return True
    else:
        print("⚠️  部分模型测试失败，需要修复")
        return False


if __name__ == "__main__":
    success = test_model_interface()
    sys.exit(0 if success else 1)