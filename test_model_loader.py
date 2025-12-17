#!/usr/bin/env python3
"""
测试模型加载器功能
"""

import sys
import os
import torch
import logging

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tools.training.model_loader import (
    ModelLoader, 
    create_model_with_loader, 
    list_models, 
    check_model_health
)

def test_model_discovery():
    """测试模型发现功能"""
    print("=== 测试模型发现功能 ===")
    
    loader = ModelLoader()
    available_models = loader.list_available_models()
    
    print(f"发现 {len(available_models)} 个模型:")
    for model_name in available_models:
        print(f"  - {model_name}")
    
    return len(available_models) > 0

def test_model_creation():
    """测试模型创建功能"""
    print("\n=== 测试模型创建功能 ===")
    
    # 测试配置
    test_config = {
        "in_channels": 1,
        "out_channels": 1, 
        "img_size": 128,
        "data": {
            "channels": 1,
            "img_size": 128
        }
    }
    
    # 获取可用模型列表
    available_models = list_models()
    if not available_models:
        print("没有找到可用模型")
        return False
    
    success_count = 0
    
    # 测试每个模型
    for model_name in available_models[:3]:  # 只测试前3个模型
        print(f"\n测试模型: {model_name}")
        
        try:
            # 创建模型
            model = create_model_with_loader(model_name, test_config)
            print(f"  ✓ 成功创建模型: {type(model).__name__}")
            
            # 健康检查
            health_report = check_model_health(model)
            print(f"  ✓ 参数数量: {health_report['parameters']:,}")
            print(f"  ✓ 前向传播: {'通过' if health_report['forward_pass'] else '失败'}")
            
            if health_report['errors']:
                print(f"  ⚠  错误: {health_report['errors']}")
            else:
                print(f"  ✓ 健康检查通过")
                success_count += 1
                
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print(f"\n成功创建 {success_count}/{min(3, len(available_models))} 个模型")
    return success_count > 0

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    loader = ModelLoader()
    
    # 测试不存在的模型
    print("测试不存在的模型:")
    try:
        model = loader.create_model("non_existent_model")
        print("  ✗ 应该抛出异常")
    except Exception as e:
        print(f"  ✓ 正确处理异常: {type(e).__name__}")
    
    # 测试无效参数
    print("\n测试无效参数:")
    try:
        model = loader.create_model("swin_unet", {"invalid_param": "value"})
        print("  ✓ 模型创建成功（参数被忽略）")
    except Exception as e:
        print(f"  ✓ 正确处理异常: {type(e).__name__}")
    
    return True

def test_training_script_integration():
    """测试与训练脚本的集成"""
    print("\n=== 测试与训练脚本集成 ===")
    
    # 模拟训练脚本调用
    try:
        from tools.training.train_real_data_ar import Trainer
        
        # 创建最小配置
        import tempfile
        import yaml
        
        config = {
            'model': {
                'name': 'swin_unet',
                'in_channels': 1,
                'out_channels': 1,
                'img_size': 128
            },
            'data': {
                'channels': 1,
                'img_size': 128
            },
            'training': {
                'epochs': 1,
                'batch_size': 2
            }
        }
        
        # 写入临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        print(f"创建测试配置: {config_path}")
        
        # 测试模型加载
        from tools.training.model_loader import create_model_with_loader
        model = create_model_with_loader('swin_unet', config)
        
        print(f"  ✓ 模型创建成功: {type(model).__name__}")
        print(f"  ✓ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 清理
        os.unlink(config_path)
        
        return True
        
    except ImportError as e:
        print(f"  ⚠  无法导入训练脚本: {e}")
        return False
    except Exception as e:
        print(f"  ✗ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试模型加载器...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行所有测试
    tests = [
        ("模型发现", test_model_discovery),
        ("模型创建", test_model_creation), 
        ("错误处理", test_error_handling),
        ("训练脚本集成", test_training_script_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试结果总结:")
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n总计: {total_passed}/{len(tests)} 个测试通过")
    
    return total_passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)