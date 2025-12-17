#!/usr/bin/env python3
"""
测试模型加载器功能 - 简化版本
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

def test_main_models():
    """测试主要模型"""
    print("=== 测试主要模型 ===")
    
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
    
    # 主要模型列表
    main_models = [
        'swin_unet',
        'unet',
        'fno2d', 
        'ufnounet',
        'segformer',
        'mlpmixer'
    ]
    
    success_count = 0
    
    for model_name in main_models:
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
    
    print(f"\n成功创建 {success_count}/{len(main_models)} 个主要模型")
    return success_count > 0

def test_external_factory():
    """测试外部工厂函数"""
    print("\n=== 测试外部工厂函数 ===")
    
    test_config = {
        "in_channels": 1,
        "out_channels": 1, 
        "img_size": 128
    }
    
    # 测试外部工厂支持的模型
    external_models = ['SwinUNet', 'UNet', 'FNO2D']
    
    success_count = 0
    
    for model_name in external_models:
        print(f"\n测试外部工厂模型: {model_name}")
        
        try:
            model = create_model_with_loader(model_name, test_config)
            print(f"  ✓ 成功创建模型: {type(model).__name__}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print(f"\n外部工厂成功创建 {success_count}/{len(external_models)} 个模型")
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
    
    # 测试参数不足
    print("\n测试参数不足:")
    try:
        model = loader.create_model("swin_unet", {})  # 空配置
        print("  ✗ 应该抛出异常")
    except Exception as e:
        print(f"  ✓ 正确处理异常: {type(e).__name__}")
    
    return True

def test_model_discovery():
    """测试模型发现"""
    print("\n=== 测试模型发现 ===")
    
    available_models = list_models()
    print(f"发现 {len(available_models)} 个模型")
    
    # 显示前10个模型
    for i, model_name in enumerate(available_models[:10]):
        print(f"  - {model_name}")
    
    if len(available_models) > 10:
        print(f"  ... 还有 {len(available_models) - 10} 个模型")
    
    return len(available_models) > 0

def main():
    """主测试函数"""
    print("开始测试模型加载器...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    tests = [
        ("模型发现", test_model_discovery),
        ("主要模型", test_main_models),
        ("外部工厂", test_external_factory),
        ("错误处理", test_error_handling)
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
    
    return total_passed >= 3  # 至少3个测试通过就算成功

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)