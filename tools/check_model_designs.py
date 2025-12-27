#!/usr/bin/env python3
"""
全面检查模型设计和模型加载器兼容性
"""

import sys
import os
import torch
import logging
import traceback

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from tools.training.model_loader import (
    ModelLoader, 
    create_model_with_loader, 
    list_models, 
    check_model_health
)

def check_model_design_issues():
    """检查模型设计问题"""
    print("=== 检查模型设计问题 ===")
    
    loader = ModelLoader()
    available_models = list_models()
    
    print(f"发现 {len(available_models)} 个模型")
    
    design_issues = []
    successful_models = []
    failed_models = []
    
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
    
    for model_name in available_models:
        print(f"\n检查模型: {model_name}")
        
        try:
            # 尝试创建模型
            model = create_model_with_loader(model_name, test_config)
            
            if model is None:
                print(f"  ⚠ 模型创建返回 None")
                failed_models.append(model_name)
                continue
            
            # 检查模型类型
            if not isinstance(model, torch.nn.Module):
                print(f"  ✗ 模型不是 nn.Module 实例: {type(model)}")
                design_issues.append(f"{model_name}: 不是 nn.Module 实例")
                failed_models.append(model_name)
                continue
            
            # 检查基本属性
            missing_attrs = []
            for attr in ['in_channels', 'out_channels', 'img_size']:
                if not hasattr(model, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                print(f"  ⚠ 缺少属性: {missing_attrs}")
                design_issues.append(f"{model_name}: 缺少属性 {missing_attrs}")
            
            # 健康检查
            health_report = check_model_health(model)
            
            if health_report['status'] == 'healthy':
                print(f"  ✓ 模型健康，参数数量: {health_report['parameters']:,}")
                successful_models.append(model_name)
            else:
                print(f"  ⚠ 健康检查问题: {health_report['errors']}")
                design_issues.append(f"{model_name}: 健康检查问题 - {health_report['errors']}")
                
                # 如果前向传播失败，添加到失败列表
                if not health_report['forward_pass']:
                    failed_models.append(model_name)
                else:
                    successful_models.append(model_name)
            
        except Exception as e:
            print(f"  ✗ 模型创建失败: {e}")
            failed_models.append(model_name)
            design_issues.append(f"{model_name}: 创建失败 - {str(e)[:100]}...")
    
    # 总结
    print(f"\n=== 检查结果总结 ===")
    print(f"总模型数: {len(available_models)}")
    print(f"成功模型: {len(successful_models)}")
    print(f"失败模型: {len(failed_models)}")
    print(f"设计问题: {len(design_issues)}")
    
    if successful_models:
        print(f"\n成功模型:")
        for model in successful_models[:10]:  # 只显示前10个
            print(f"  ✓ {model}")
        if len(successful_models) > 10:
            print(f"  ... 还有 {len(successful_models) - 10} 个")
    
    if failed_models:
        print(f"\n失败模型:")
        for model in failed_models[:10]:  # 只显示前10个
            print(f"  ✗ {model}")
        if len(failed_models) > 10:
            print(f"  ... 还有 {len(failed_models) - 10} 个")
    
    if design_issues:
        print(f"\n设计问题详情:")
        for issue in design_issues[:10]:  # 只显示前10个
            print(f"  ⚠ {issue}")
        if len(design_issues) > 10:
            print(f"  ... 还有 {len(design_issues) - 10} 个问题")
    
    return len(failed_models) == 0 and len(design_issues) == 0

def check_model_interfaces():
    """检查模型接口一致性"""
    print("\n=== 检查模型接口一致性 ===")
    
    from tools.training.model_loader import get_model_loader
    loader = get_model_loader()
    
    # 检查几个关键模型的接口
    key_models = ['swin_unet', 'unet', 'fno2d', 'segformer']
    
    interface_issues = []
    
    for model_name in key_models:
        try:
            model_info = loader.model_registry.get(model_name)
            if not model_info:
                interface_issues.append(f"{model_name}: 在注册表中找不到")
                continue
            
            model_class = model_info['model_class']
            
            # 检查构造函数签名
            import inspect
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.keys())
            
            # 期望的参数
            expected_params = ['self', 'in_channels', 'out_channels', 'img_size']
            missing_params = [p for p in expected_params if p not in params and p != 'self']
            
            if missing_params:
                interface_issues.append(f"{model_name}: 缺少构造函数参数 {missing_params}")
                print(f"  ⚠ {model_name}: 缺少参数 {missing_params}")
            else:
                print(f"  ✓ {model_name}: 接口正确")
            
            # 检查是否有 **kwargs
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_kwargs:
                print(f"  ⚠ {model_name}: 缺少 **kwargs 参数，可能影响兼容性")
            
        except Exception as e:
            interface_issues.append(f"{model_name}: 接口检查失败 - {e}")
            print(f"  ✗ {model_name}: 接口检查失败 - {e}")
    
    return len(interface_issues) == 0

def check_external_factory_compatibility():
    """检查外部工厂函数兼容性"""
    print("\n=== 检查外部工厂函数兼容性 ===")
    
    try:
        from models import create_model
        
        # 测试外部工厂支持的模型
        external_models = ['UNet', 'FNO2D', 'SwinUNet', 'SegFormer']
        
        test_config = {
            "in_channels": 1,
            "out_channels": 1, 
            "img_size": 128
        }
        
        success_count = 0
        
        for model_name in external_models:
            try:
                model = create_model(model_name, **test_config)
                if model is not None and isinstance(model, torch.nn.Module):
                    print(f"  ✓ {model_name}: 外部工厂成功")
                    success_count += 1
                else:
                    print(f"  ✗ {model_name}: 外部工厂返回无效模型")
            except Exception as e:
                print(f"  ✗ {model_name}: 外部工厂失败 - {e}")
        
        print(f"\n外部工厂兼容性: {success_count}/{len(external_models)} 个模型成功")
        return success_count == len(external_models)
        
    except ImportError as e:
        print(f"  ✗ 无法导入外部工厂: {e}")
        return False

def main():
    """主检查函数"""
    print("开始全面检查模型设计和兼容性...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行所有检查
    checks = [
        ("模型接口一致性", check_model_interfaces),
        ("外部工厂兼容性", check_external_factory_compatibility),
        ("模型设计和健康", check_model_design_issues)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            print(f"\n{'='*60}")
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"检查 {check_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))
    
    # 最终总结
    print(f"\n{'='*60}")
    print("最终检查结果:")
    for check_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {check_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n总计: {total_passed}/{len(checks)} 项检查通过")
    
    if total_passed == len(checks):
        print("\n🎉 所有检查通过！模型设计良好，兼容性优秀。")
        return True
    else:
        print(f"\n⚠️  有 {len(checks) - total_passed} 项检查未通过，需要关注。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)