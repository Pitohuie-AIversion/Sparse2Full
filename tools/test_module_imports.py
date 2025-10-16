"""模块导入测试工具

测试所有核心模块的导入功能，确保无依赖错误。
按照开发手册要求，验证系统完整性。
"""

import sys
import traceback
from typing import Dict, List, Tuple
import importlib


def test_module_import(module_name: str) -> Tuple[bool, str]:
    """测试单个模块的导入
    
    Args:
        module_name: 模块名称
        
    Returns:
        (是否成功, 错误信息)
    """
    try:
        # 添加当前目录到Python路径
        if '.' not in sys.path:
            sys.path.insert(0, '.')
        
        importlib.import_module(module_name)
        return True, ""
    except Exception as e:
        return False, str(e)


def test_all_module_imports() -> Dict:
    """测试所有核心模块的导入
    
    Returns:
        测试结果字典
    """
    print("开始模块导入测试...")
    print("="*60)
    
    # 定义需要测试的模块列表
    core_modules = [
        # 数据相关模块
        "datasets",
        "datasets.pde_bench",
        
        # 模型相关模块
        "models",
        "models.swin_unet", 
        "models.hybrid",
        "models.mlp",
        
        # 核心算子模块
        "ops",
        "ops.degradation",
        "ops.losses", 
        "ops.metrics",
        
        # 工具模块
        "utils",
        "utils.config",
        "utils.logging",
        "utils.visualization",
    ]
    
    # 可选模块（可能不存在）
    optional_modules = [
        "tools.train",
        "tools.eval", 
        "tools.visualize",
    ]
    
    results = {
        'total_modules': len(core_modules) + len(optional_modules),
        'core_modules': len(core_modules),
        'optional_modules': len(optional_modules),
        'passed_core': 0,
        'passed_optional': 0,
        'failed_core': [],
        'failed_optional': [],
        'success_rate': 0.0
    }
    
    # 测试核心模块
    print("测试核心模块:")
    print("-" * 40)
    
    for module_name in core_modules:
        success, error_msg = test_module_import(module_name)
        
        if success:
            print(f"✅ {module_name}")
            results['passed_core'] += 1
        else:
            print(f"❌ {module_name}: {error_msg}")
            results['failed_core'].append({
                'module': module_name,
                'error': error_msg
            })
    
    # 测试可选模块
    print("\n测试可选模块:")
    print("-" * 40)
    
    for module_name in optional_modules:
        success, error_msg = test_module_import(module_name)
        
        if success:
            print(f"✅ {module_name}")
            results['passed_optional'] += 1
        else:
            print(f"⚠️  {module_name}: {error_msg}")
            results['failed_optional'].append({
                'module': module_name,
                'error': error_msg
            })
    
    # 计算成功率
    total_passed = results['passed_core'] + results['passed_optional']
    results['success_rate'] = total_passed / results['total_modules']
    
    return results


def test_specific_imports() -> Dict:
    """测试特定的重要导入
    
    Returns:
        测试结果
    """
    print("\n测试特定重要导入:")
    print("-" * 40)
    
    specific_tests = []
    
    # 测试观测算子H
    try:
        from ops.degradation import apply_degradation_operator
        print("✅ 观测算子H导入成功")
        specific_tests.append(('degradation_operator', True, ''))
    except Exception as e:
        print(f"❌ 观测算子H导入失败: {e}")
        specific_tests.append(('degradation_operator', False, str(e)))
    
    # 测试损失函数
    try:
        from ops.losses import compute_total_loss
        print("✅ 损失函数导入成功")
        specific_tests.append(('loss_functions', True, ''))
    except Exception as e:
        print(f"❌ 损失函数导入失败: {e}")
        specific_tests.append(('loss_functions', False, str(e)))
    
    # 测试评估指标
    try:
        from ops.metrics import compute_all_metrics
        print("✅ 评估指标导入成功")
        specific_tests.append(('metrics', True, ''))
    except Exception as e:
        print(f"❌ 评估指标导入失败: {e}")
        specific_tests.append(('metrics', False, str(e)))
    
    # 测试模型
    try:
        from models.swin_unet import SwinUNet
        from models.hybrid import HybridModel
        from models.mlp import MLPModel
        print("✅ 所有模型导入成功")
        specific_tests.append(('models', True, ''))
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        specific_tests.append(('models', False, str(e)))
    
    # 测试PyTorch相关
    try:
        import torch
        import torch.nn.functional as F
        print(f"✅ PyTorch {torch.__version__} 导入成功")
        specific_tests.append(('pytorch', True, ''))
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        specific_tests.append(('pytorch', False, str(e)))
    
    return {
        'tests': specific_tests,
        'passed': sum(1 for _, success, _ in specific_tests if success),
        'total': len(specific_tests)
    }


def print_summary_report(general_results: Dict, specific_results: Dict):
    """打印汇总报告"""
    print("\n" + "="*60)
    print("模块导入测试汇总报告")
    print("="*60)
    
    print(f"总模块数: {general_results['total_modules']}")
    print(f"核心模块: {general_results['core_modules']} (通过: {general_results['passed_core']})")
    print(f"可选模块: {general_results['optional_modules']} (通过: {general_results['passed_optional']})")
    print(f"整体成功率: {general_results['success_rate']:.1%}")
    
    print(f"\n特定导入测试: {specific_results['passed']}/{specific_results['total']} 通过")
    
    # 核心模块失败情况
    if general_results['failed_core']:
        print(f"\n❌ 核心模块失败 ({len(general_results['failed_core'])} 个):")
        for failure in general_results['failed_core']:
            print(f"  - {failure['module']}: {failure['error']}")
    
    # 可选模块失败情况
    if general_results['failed_optional']:
        print(f"\n⚠️  可选模块失败 ({len(general_results['failed_optional'])} 个):")
        for failure in general_results['failed_optional']:
            print(f"  - {failure['module']}: {failure['error']}")
    
    # 特定导入失败情况
    failed_specific = [test for test in specific_results['tests'] if not test[1]]
    if failed_specific:
        print(f"\n❌ 特定导入失败:")
        for name, _, error in failed_specific:
            print(f"  - {name}: {error}")
    
    # 总体评估
    core_success_rate = general_results['passed_core'] / general_results['core_modules']
    specific_success_rate = specific_results['passed'] / specific_results['total']
    
    print(f"\n总体评估:")
    if core_success_rate >= 0.9 and specific_success_rate >= 0.8:
        print("✅ 系统模块导入状态良好，可以进行后续测试")
        overall_status = "PASS"
    elif core_success_rate >= 0.7:
        print("⚠️  系统模块导入存在部分问题，建议修复后继续")
        overall_status = "WARNING"
    else:
        print("❌ 系统模块导入存在严重问题，需要修复")
        overall_status = "FAIL"
    
    return overall_status


def main():
    """主函数"""
    try:
        # 运行一般模块导入测试
        general_results = test_all_module_imports()
        
        # 运行特定导入测试
        specific_results = test_specific_imports()
        
        # 打印汇总报告
        overall_status = print_summary_report(general_results, specific_results)
        
        # 保存结果
        import json
        import os
        
        results = {
            'general_results': general_results,
            'specific_results': specific_results,
            'overall_status': overall_status,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
        }
        
        os.makedirs('runs/module_tests', exist_ok=True)
        with open('runs/module_tests/import_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: runs/module_tests/import_test_results.json")
        
        # 返回适当的退出码
        if overall_status == "FAIL":
            sys.exit(1)
        elif overall_status == "WARNING":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()