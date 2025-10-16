#!/usr/bin/env python3
"""
测试所有工具脚本的导入状态
"""

import sys
import traceback

def test_module_import(module_name):
    """测试单个模块的导入"""
    try:
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    print("验证所有工具脚本导入...")
    print("=" * 50)
    
    # 测试各个工具模块
    modules = [
        'tools.train',
        'tools.eval', 
        'tools.check_dc_equivalence',
        'tools.generate_paper_package'
    ]
    
    success_count = 0
    failed_modules = []
    
    for module in modules:
        success, error = test_module_import(module)
        if success:
            print(f'✅ {module} - 导入成功')
            success_count += 1
        else:
            print(f'❌ {module} - 导入失败: {error}')
            failed_modules.append((module, error))
    
    print("=" * 50)
    print(f'总结: {success_count}/{len(modules)} 个模块导入成功')
    
    if failed_modules:
        print("\n失败模块详情:")
        for module, error in failed_modules:
            print(f"- {module}: {error}")
    
    return success_count == len(modules)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)