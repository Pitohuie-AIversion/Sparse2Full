#!/usr/bin/env python3
"""
PDEBench稀疏观测重建系统 - 测试运行器

统一运行所有测试，生成测试报告和覆盖率统计。
遵循开发手册的测试要求，确保系统质量。
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

def run_command(cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
    """运行命令并返回结果"""
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        end_time = time.time()
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': end_time - start_time
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out after 300 seconds',
            'duration': 300
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': 0
        }

def run_test_suite(test_file: str, python_path: str, project_root: Path) -> Dict[str, Any]:
    """运行单个测试套件"""
    print(f"\n{'='*60}")
    print(f"运行测试: {test_file}")
    print(f"{'='*60}")
    
    cmd = [python_path, "-m", "pytest", f"tests/{test_file}", "-v", "--tb=short"]
    result = run_command(cmd, cwd=project_root)
    
    if result['success']:
        print(f"✅ {test_file} 测试通过")
    else:
        print(f"❌ {test_file} 测试失败")
        if result['stderr']:
            print(f"错误信息: {result['stderr']}")
    
    print(f"执行时间: {result['duration']:.2f}秒")
    
    return result

def run_coverage_analysis(python_path: str, project_root: Path) -> Dict[str, Any]:
    """运行覆盖率分析"""
    print(f"\n{'='*60}")
    print("运行覆盖率分析")
    print(f"{'='*60}")
    
    # 安装coverage包（如果需要）
    install_cmd = [python_path, "-m", "pip", "install", "coverage", "pytest-cov"]
    install_result = run_command(install_cmd, cwd=project_root)
    
    if not install_result['success']:
        print("⚠️  无法安装coverage包，跳过覆盖率分析")
        return {'success': False, 'message': 'Coverage package not available'}
    
    # 运行覆盖率测试
    coverage_cmd = [
        python_path, "-m", "pytest",
        "--cov=models",
        "--cov=datasets", 
        "--cov=ops",
        "--cov=utils",
        "--cov=tools",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "tests/",
        "-v"
    ]
    
    result = run_command(coverage_cmd, cwd=project_root)
    
    if result['success']:
        print("✅ 覆盖率分析完成")
        print("📊 覆盖率报告已生成到 htmlcov/ 目录")
    else:
        print("❌ 覆盖率分析失败")
    
    return result

def generate_test_report(results: Dict[str, Dict[str, Any]], output_file: Path):
    """生成测试报告"""
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration'] for r in results.values())
    
    report = f"""# PDEBench稀疏观测重建系统 - 测试报告

## 测试概览
- 总测试套件数: {total_tests}
- 通过测试: {passed_tests}
- 失败测试: {failed_tests}
- 成功率: {passed_tests/total_tests*100:.1f}%
- 总执行时间: {total_duration:.2f}秒

## 详细结果

"""
    
    for test_name, result in results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        report += f"### {test_name}\n"
        report += f"- 状态: {status}\n"
        report += f"- 执行时间: {result['duration']:.2f}秒\n"
        
        if not result['success']:
            report += f"- 错误信息: {result['stderr'][:200]}...\n"
        
        report += "\n"
    
    # 写入报告文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 测试报告已生成: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDEBench测试运行器')
    parser.add_argument('--python', type=str, default='python', help='Python解释器路径')
    parser.add_argument('--coverage', action='store_true', help='运行覆盖率分析')
    parser.add_argument('--output', type=str, default='test_report.md', help='测试报告输出文件')
    parser.add_argument('--tests', nargs='*', help='指定要运行的测试文件')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 默认测试套件
    default_test_suites = [
        'test_e2e_comprehensive.py',
        'test_model_interface_consistency.py', 
        'test_data_pipeline.py',
        'unit/test_models.py',
        'unit/test_datasets.py',
        'unit/test_losses.py',
        'unit/test_metrics.py',
        'unit/test_ops.py'
    ]
    
    # 确定要运行的测试
    test_suites = args.tests if args.tests else default_test_suites
    
    print("🚀 开始运行PDEBench测试套件")
    print(f"项目根目录: {project_root}")
    print(f"Python解释器: {args.python}")
    print(f"测试套件: {test_suites}")
    
    # 运行测试
    results = {}
    
    for test_file in test_suites:
        test_path = project_root / "tests" / test_file
        if test_path.exists():
            results[test_file] = run_test_suite(test_file, args.python, project_root)
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
            results[test_file] = {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test file not found: {test_file}',
                'duration': 0
            }
    
    # 运行覆盖率分析
    if args.coverage:
        coverage_result = run_coverage_analysis(args.python, project_root)
        results['coverage_analysis'] = coverage_result
    
    # 生成测试报告
    output_path = project_root / args.output
    generate_test_report(results, output_path)
    
    # 打印总结
    total_tests = len([r for k, r in results.items() if k != 'coverage_analysis'])
    passed_tests = sum(1 for k, r in results.items() if k != 'coverage_analysis' and r['success'])
    
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"总测试套件: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("💥 部分测试失败，请检查测试报告")
        sys.exit(1)

if __name__ == "__main__":
    main()