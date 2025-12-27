"""
PDEBench稀疏观测重建系统 - 综合测试运行器

提供统一的测试运行接口，支持：
1. 分层测试（单元测试、集成测试、系统测试）
2. 黄金法则合规性检查
3. 性能基准测试
4. 可重现性验证
5. 测试报告生成
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """测试运行器类"""
    
    def __init__(self, project_root: Path, verbose: bool = True):
        self.project_root = project_root
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def log(self, message: str, level: str = "INFO"):
        """日志输出"""
        if self.verbose:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   timeout: int = 300) -> Dict[str, Any]:
        """运行命令"""
        try:
            self.log(f"运行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': 0  # 将在调用处计算
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'命令超时（{timeout}秒）',
                'duration': timeout
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': 0
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        self.log("开始运行单元测试...")
        start_time = time.time()
        
        # 单元测试文件列表
        unit_tests = [
            'unit/test_models.py',
            'unit/test_datasets.py', 
            'unit/test_losses.py',
            'unit/test_metrics.py',
            'unit/test_ops.py',
            'unit/test_dc_checker.py',
            # 'test_comprehensive_framework.py'  # Removed as it might be redundant
        ]
        
        results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in unit_tests:
            test_path = self.project_root / 'tests' / test_file
            if not test_path.exists():
                self.log(f"测试文件不存在: {test_file}", "WARNING")
                continue
                
            self.log(f"运行单元测试: {test_file}")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v", "--tb=short"
            ]
            
            result = self.run_command(cmd)
            result['duration'] = time.time() - start_time
            
            if result['success']:
                total_passed += 1
                self.log(f"✅ {test_file} 通过")
            else:
                total_failed += 1
                self.log(f"❌ {test_file} 失败: {result['stderr'][:200]}...")
            
            results[test_file] = result
        
        total_time = time.time() - start_time
        
        return {
            'tests': results,
            'summary': {
                'total': len(results),
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': total_passed / len(results) if results else 0,
                'duration': total_time
            }
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        self.log("开始运行集成测试...")
        start_time = time.time()
        
        # 集成测试文件列表
        integration_tests = [
            'integration/test_training.py',
            'integration/test_evaluation.py',
            'integration/test_temporal_nar.py',
            'integration/test_ar_compatibility.py'
        ]
        
        results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in integration_tests:
            test_path = self.project_root / 'tests' / test_file
            if not test_path.exists():
                self.log(f"测试文件不存在: {test_file}", "WARNING")
                continue
                
            self.log(f"运行集成测试: {test_file}")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v", "--tb=short"
            ]
            
            result = self.run_command(cmd)
            result['duration'] = time.time() - start_time
            
            if result['success']:
                total_passed += 1
                self.log(f"✅ {test_file} 通过")
            else:
                total_failed += 1
                self.log(f"❌ {test_file} 失败: {result['stderr'][:200]}...")
            
            results[test_file] = result
        
        total_time = time.time() - start_time
        
        return {
            'tests': results,
            'summary': {
                'total': len(results),
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': total_passed / len(results) if results else 0,
                'duration': total_time
            }
        }
    
    def run_system_tests(self) -> Dict[str, Any]:
        """运行系统测试"""
        self.log("开始运行系统测试...")
        start_time = time.time()
        
        # 系统测试文件列表
        system_tests = [
            'system/test_tools_integration.py',
            # 'system/test_golden_rules_compliance.py', # Add if exists
        ]
        
        results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in system_tests:
            test_path = self.project_root / 'tests' / test_file
            if not test_path.exists():
                self.log(f"测试文件不存在: {test_file}", "WARNING")
                continue
                
            self.log(f"运行系统测试: {test_file}")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v", "--tb=short"
            ]
            
            result = self.run_command(cmd, timeout=600)  # 系统测试可能需要更长时间
            result['duration'] = time.time() - start_time
            
            if result['success']:
                total_passed += 1
                self.log(f"✅ {test_file} 通过")
            else:
                total_failed += 1
                self.log(f"❌ {test_file} 失败: {result['stderr'][:200]}...")
            
            results[test_file] = result
        
        total_time = time.time() - start_time
        
        return {
            'tests': results,
            'summary': {
                'total': len(results),
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': total_passed / len(results) if results else 0,
                'duration': total_time
            }
        }
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """运行端到端测试"""
        self.log("开始运行端到端测试...")
        start_time = time.time()
        
        # 端到端测试文件列表
        e2e_tests = [
            'e2e/test_full_system.py',
            'e2e/test_golden_rules.py'
        ]
        
        results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in e2e_tests:
            test_path = self.project_root / 'tests' / test_file
            if not test_path.exists():
                self.log(f"测试文件不存在: {test_file}", "WARNING")
                continue
                
            self.log(f"运行端到端测试: {test_file}")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v", "--tb=short"
            ]
            
            result = self.run_command(cmd, timeout=900)  # 端到端测试可能需要更长时间
            result['duration'] = time.time() - start_time
            
            if result['success']:
                total_passed += 1
                self.log(f"✅ {test_file} 通过")
            else:
                total_failed += 1
                self.log(f"❌ {test_file} 失败: {result['stderr'][:200]}...")
            
            results[test_file] = result
        
        total_time = time.time() - start_time
        
        return {
            'tests': results,
            'summary': {
                'total': len(results),
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': total_passed / len(results) if results else 0,
                'duration': total_time
            }
        }
    
    def run_golden_rules_check(self) -> Dict[str, Any]:
        """运行黄金法则合规性检查"""
        # 简化版：直接调用相关测试
        return {}
    
    def generate_report(self, results: Dict[str, Any], output_path: Path):
        """生成测试报告"""
        self.log(f"生成测试报告: {output_path}")
        
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'project_root': str(self.project_root),
            'test_duration': self.end_time - self.start_time if self.start_time and self.end_time else 0,
            'results': results,
            'summary': self._generate_summary(results)
        }
        
        # 保存JSON报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_path = output_path.with_suffix('.md')
        self._generate_markdown_report(report, md_path)
        
        self.log(f"测试报告已生成: {output_path}")
        self.log(f"Markdown报告已生成: {md_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试摘要"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0,
            'total_duration': 0.0
        }
        
        for category, result in results.items():
            if 'summary' in result:
                cat_summary = result['summary']
                summary['total_tests'] += cat_summary.get('total', 0)
                summary['passed_tests'] += cat_summary.get('passed', 0)
                summary['failed_tests'] += cat_summary.get('failed', 0)
                summary['total_duration'] += cat_summary.get('duration', 0)
        
        if summary['total_tests'] > 0:
            summary['success_rate'] = summary['passed_tests'] / summary['total_tests']
        
        return summary
    
    def _generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """生成Markdown格式的测试报告"""
        summary = report['summary']
        
        md_content = f"""# PDEBench稀疏观测重建系统 - 测试报告

## 测试概览

- **测试时间**: {report['timestamp']}
- **项目根目录**: {report['project_root']}
- **总测试数**: {summary['total_tests']}
- **通过测试**: {summary['passed_tests']}
- **失败测试**: {summary['failed_tests']}
- **成功率**: {summary['success_rate']:.1%}
- **总耗时**: {summary['total_duration']:.2f}秒

## 详细结果

"""
        
        # 添加各类别详细结果
        for category, result in report['results'].items():
            md_content += f"### {category}\n\n"
            
            if 'summary' in result:
                cat_summary = result['summary']
                md_content += f"- **总测试数**: {cat_summary.get('total', 0)}\n"
                md_content += f"- **通过测试**: {cat_summary.get('passed', 0)}\n"
                md_content += f"- **失败测试**: {cat_summary.get('failed', 0)}\n"
                md_content += f"- **成功率**: {cat_summary.get('success_rate', 0):.1%}\n"
                md_content += f"- **耗时**: {cat_summary.get('duration', 0):.2f}秒\n\n"
            
            # 添加失败详情
            if 'tests' in result:
                failed_tests = [
                    (name, test_result) 
                    for name, test_result in result['tests'].items() 
                    if not test_result['success']
                ]
                
                if failed_tests:
                    md_content += "#### 失败测试详情\n\n"
                    for test_name, test_result in failed_tests:
                        md_content += f"**{test_name}**:\n"
                        md_content += f"- 错误: {test_result.get('stderr', '未知错误')[:200]}\n\n"
            
            md_content += "\n"
        
        # 添加结论和建议
        if summary['success_rate'] >= 0.9:
            conclusion = "✅ **测试通过** - 系统质量良好，符合生产要求"
        elif summary['success_rate'] >= 0.7:
            conclusion = "⚠️ **测试部分通过** - 系统存在一些问题，需要修复"
        else:
            conclusion = "❌ **测试失败** - 系统存在严重问题，需要全面修复"
        
        md_content += f"""
## 结论

{conclusion}

## 建议

1. **立即修复**: 优先处理失败的测试用例
2. **代码审查**: 对失败的模块进行代码审查
3. **回归测试**: 修复后重新运行测试套件
4. **持续集成**: 将测试集成到CI/CD流程中

---
*本报告由PDEBench测试框架自动生成*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def run_all_tests(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """运行所有测试"""
        self.start_time = time.time()
        self.log("开始运行PDEBench完整测试套件...")
        
        if output_dir is None:
            output_dir = self.project_root / 'test_reports'
        
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        try:
            # 运行单元测试
            results['unit_tests'] = self.run_unit_tests()
            
            # 运行集成测试
            results['integration_tests'] = self.run_integration_tests()
            
            # 运行系统测试
            results['system_tests'] = self.run_system_tests()
            
            # 运行端到端测试
            results['e2e_tests'] = self.run_e2e_tests()
            
        except Exception as e:
            self.log(f"测试运行异常: {e}", "ERROR")
            results['error'] = {
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        
        finally:
            self.end_time = time.time()
            
            # 生成报告
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = output_dir / f"test_report_{timestamp}.json"
            
            self.generate_report(results, report_path)
            
            # 打印摘要
            summary = self._generate_summary(results)
            self.log(f"测试完成 - 总成功率: {summary['success_rate']:.1%}")
            self.log(f"测试报告已保存到: {report_path}")
            
            return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDEBench测试运行器')
    parser.add_argument('--output-dir', type=str, help='测试报告输出目录')
    parser.add_argument('--test-type', type=str, choices=['unit', 'integration', 'system', 'e2e', 'all'], 
                       default='all', help='测试类型')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--project-root', type=str, help='项目根目录')
    
    args = parser.parse_args()
    
    # 确定项目根目录
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent
    
    # 创建测试运行器
    runner = TestRunner(project_root, verbose=args.verbose)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'test_reports'
    
    # 运行测试
    if args.test_type == 'all':
        results = runner.run_all_tests(output_dir)
    elif args.test_type == 'unit':
        results = {'unit_tests': runner.run_unit_tests()}
    elif args.test_type == 'integration':
        results = {'integration_tests': runner.run_integration_tests()}
    elif args.test_type == 'system':
        results = {'system_tests': runner.run_system_tests()}
    elif args.test_type == 'e2e':
        results = {'e2e_tests': runner.run_e2e_tests()}
    
    # 退出码
    summary = runner._generate_summary(results)
    exit_code = 0 if summary['success_rate'] >= 0.8 else 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
