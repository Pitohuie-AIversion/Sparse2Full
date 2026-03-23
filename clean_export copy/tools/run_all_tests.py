#!/usr/bin/env python3
"""
测试协调器
运行所有测试套件并生成综合报告
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """单个测试结果"""
    name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    execution_time: float
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None

@dataclass
class TestSuite:
    """测试套件"""
    name: str
    script_path: str
    description: str
    required_files: List[str]
    optional: bool = False
    timeout: int = 1800  # 30分钟默认超时

@dataclass
class TestReport:
    """测试报告"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time: float
    results: List[TestResult]
    summary: Dict[str, Any]

class TestRunner:
    """测试运行器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        # 定义测试套件
        self.test_suites = self._define_test_suites()
        
        # 检查结果目录
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def _define_test_suites(self) -> List[TestSuite]:
        """定义测试套件"""
        base_path = Path(__file__).parent
        
        return [
            TestSuite(
                name="unit_tests",
                script_path=str(base_path / ".." / "tests" / "test_train_real_data_ar_refactored.py"),
                description="单元测试套件",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml"
                ],
                timeout=600  # 10分钟
            ),
            TestSuite(
                name="validation_tool",
                script_path=str(base_path / "validate_refactored_script.py"),
                description="重构脚本验证工具",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml"
                ],
                timeout=300  # 5分钟
            ),
            TestSuite(
                name="integration_consistency",
                script_path=str(base_path / "test_integration_consistency.py"),
                description="集成一致性测试",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml"
                ],
                timeout=900  # 15分钟
            ),
            TestSuite(
                name="benchmark_performance",
                script_path=str(base_path / "benchmark_refactored_script.py"),
                description="重构脚本基准测试",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml"
                ],
                timeout=1800  # 30分钟
            ),
            TestSuite(
                name="performance_comparison",
                script_path=str(base_path / "benchmark_performance_comparison.py"),
                description="性能对比分析",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml"
                ],
                timeout=3600  # 60分钟
            ),
            TestSuite(
                name="code_quality_check",
                script_path=str(base_path / "code_quality_check.py"),
                description="代码质量检查",
                required_files=[
                    "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py"
                ],
                optional=True,  # 可选测试
                timeout=300  # 5分钟
            )
        ]
    
    def check_prerequisites(self, test_suite: TestSuite) -> Tuple[bool, Optional[str]]:
        """检查测试先决条件"""
        # 检查必需文件
        for required_file in test_suite.required_files:
            if not Path(required_file).exists():
                return False, f"必需文件不存在: {required_file}"
        
        # 检查脚本文件
        if not Path(test_suite.script_path).exists():
            return False, f"测试脚本不存在: {test_suite.script_path}"
        
        # 检查Python环境
        try:
            import torch
            import numpy as np
            import yaml
        except ImportError as e:
            return False, f"缺少必需的Python包: {e}"
        
        return True, None
    
    def run_single_test(self, test_suite: TestSuite) -> TestResult:
        """运行单个测试套件"""
        logger.info(f"开始运行测试: {test_suite.name}")
        
        start_time = time.time()
        
        # 检查先决条件
        prerequisites_ok, error_msg = self.check_prerequisites(test_suite)
        if not prerequisites_ok:
            logger.warning(f"跳过测试 {test_suite.name}: {error_msg}")
            return TestResult(
                name=test_suite.name,
                status='skipped',
                execution_time=0,
                error_message=error_msg
            )
        
        # 准备输出文件
        output_file = self.results_dir / f"{test_suite.name}_results.txt"
        
        try:
            # 构建命令
            cmd = [sys.executable, test_suite.script_path]
            
            # 添加输出目录参数
            test_output_dir = self.results_dir / test_suite.name
            cmd.extend(['--output', str(test_output_dir)])
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行测试
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_suite.timeout,
                cwd=str(project_root)
            )
            
            execution_time = time.time() - start_time
            
            # 保存输出
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"测试: {test_suite.name}\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n")
                f.write(f"退出码: {process.returncode}\n")
                f.write("\n标准输出:\n")
                f.write(process.stdout)
                f.write("\n标准错误:\n")
                f.write(process.stderr)
            
            # 确定状态
            if process.returncode == 0:
                status = 'passed'
                logger.info(f"✓ 测试 {test_suite.name} 通过 ({execution_time:.1f}s)")
            else:
                status = 'failed'
                logger.warning(f"✗ 测试 {test_suite.name} 失败 (退出码: {process.returncode})")
            
            return TestResult(
                name=test_suite.name,
                status=status,
                execution_time=execution_time,
                output_file=str(output_file),
                exit_code=process.returncode,
                error_message=process.stderr[-500:] if process.returncode != 0 else None  # 取最后500字符
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"✗ 测试 {test_suite.name} 超时 ({execution_time:.1f}s)")
            return TestResult(
                name=test_suite.name,
                status='error',
                execution_time=execution_time,
                output_file=str(output_file),
                error_message=f"测试超时（超过{test_suite.timeout}秒）"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ 测试 {test_suite.name} 执行错误: {e}")
            return TestResult(
                name=test_suite.name,
                status='error',
                execution_time=execution_time,
                output_file=str(output_file),
                error_message=str(e)
            )
    
    def run_all_tests(self, test_names: Optional[List[str]] = None, 
                     skip_optional: bool = False) -> TestReport:
        """运行所有测试"""
        logger.info("开始运行所有测试套件...")
        
        results = []
        total_start_time = time.time()
        
        # 筛选测试
        tests_to_run = []
        for test_suite in self.test_suites:
            # 检查测试名称筛选
            if test_names and test_suite.name not in test_names:
                continue
            
            # 检查可选测试
            if skip_optional and test_suite.optional:
                logger.info(f"跳过可选测试: {test_suite.name}")
                continue
            
            tests_to_run.append(test_suite)
        
        logger.info(f"计划运行 {len(tests_to_run)} 个测试套件")
        
        # 运行测试
        for i, test_suite in enumerate(tests_to_run, 1):
            logger.info(f"\n[{i}/{len(tests_to_run)}] 运行测试: {test_suite.name}")
            logger.info(f"描述: {test_suite.description}")
            
            result = self.run_single_test(test_suite)
            results.append(result)
            
            # 短暂休息以避免系统过载
            time.sleep(1)
        
        total_execution_time = time.time() - total_start_time
        
        # 统计结果
        passed_tests = sum(1 for r in results if r.status == 'passed')
        failed_tests = sum(1 for r in results if r.status == 'failed')
        skipped_tests = sum(1 for r in results if r.status == 'skipped')
        error_tests = sum(1 for r in results if r.status == 'error')
        
        # 生成总结
        summary = {
            'pass_rate': (passed_tests / max(len(results), 1)) * 100,
            'total_execution_time': total_execution_time,
            'average_test_time': np.mean([r.execution_time for r in results]) if results else 0,
            'longest_test': max(results, key=lambda r: r.execution_time).name if results else None,
            'failed_test_names': [r.name for r in results if r.status == 'failed'],
            'error_test_names': [r.name for r in results if r.status == 'error'],
            'skipped_test_names': [r.name for r in results if r.status == 'skipped']
        }
        
        report = TestReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_execution_time=total_execution_time,
            results=results,
            summary=summary
        )
        
        return report

class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, report: TestReport):
        self.report = report
    
    def generate_reports(self, output_dir: Path):
        """生成所有报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        self._generate_text_report(output_dir / "test_report.txt")
        
        # 生成JSON报告
        self._generate_json_report(output_dir / "test_results.json")
        
        # 生成HTML报告
        self._generate_html_report(output_dir / "test_report.html")
        
        # 生成详细分析报告
        self._generate_analysis_report(output_dir / "test_analysis.md")
        
        # 生成失败测试详细报告
        if self.report.failed_tests > 0 or self.report.error_tests > 0:
            self._generate_failure_report(output_dir / "failed_tests_report.md")
        
        logger.info(f"测试报告已生成: {output_dir}")
    
    def _generate_text_report(self, output_file: Path):
        """生成文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("测试套件执行报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"测试时间: {self.report.timestamp}\n")
            f.write(f"总测试数: {self.report.total_tests}\n")
            f.write(f"通过测试: {self.report.passed_tests}\n")
            f.write(f"失败测试: {self.report.failed_tests}\n")
            f.write(f"跳过测试: {self.report.skipped_tests}\n")
            f.write(f"错误测试: {self.report.error_tests}\n")
            f.write(f"总执行时间: {self.report.total_execution_time:.1f}秒\n")
            f.write(f"通过率: {self.report.summary['pass_rate']:.1f}%\n\n")
            
            # 详细结果
            f.write("详细测试结果:\n")
            f.write("-" * 60 + "\n\n")
            
            for result in self.report.results:
                status_symbol = {
                    'passed': '✓',
                    'failed': '✗',
                    'skipped': '○',
                    'error': '⚠'
                }.get(result.status, '?')
                
                f.write(f"{status_symbol} {result.name:20} {result.status:8} {result.execution_time:6.1f}s")
                if result.error_message:
                    f.write(f" - {result.error_message[:100]}...")
                f.write("\n")
            
            # 总结
            f.write(f"\n最长测试: {self.report.summary['longest_test']} ({max(r.execution_time for r in self.report.results):.1f}s)\n")
            
            if self.report.summary['failed_test_names']:
                f.write(f"失败测试: {', '.join(self.report.summary['failed_test_names'])}\n")
            
            if self.report.summary['error_test_names']:
                f.write(f"错误测试: {', '.join(self.report.summary['error_test_names'])}\n")
            
            if self.report.summary['skipped_test_names']:
                f.write(f"跳过测试: {', '.join(self.report.summary['skipped_test_names'])}\n")
    
    def _generate_json_report(self, output_file: Path):
        """生成JSON报告"""
        report_data = {
            'metadata': {
                'timestamp': self.report.timestamp,
                'total_tests': self.report.total_tests,
                'passed_tests': self.report.passed_tests,
                'failed_tests': self.report.failed_tests,
                'skipped_tests': self.report.skipped_tests,
                'error_tests': self.report.error_tests,
                'total_execution_time': self.report.total_execution_time,
                'pass_rate': self.report.summary['pass_rate']
            },
            'summary': self.report.summary,
            'results': [asdict(result) for result in self.report.results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self, output_file: Path):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>测试套件执行报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
        }}
        .metric {{
            display: inline-block;
            margin-right: 30px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table th,
        .results-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .results-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .passed {{
            color: #28a745;
            font-weight: bold;
        }}
        .failed {{
            color: #dc3545;
            font-weight: bold;
        }}
        .skipped {{
            color: #6c757d;
            font-weight: bold;
        }}
        .error {{
            color: #fd7e14;
            font-weight: bold;
        }}
        .error-message {{
            color: #dc3545;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #28a745;
            transition: width 0.3s ease;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>测试套件执行报告</h1>
        
        <div class="summary">
            <h2>测试概览</h2>
            <div class="metric">
                <div class="metric-value">{self.report.total_tests}</div>
                <div class="metric-label">总测试数</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #28a745;">{self.report.passed_tests}</div>
                <div class="metric-label">通过测试</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #dc3545;">{self.report.failed_tests}</div>
                <div class="metric-label">失败测试</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #6c757d;">{self.report.skipped_tests}</div>
                <div class="metric-label">跳过测试</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #fd7e14;">{self.report.error_tests}</div>
                <div class="metric-label">错误测试</div>
            </div>
            
            <div style="margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>通过率</span>
                    <span>{self.report.summary['pass_rate']:.1f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {self.report.summary['pass_rate']}%"></div>
                </div>
            </div>
            
            <p style="margin-top: 10px;">
                <strong>总执行时间:</strong> {self.report.total_execution_time:.1f}秒 | 
                <strong>平均测试时间:</strong> {self.report.summary['average_test_time']:.1f}秒
            </p>
        </div>
        
        <h2>详细结果</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>测试名称</th>
                    <th>状态</th>
                    <th>执行时间</th>
                    <th>错误信息</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for result in self.report.results:
            status_class = result.status
            status_text = {
                'passed': '通过',
                'failed': '失败',
                'skipped': '跳过',
                'error': '错误'
            }.get(result.status, '未知')
            
            html_content += f"""
                <tr>
                    <td>{result.name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.execution_time:.1f}s</td>
                    <td class="error-message">{result.error_message or '-'}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
        
        <div class="footer">
            <p>测试时间: {self.report.timestamp}</p>
            <p>最长测试: {self.report.summary['longest_test']} ({max(r.execution_time for r in self.report.results):.1f}s)</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_analysis_report(self, output_file: Path):
        """生成详细分析报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 测试套件详细分析\n\n")
            
            # 总体分析
            f.write("## 总体分析\n\n")
            f.write(f"- 总测试数: {self.report.total_tests}\n")
            f.write(f"- 通过测试: {self.report.passed_tests}\n")
            f.write(f"- 失败测试: {self.report.failed_tests}\n")
            f.write(f"- 跳过测试: {self.report.skipped_tests}\n")
            f.write(f"- 错误测试: {self.report.error_tests}\n")
            f.write(f"- 通过率: {self.report.summary['pass_rate']:.1f}%\n")
            f.write(f"- 总执行时间: {self.report.total_execution_time:.1f}秒\n")
            f.write(f"- 平均测试时间: {self.report.summary['average_test_time']:.1f}秒\n\n")
            
            # 测试时间分析
            f.write("## 测试时间分析\n\n")
            execution_times = [r.execution_time for r in self.report.results if r.status != 'skipped']
            if execution_times:
                f.write(f"- 最快测试: {min(execution_times):.1f}秒 ({min(self.report.results, key=lambda r: r.execution_time if r.status != 'skipped' else float('inf')).name})\n")
                f.write(f"- 最慢测试: {max(execution_times):.1f}秒 ({max(self.report.results, key=lambda r: r.execution_time if r.status != 'skipped' else 0).name})\n")
                f.write(f"- 平均时间: {np.mean(execution_times):.1f}秒\n")
                f.write(f"- 时间标准差: {np.std(execution_times):.1f}秒\n\n")
            
            # 失败分析
            if self.report.failed_tests > 0:
                f.write("## 失败分析\n\n")
                f.write("以下测试失败，需要关注:\n\n")
                
                for result in self.report.results:
                    if result.status == 'failed':
                        f.write(f"### {result.name}\n\n")
                        f.write(f"**执行时间**: {result.execution_time:.1f}秒\n\n")
                        f.write(f"**退出码**: {result.exit_code}\n\n")
                        if result.error_message:
                            f.write(f"**错误信息**: {result.error_message}\n\n")
                        
                        # 提供建议
                        suggestions = self._get_improvement_suggestions(result)
                        if suggestions:
                            f.write("**建议**:\n")
                            for suggestion in suggestions:
                                f.write(f"- {suggestion}\n")
                            f.write("\n")
            
            # 错误分析
            if self.report.error_tests > 0:
                f.write("## 错误分析\n\n")
                f.write("以下测试执行出错:\n\n")
                
                for result in self.report.results:
                    if result.status == 'error':
                        f.write(f"### {result.name}\n\n")
                        f.write(f"**执行时间**: {result.execution_time:.1f}秒\n\n")
                        if result.error_message:
                            f.write(f"**错误信息**: {result.error_message}\n\n")
            
            # 跳过分析
            if self.report.skipped_tests > 0:
                f.write("## 跳过分析\n\n")
                f.write("以下测试被跳过:\n\n")
                
                for result in self.report.results:
                    if result.status == 'skipped':
                        f.write(f"- **{result.name}**: {result.error_message}\n")
                f.write("\n")
            
            # 成功分析
            if self.report.passed_tests > 0:
                f.write("## 成功分析\n\n")
                f.write("以下测试成功通过:\n\n")
                
                passed_results = [r for r in self.report.results if r.status == 'passed']
                for result in passed_results[:10]:  # 只显示前10个
                    f.write(f"- **{result.name}**: {result.execution_time:.1f}秒\n")
                
                if len(passed_results) > 10:
                    f.write(f"\n... 还有 {len(passed_results) - 10} 个测试通过\n")
            
            f.write("\n---\n")
            f.write(f"分析生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _generate_failure_report(self, output_file: Path):
        """生成失败测试详细报告"""
        failed_results = [r for r in self.report.results if r.status in ['failed', 'error']]
        
        if not failed_results:
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 失败测试详细报告\n\n")
            
            for result in failed_results:
                f.write(f"## {result.name} ({result.status})\n\n")
                f.write(f"**执行时间**: {result.execution_time:.1f}秒\n\n")
                f.write(f"**状态**: {result.status}\n\n")
                
                if result.exit_code is not None:
                    f.write(f"**退出码**: {result.exit_code}\n\n")
                
                if result.error_message:
                    f.write(f"**错误信息**:\n```\n{result.error_message}\n```\n\n")
                
                if result.output_file and Path(result.output_file).exists():
                    f.write(f"**输出文件**: {result.output_file}\n\n")
                    
                    # 读取输出文件内容
                    try:
                        with open(result.output_file, 'r', encoding='utf-8') as output_f:
                            content = output_f.read()
                            if len(content) > 1000:
                                content = content[:1000] + "\n... (内容过长，已截断)"
                            f.write(f"**详细输出**:\n```\n{content}\n```\n\n")
                    except Exception as e:
                        f.write(f"**读取输出文件失败**: {e}\n\n")
                
                f.write("---\n\n")
            
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _get_improvement_suggestions(self, result: TestResult) -> List[str]:
        """获取改进建议"""
        suggestions = []
        
        if result.exit_code == 1:
            suggestions.append("检查测试逻辑和断言条件")
        elif result.exit_code == 2:
            suggestions.append("检查Python语法错误")
        elif result.exit_code == 127:
            suggestions.append("检查系统命令或脚本路径")
        
        if result.name == "unit_tests":
            suggestions.append("检查单元测试覆盖率和测试用例完整性")
            suggestions.append("验证重构代码的函数签名和返回值")
        elif result.name == "validation_tool":
            suggestions.append("检查重构脚本的代码结构和模块化程度")
            suggestions.append("验证配置文件的正确性")
        elif result.name == "integration_consistency":
            suggestions.append("检查重构版本与原始版本的功能一致性")
            suggestions.append("验证数值计算的精度和容差设置")
        elif result.name == "benchmark_performance":
            suggestions.append("检查性能基准测试的配置参数")
            suggestions.append("验证系统资源是否充足")
        elif result.name == "performance_comparison":
            suggestions.append("检查性能对比测试的环境一致性")
            suggestions.append("验证原始脚本和重构脚本的可用性")
        
        if not suggestions:
            suggestions.append("查看详细的错误输出和日志信息")
            suggestions.append("检查相关的配置文件和依赖项")
        
        return suggestions

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试协调器")
    parser.add_argument("--output", type=str, default="test_results",
                       help="输出目录")
    parser.add_argument("--tests", nargs='*',
                       help="指定运行的测试名称（如果不指定则运行所有）")
    parser.add_argument("--skip-optional", action='store_true',
                       help="跳过可选测试")
    parser.add_argument("--verbose", action='store_true',
                       help="详细输出")
    parser.add_argument("--quiet", action='store_true',
                       help="安静模式，只显示错误")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("开始运行测试协调器...")
    
    # 创建测试运行器
    runner = TestRunner(args.output)
    
    # 运行测试
    report = runner.run_all_tests(
        test_names=args.tests,
        skip_optional=args.skip_optional
    )
    
    # 生成报告
    reporter = TestReporter(report)
    reporter.generate_reports(args.output)
    
    # 打印总结
    logger.info(f"\n测试执行完成!")
    logger.info(f"总测试数: {report.total_tests}")
    logger.info(f"通过测试: {report.passed_tests}")
    logger.info(f"失败测试: {report.failed_tests}")
    logger.info(f"跳过测试: {report.skipped_tests}")
    logger.info(f"错误测试: {report.error_tests}")
    logger.info(f"通过率: {report.summary['pass_rate']:.1f}%")
    logger.info(f"总执行时间: {report.total_execution_time:.1f}秒")
    
    if report.failed_tests > 0:
        logger.warning(f"失败测试: {', '.join(report.summary['failed_test_names'])}")
    
    if report.error_tests > 0:
        logger.error(f"错误测试: {', '.join(report.summary['error_test_names'])}")
    
    logger.info(f"详细报告已保存到: {args.output}")
    
    # 返回适当的退出码
    sys.exit(0 if report.failed_tests == 0 and report.error_tests == 0 else 1)

if __name__ == "__main__":
    main()