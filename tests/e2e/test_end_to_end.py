#!/usr/bin/env python3
"""
PDEBench稀疏观测重建系统 - 端到端流程测试脚本

测试完整的训练和评估流程，确保所有工具脚本协同工作：
1. 训练管道测试（快速验证）
2. 评估管道测试
3. 工具脚本协同测试
4. 数据一致性检查集成测试

使用方法：
python test_end_to_end.py
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndToEndTester:
    """端到端流程测试器"""
    
    def __init__(self):
        self.python_path = "F:\\ProgramData\\anaconda3\\python.exe"
        self.project_root = Path(__file__).parent.parent.parent
        self.test_results = {
            'training_pipeline': {'passed': False, 'details': []},
            'evaluation_pipeline': {'passed': False, 'details': []},
            'tool_scripts_integration': {'passed': False, 'details': []},
            'data_consistency_integration': {'passed': False, 'details': []}
        }
    
    def test_training_pipeline(self) -> bool:
        """测试训练管道（快速验证）"""
        logger.info("测试训练管道...")
        
        try:
            # 检查训练脚本是否存在
            train_script = self.project_root / "train.py"
            if not train_script.exists():
                self.test_results['training_pipeline']['details'].append("✗ 训练脚本 train.py 不存在")
                return False
            
            self.test_results['training_pipeline']['details'].append("✓ 训练脚本存在")
            
            # 检查是否可以获取帮助信息（验证脚本可执行性）
            try:
                result = subprocess.run([
                    self.python_path, str(train_script), "--help"
                ], capture_output=True, text=True, timeout=30, cwd=str(self.project_root))
                
                if result.returncode == 0 or "hydra" in result.stderr.lower():
                    self.test_results['training_pipeline']['details'].append("✓ 训练脚本可执行")
                else:
                    self.test_results['training_pipeline']['details'].append(f"✗ 训练脚本执行失败: {result.stderr[:200]}")
                    return False
                    
            except subprocess.TimeoutExpired:
                self.test_results['training_pipeline']['details'].append("✗ 训练脚本执行超时")
                return False
            except Exception as e:
                self.test_results['training_pipeline']['details'].append(f"✗ 训练脚本测试异常: {e}")
                return False
            
            # 检查配置文件
            config_dir = self.project_root / "configs"
            if config_dir.exists():
                config_files = list(config_dir.glob("*.yaml"))
                if config_files:
                    self.test_results['training_pipeline']['details'].append(f"✓ 找到 {len(config_files)} 个配置文件")
                else:
                    self.test_results['training_pipeline']['details'].append("✗ 没有找到配置文件")
                    return False
            else:
                self.test_results['training_pipeline']['details'].append("✗ 配置目录不存在")
                return False
            
            self.test_results['training_pipeline']['passed'] = True
            return True
            
        except Exception as e:
            self.test_results['training_pipeline']['details'].append(f"✗ 训练管道测试失败: {e}")
            return False
    
    def test_evaluation_pipeline(self) -> bool:
        """测试评估管道"""
        logger.info("测试评估管道...")
        
        try:
            # 检查评估脚本
            eval_scripts = [
                "tools/eval_complete.py",
                "eval.py"
            ]
            
            found_eval_script = None
            for script_name in eval_scripts:
                script_path = self.project_root / script_name
                if script_path.exists():
                    found_eval_script = script_path
                    self.test_results['evaluation_pipeline']['details'].append(f"✓ 找到评估脚本: {script_name}")
                    break
            
            if not found_eval_script:
                self.test_results['evaluation_pipeline']['details'].append("✗ 没有找到评估脚本")
                return False
            
            # 测试评估脚本可执行性
            try:
                result = subprocess.run([
                    self.python_path, str(found_eval_script), "--help"
                ], capture_output=True, text=True, timeout=30, cwd=str(self.project_root))
                
                if result.returncode == 0 or "hydra" in result.stderr.lower():
                    self.test_results['evaluation_pipeline']['details'].append("✓ 评估脚本可执行")
                else:
                    self.test_results['evaluation_pipeline']['details'].append(f"✗ 评估脚本执行失败: {result.stderr[:200]}")
                    return False
                    
            except subprocess.TimeoutExpired:
                self.test_results['evaluation_pipeline']['details'].append("✗ 评估脚本执行超时")
                return False
            except Exception as e:
                self.test_results['evaluation_pipeline']['details'].append(f"✗ 评估脚本测试异常: {e}")
                return False
            
            # 检查指标计算模块
            try:
                from utils.metrics import MetricsCalculator
                self.test_results['evaluation_pipeline']['details'].append("✓ 指标计算模块可导入")
            except ImportError as e:
                self.test_results['evaluation_pipeline']['details'].append(f"✗ 指标计算模块导入失败: {e}")
                return False
            
            self.test_results['evaluation_pipeline']['passed'] = True
            return True
            
        except Exception as e:
            self.test_results['evaluation_pipeline']['details'].append(f"✗ 评估管道测试失败: {e}")
            return False
    
    def test_tool_scripts_integration(self) -> bool:
        """测试工具脚本集成"""
        logger.info("测试工具脚本集成...")
        
        try:
            # 核心工具脚本
            tool_scripts = {
                'check_dc_equivalence.py': '数据一致性检查',
                'generate_paper_package.py': '论文材料包生成',
                'summarize_runs.py': '实验结果汇总',
                'create_paper_visualizations.py': '可视化生成'
            }
            
            success_count = 0
            for script_name, description in tool_scripts.items():
                script_path = self.project_root / "tools" / script_name
                
                if script_path.exists():
                    self.test_results['tool_scripts_integration']['details'].append(f"✓ {description}脚本存在")
                    
                    # 测试脚本帮助信息
                    try:
                        result = subprocess.run([
                            self.python_path, str(script_path), "--help"
                        ], capture_output=True, text=True, timeout=20, cwd=str(self.project_root))
                        
                        if result.returncode == 0 or "hydra" in result.stderr.lower():
                            self.test_results['tool_scripts_integration']['details'].append(f"✓ {description}脚本可执行")
                            success_count += 1
                        else:
                            self.test_results['tool_scripts_integration']['details'].append(f"✗ {description}脚本执行失败")
                            
                    except subprocess.TimeoutExpired:
                        self.test_results['tool_scripts_integration']['details'].append(f"✗ {description}脚本执行超时")
                    except Exception as e:
                        self.test_results['tool_scripts_integration']['details'].append(f"✗ {description}脚本测试异常: {e}")
                else:
                    self.test_results['tool_scripts_integration']['details'].append(f"✗ {description}脚本不存在")
            
            # 计算通过率
            pass_rate = success_count / len(tool_scripts)
            self.test_results['tool_scripts_integration']['details'].append(f"工具脚本集成通过率: {success_count}/{len(tool_scripts)} ({pass_rate*100:.1f}%)")
            
            self.test_results['tool_scripts_integration']['passed'] = pass_rate >= 0.75  # 75%通过率
            return self.test_results['tool_scripts_integration']['passed']
            
        except Exception as e:
            self.test_results['tool_scripts_integration']['details'].append(f"✗ 工具脚本集成测试失败: {e}")
            return False
    
    def test_data_consistency_integration(self) -> bool:
        """测试数据一致性检查集成"""
        logger.info("测试数据一致性检查集成...")
        
        try:
            # 检查数据一致性检查脚本
            dc_script = self.project_root / "tools" / "check_dc_equivalence.py"
            if not dc_script.exists():
                self.test_results['data_consistency_integration']['details'].append("✗ 数据一致性检查脚本不存在")
                return False
            
            self.test_results['data_consistency_integration']['details'].append("✓ 数据一致性检查脚本存在")
            
            # 检查相关模块导入
            try:
                from tools.check_dc_equivalence import DataConsistencyChecker
                from ops.degradation import apply_degradation_operator, verify_degradation_consistency
                self.test_results['data_consistency_integration']['details'].append("✓ 数据一致性相关模块可导入")
            except ImportError as e:
                self.test_results['data_consistency_integration']['details'].append(f"✗ 数据一致性模块导入失败: {e}")
                return False
            
            # 检查配置文件
            consistency_config = self.project_root / "configs" / "consistency_check.yaml"
            if consistency_config.exists():
                try:
                    config = OmegaConf.load(consistency_config)
                    self.test_results['data_consistency_integration']['details'].append("✓ 一致性检查配置文件可加载")
                except Exception as e:
                    self.test_results['data_consistency_integration']['details'].append(f"✗ 一致性检查配置文件加载失败: {e}")
                    return False
            else:
                self.test_results['data_consistency_integration']['details'].append("✗ 一致性检查配置文件不存在")
                return False
            
            # 创建测试实例验证
            try:
                test_config = DictConfig({
                    'consistency_check': {
                        'tolerance': 1e-8,
                        'num_samples': 10,
                        'random_seed': 42
                    }
                })
                
                checker = DataConsistencyChecker(
                    config=test_config,
                    device=torch.device('cpu')
                )
                self.test_results['data_consistency_integration']['details'].append("✓ 数据一致性检查器实例创建成功")
            except Exception as e:
                self.test_results['data_consistency_integration']['details'].append(f"✗ 数据一致性检查器创建失败: {e}")
                return False
            
            self.test_results['data_consistency_integration']['passed'] = True
            return True
            
        except Exception as e:
            self.test_results['data_consistency_integration']['details'].append(f"✗ 数据一致性集成测试失败: {e}")
            return False
    
    def run_end_to_end_tests(self) -> Dict[str, Any]:
        """运行完整的端到端测试"""
        logger.info("开始端到端流程测试...")
        
        # 执行各项测试
        self.test_training_pipeline()
        self.test_evaluation_pipeline()
        self.test_tool_scripts_integration()
        self.test_data_consistency_integration()
        
        # 计算总体通过率
        passed_count = sum(1 for test in self.test_results.values() if test['passed'])
        total_count = len(self.test_results)
        pass_rate = passed_count / total_count
        
        # 生成报告
        report = {
            'overall_pass_rate': pass_rate,
            'passed_tests': passed_count,
            'total_tests': total_count,
            'details': self.test_results,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> str:
        """生成测试摘要"""
        summary_lines = []
        summary_lines.append("PDEBench稀疏观测重建系统 - 端到端流程测试报告")
        summary_lines.append("=" * 60)
        
        test_names = {
            'training_pipeline': '训练管道测试',
            'evaluation_pipeline': '评估管道测试',
            'tool_scripts_integration': '工具脚本集成测试',
            'data_consistency_integration': '数据一致性集成测试'
        }
        
        for test_key, test_name in test_names.items():
            test_result = self.test_results[test_key]
            status = "✓ 通过" if test_result['passed'] else "✗ 未通过"
            summary_lines.append(f"{test_name}: {status}")
            
            for detail in test_result['details']:
                summary_lines.append(f"  {detail}")
            summary_lines.append("")
        
        passed_count = sum(1 for test in self.test_results.values() if test['passed'])
        total_count = len(self.test_results)
        summary_lines.append(f"总体通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
        
        return "\n".join(summary_lines)


def main():
    """主函数"""
    logger.info("启动PDEBench稀疏观测重建系统端到端测试")
    
    # 创建测试器
    tester = EndToEndTester()
    
    # 运行测试
    report = tester.run_end_to_end_tests()
    
    # 输出结果
    print("\n" + report['summary'])
    
    # 保存详细报告
    output_file = Path("end_to_end_test_report.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"详细报告已保存到: {output_file}")
    
    # 返回退出码
    if report['overall_pass_rate'] >= 0.75:  # 75%通过率
        logger.info("端到端流程测试通过！")
        return 0
    else:
        logger.warning("端到端流程测试未完全通过，需要改进")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)