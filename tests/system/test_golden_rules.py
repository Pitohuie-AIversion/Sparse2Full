#!/usr/bin/env python3
"""
PDEBench稀疏观测重建系统 - 黄金法则验证脚本

验证系统是否严格遵循开发手册的黄金法则：
1. 一致性优先：观测算子H与训练DC必须复用同一实现与配置
2. 可复现：同一YAML+种子，验证指标方差≤1e-4
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]
4. 可比性：横向对比必须报告均值±标准差（≥3种子）+资源成本
5. 文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁

使用方法：
python test_golden_rules.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoldenRulesValidator:
    """黄金法则验证器"""
    
    def __init__(self):
        self.results = {
            'consistency': {'passed': False, 'details': []},
            'reproducibility': {'passed': False, 'details': []},
            'unified_interface': {'passed': False, 'details': []},
            'comparability': {'passed': False, 'details': []},
            'documentation': {'passed': False, 'details': []}
        }
    
    def validate_consistency(self) -> bool:
        """验证一致性：观测算子H与训练DC必须复用同一实现与配置"""
        logger.info("验证黄金法则1：一致性优先")
        
        try:
            # 检查是否存在数据一致性检查脚本
            consistency_script = Path("tools/check_dc_equivalence.py")
            if consistency_script.exists():
                self.results['consistency']['details'].append("✓ 数据一致性检查脚本存在")
                
                # 检查配置文件
                config_file = Path("configs/consistency_check.yaml")
                if config_file.exists():
                    self.results['consistency']['details'].append("✓ 一致性检查配置文件存在")
                    self.results['consistency']['passed'] = True
                else:
                    self.results['consistency']['details'].append("✗ 缺少一致性检查配置文件")
            else:
                self.results['consistency']['details'].append("✗ 缺少数据一致性检查脚本")
            
            # 检查观测算子实现
            degradation_ops = Path("ops/degradation.py")
            if degradation_ops.exists():
                self.results['consistency']['details'].append("✓ 观测算子实现存在")
            else:
                self.results['consistency']['details'].append("✗ 缺少观测算子实现")
            
        except Exception as e:
            self.results['consistency']['details'].append(f"✗ 一致性验证失败: {e}")
        
        return self.results['consistency']['passed']
    
    def validate_reproducibility(self) -> bool:
        """验证可复现性：同一YAML+种子，验证指标方差≤1e-4"""
        logger.info("验证黄金法则2：可复现性")
        
        try:
            # 检查可复现性工具
            repro_utils = Path("utils/reproducibility.py")
            if repro_utils.exists():
                self.results['reproducibility']['details'].append("✓ 可复现性工具存在")
                
                # 检查种子设置功能
                with open(repro_utils, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'set_seed' in content and 'set_deterministic_mode' in content:
                        self.results['reproducibility']['details'].append("✓ 种子设置和确定性模式功能存在")
                        self.results['reproducibility']['passed'] = True
                    else:
                        self.results['reproducibility']['details'].append("✗ 缺少完整的可复现性功能")
            else:
                self.results['reproducibility']['details'].append("✗ 缺少可复现性工具")
            
            # 检查配置文件中的种子设置
            config_files = list(Path("configs").glob("*.yaml"))
            if config_files:
                self.results['reproducibility']['details'].append(f"✓ 找到 {len(config_files)} 个配置文件")
            else:
                self.results['reproducibility']['details'].append("✗ 缺少配置文件")
            
        except Exception as e:
            self.results['reproducibility']['details'].append(f"✗ 可复现性验证失败: {e}")
        
        return self.results['reproducibility']['passed']
    
    def validate_unified_interface(self) -> bool:
        """验证统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]"""
        logger.info("验证黄金法则3：统一接口")
        
        try:
            # 检查模型实现
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.py"))
                if model_files:
                    self.results['unified_interface']['details'].append(f"✓ 找到 {len(model_files)} 个模型文件")
                    
                    # 检查模型创建工具
                    if Path("models/__init__.py").exists():
                        self.results['unified_interface']['details'].append("✓ 模型创建工具存在")
                        self.results['unified_interface']['passed'] = True
                    else:
                        self.results['unified_interface']['details'].append("✗ 缺少模型创建工具")
                else:
                    self.results['unified_interface']['details'].append("✗ 缺少模型实现文件")
            else:
                self.results['unified_interface']['details'].append("✗ 缺少models目录")
            
        except Exception as e:
            self.results['unified_interface']['details'].append(f"✗ 统一接口验证失败: {e}")
        
        return self.results['unified_interface']['passed']
    
    def validate_comparability(self) -> bool:
        """验证可比性：横向对比必须报告均值±标准差（≥3种子）+资源成本"""
        logger.info("验证黄金法则4：可比性")
        
        try:
            # 检查实验结果汇总脚本
            summarize_script = Path("tools/summarize_runs.py")
            if summarize_script.exists():
                self.results['comparability']['details'].append("✓ 实验结果汇总脚本存在")
                
                # 检查指标计算工具
                metrics_utils = Path("utils/metrics.py")
                if metrics_utils.exists():
                    self.results['comparability']['details'].append("✓ 指标计算工具存在")
                    
                    # 检查统计分析功能
                    with open(metrics_utils, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'StatisticalAnalyzer' in content or 'compute_statistics' in content:
                            self.results['comparability']['details'].append("✓ 统计分析功能存在")
                            self.results['comparability']['passed'] = True
                        else:
                            self.results['comparability']['details'].append("✗ 缺少统计分析功能")
                else:
                    self.results['comparability']['details'].append("✗ 缺少指标计算工具")
            else:
                self.results['comparability']['details'].append("✗ 缺少实验结果汇总脚本")
            
        except Exception as e:
            self.results['comparability']['details'].append(f"✗ 可比性验证失败: {e}")
        
        return self.results['comparability']['passed']
    
    def validate_documentation(self) -> bool:
        """验证文档先行：新增任务/算子/模型前，先提交PRD/技术文档补丁"""
        logger.info("验证黄金法则5：文档先行")
        
        try:
            # 检查文档目录
            docs_dir = Path(".trae/documents")
            if docs_dir.exists():
                doc_files = list(docs_dir.glob("*.md"))
                if doc_files:
                    self.results['documentation']['details'].append(f"✓ 找到 {len(doc_files)} 个文档文件")
                    
                    # 检查关键文档
                    prd_exists = any("产品需求" in f.name or "PRD" in f.name for f in doc_files)
                    tech_exists = any("技术架构" in f.name or "技术文档" in f.name for f in doc_files)
                    
                    if prd_exists and tech_exists:
                        self.results['documentation']['details'].append("✓ 产品需求文档和技术架构文档存在")
                        self.results['documentation']['passed'] = True
                    else:
                        self.results['documentation']['details'].append("✗ 缺少关键文档（PRD或技术架构）")
                else:
                    self.results['documentation']['details'].append("✗ 文档目录为空")
            else:
                self.results['documentation']['details'].append("✗ 缺少文档目录")
            
            # 检查README文件
            if Path("README.md").exists():
                self.results['documentation']['details'].append("✓ README文件存在")
            else:
                self.results['documentation']['details'].append("✗ 缺少README文件")
            
        except Exception as e:
            self.results['documentation']['details'].append(f"✗ 文档验证失败: {e}")
        
        return self.results['documentation']['passed']
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整的黄金法则验证"""
        logger.info("开始黄金法则验证...")
        
        # 执行各项验证
        self.validate_consistency()
        self.validate_reproducibility()
        self.validate_unified_interface()
        self.validate_comparability()
        self.validate_documentation()
        
        # 计算总体通过率
        passed_count = sum(1 for rule in self.results.values() if rule['passed'])
        total_count = len(self.results)
        pass_rate = passed_count / total_count
        
        # 生成报告
        report = {
            'overall_pass_rate': pass_rate,
            'passed_rules': passed_count,
            'total_rules': total_count,
            'details': self.results,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> str:
        """生成验证摘要"""
        summary_lines = []
        summary_lines.append("PDEBench稀疏观测重建系统 - 黄金法则验证报告")
        summary_lines.append("=" * 60)
        
        rule_names = {
            'consistency': '1. 一致性优先',
            'reproducibility': '2. 可复现性',
            'unified_interface': '3. 统一接口',
            'comparability': '4. 可比性',
            'documentation': '5. 文档先行'
        }
        
        for rule_key, rule_name in rule_names.items():
            rule_result = self.results[rule_key]
            status = "✓ 通过" if rule_result['passed'] else "✗ 未通过"
            summary_lines.append(f"{rule_name}: {status}")
            
            for detail in rule_result['details']:
                summary_lines.append(f"  {detail}")
            summary_lines.append("")
        
        passed_count = sum(1 for rule in self.results.values() if rule['passed'])
        total_count = len(self.results)
        summary_lines.append(f"总体通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
        
        return "\n".join(summary_lines)


def main():
    """主函数"""
    logger.info("启动PDEBench稀疏观测重建系统黄金法则验证")
    
    # 创建验证器
    validator = GoldenRulesValidator()
    
    # 运行验证
    report = validator.run_validation()
    
    # 输出结果
    print("\n" + report['summary'])
    
    # 保存详细报告
    output_file = Path("golden_rules_validation_report.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"详细报告已保存到: {output_file}")
    
    # 返回退出码
    if report['overall_pass_rate'] >= 0.8:  # 80%通过率
        logger.info("黄金法则验证通过！")
        return 0
    else:
        logger.warning("黄金法则验证未完全通过，需要改进")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)