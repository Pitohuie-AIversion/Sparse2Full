#!/usr/bin/env python3
"""
重构脚本验证工具
验证重构版本的train_real_data_ar_refactored.py是否符合所有要求和规范
"""

import os
import sys
import json
import yaml
import ast
import inspect
import importlib.util
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    SUCCESS = "SUCCESS"

@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None

class ValidationReport:
    """验证报告"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def add_result(self, result: ValidationResult):
        """添加验证结果"""
        self.results.append(result)
        if result.level == ValidationLevel.ERROR:
            self.failed += 1
        elif result.level == ValidationLevel.WARNING:
            self.warnings += 1
        elif result.level == ValidationLevel.SUCCESS:
            self.passed += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取验证总结"""
        return {
            'total': len(self.results),
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'success_rate': self.passed / max(len(self.results), 1) * 100
        }
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "="*80)
        print("重构脚本验证报告")
        print("="*80)
        
        # 按类别分组
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # 打印每个类别的结果
        for category, results in categories.items():
            print(f"\n{category.upper()}:")
            print("-" * 60)
            
            for result in results:
                level_color = {
                    ValidationLevel.ERROR: "\033[91m",    # 红色
                    ValidationLevel.WARNING: "\033[93m",  # 黄色
                    ValidationLevel.INFO: "\033[94m",     # 蓝色
                    ValidationLevel.SUCCESS: "\033[92m"   # 绿色
                }.get(result.level, "")
                
                reset_color = "\033[0m"
                
                print(f"{level_color}[{result.level.value}]{reset_color} {result.message}")
                
                if result.details:
                    print(f"  详情: {result.details}")
                
                if result.suggestion:
                    print(f"  建议: {result.suggestion}")
        
        # 打印总结
        summary = self.get_summary()
        print(f"\n{'='*80}")
        print(f"验证总结:")
        print(f"  总检查项: {summary['total']}")
        print(f"  通过: {summary['passed']} ({summary['success_rate']:.1f}%)")
        print(f"  失败: {summary['failed']}")
        print(f"  警告: {summary['warnings']}")
        print("="*80)

class ScriptValidator:
    """脚本验证器"""
    
    def __init__(self, script_path: str, config_path: str):
        self.script_path = Path(script_path)
        self.config_path = Path(config_path)
        self.report = ValidationReport()
        self.script_content = None
        self.config_content = None
        self.ast_tree = None
    
    def validate_all(self) -> ValidationReport:
        """执行所有验证"""
        print("开始验证重构脚本...")
        
        # 基础验证
        self._validate_file_exists()
        self._validate_file_readability()
        
        if self.script_content is None or self.config_content is None:
            return self.report
        
        # 代码结构验证
        self._validate_code_structure()
        self._validate_imports()
        self._validate_class_structure()
        self._validate_function_structure()
        
        # 模块化架构验证
        self._validate_modular_architecture()
        self._validate_manager_classes()
        self._validate_separation_of_concerns()
        
        # 配置验证
        self._validate_config_structure()
        self._validate_config_completeness()
        self._validate_config_consistency()
        
        # 代码质量验证
        self._validate_error_handling()
        self._validate_logging()
        self._validate_documentation()
        self._validate_type_hints()
        
        # 性能和安全验证
        self._validate_performance_considerations()
        self._validate_security_practices()
        self._validate_resource_management()
        
        # 兼容性验证
        self._validate_backward_compatibility()
        self._validate_pytorch_compatibility()
        self._validate_python_compatibility()
        
        return self.report
    
    def _validate_file_exists(self):
        """验证文件存在性"""
        # 验证脚本文件
        if not self.script_path.exists():
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="文件存在性",
                message=f"脚本文件不存在: {self.script_path}",
                suggestion="确保脚本文件路径正确"
            ))
            return
        
        # 验证配置文件
        if not self.config_path.exists():
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="文件存在性",
                message=f"配置文件不存在: {self.config_path}",
                suggestion="确保配置文件路径正确"
            ))
            return
        
        self.report.add_result(ValidationResult(
            level=ValidationLevel.SUCCESS,
            category="文件存在性",
            message="所有必需文件都存在"
        ))
    
    def _validate_file_readability(self):
        """验证文件可读性"""
        try:
            # 读取脚本文件
            with open(self.script_path, 'r', encoding='utf-8') as f:
                self.script_content = f.read()
            
            # 读取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_content = f.read()
            
            self.report.add_result(ValidationResult(
                level=ValidationLevel.SUCCESS,
                category="文件可读性",
                message="所有文件都可成功读取"
            ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="文件可读性",
                message=f"文件读取失败: {str(e)}",
                details=str(e),
                suggestion="检查文件权限和编码"
            ))
    
    def _validate_code_structure(self):
        """验证代码结构"""
        try:
            # 解析AST
            self.ast_tree = ast.parse(self.script_content)
            
            # 检查是否有主函数
            has_main = False
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.Compare):
                        if (isinstance(node.test.left, ast.Name) and 
                            node.test.left.id == '__name__'):
                            has_main = True
                            break
            
            if not has_main:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="代码结构",
                    message="缺少标准的if __name__ == '__main__':结构",
                    suggestion="添加主函数入口以提高代码规范性"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="代码结构",
                    message="代码结构良好，包含主函数入口"
                ))
            
            # 检查类定义
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            if len(classes) < 5:  # 期望有多个管理器类
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="代码结构",
                    message=f"类定义数量较少({len(classes)})，可能缺少模块化设计",
                    details=f"找到{len(classes)}个类定义"
                ))
            
            # 检查函数定义
            functions = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.FunctionDef)]
            if len(functions) < 10:  # 期望有足够多的函数
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="代码结构",
                    message=f"函数定义数量较少({len(functions)})，可能功能不够完整",
                    details=f"找到{len(functions)}个函数定义"
                ))
            
        except SyntaxError as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="代码结构",
                message=f"代码语法错误: {str(e)}",
                details=str(e),
                suggestion="修复语法错误"
            ))
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="代码结构",
                message=f"代码结构验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_imports(self):
        """验证导入语句"""
        required_imports = [
            'torch', 'torch.nn', 'torch.distributed', 'numpy', 'matplotlib',
            'omegaconf', 'tqdm', 'logging', 'pathlib', 'typing', 'abc'
        ]
        
        optional_imports = [
            'h5py', 'psutil', 'warnings', 'json', 'datetime', 'random'
        ]
        
        found_imports = set()
        
        try:
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        found_imports.add(node.module)
            
            # 检查必需导入
            missing_required = []
            for imp in required_imports:
                if not any(imp in found for found in found_imports):
                    missing_required.append(imp)
            
            if missing_required:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="导入验证",
                    message=f"缺少必需的导入: {', '.join(missing_required)}",
                    suggestion="添加缺少的导入语句"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="导入验证",
                    message="所有必需的导入都存在"
                ))
            
            # 检查可选导入
            found_optional = []
            for imp in optional_imports:
                if any(imp in found for found in found_imports):
                    found_optional.append(imp)
            
            if found_optional:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.INFO,
                    category="导入验证",
                    message=f"找到可选导入: {', '.join(found_optional)}"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="导入验证",
                message=f"导入验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_class_structure(self):
        """验证类结构"""
        expected_classes = [
            'ConfigManager', 'DeviceManager', 'LogManager', 'DataManager',
            'ModelManager', 'OptimizerManager', 'LossManager', 'CurriculumManager',
            'CheckpointManager', 'RealDataARTrainer', 'ValidationResult',
            'ValidationReport', 'ScriptValidator'
        ]
        
        try:
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            class_names = [cls.name for cls in classes]
            
            missing_classes = []
            for expected in expected_classes:
                if expected not in class_names:
                    missing_classes.append(expected)
            
            if missing_classes:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="类结构",
                    message=f"缺少期望的类: {', '.join(missing_classes)}",
                    details=f"找到类: {', '.join(class_names)}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="类结构",
                    message="所有期望的管理器类都存在"
                ))
            
            # 检查类的方法
            for cls in classes:
                methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
                if len(methods) < 2:  # 期望每个类至少有2个方法
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="类结构",
                        message=f"类'{cls.name}'方法数量较少({len(methods)})",
                        suggestion="考虑添加更多方法来提高类的功能性"
                    ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="类结构",
                message=f"类结构验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_function_structure(self):
        """验证函数结构"""
        try:
            functions = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.FunctionDef)]
            
            # 检查函数命名规范
            for func in functions:
                if not func.name.islower() and '_' not in func.name:
                    if not func.name[0].isupper():  # 排除类方法
                        self.report.add_result(ValidationResult(
                            level=ValidationLevel.WARNING,
                            category="函数结构",
                            message=f"函数'{func.name}'命名不符合snake_case规范",
                            suggestion="使用snake_case命名约定"
                        ))
            
            # 检查函数文档字符串
            funcs_without_docstring = []
            for func in functions:
                if not ast.get_docstring(func):
                    funcs_without_docstring.append(func.name)
            
            if funcs_without_docstring:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="函数结构",
                    message=f"缺少文档字符串的函数: {', '.join(funcs_without_docstring[:5])}",
                    details=f"共{len(funcs_without_docstring)}个函数缺少文档字符串",
                    suggestion="为所有公共函数添加文档字符串"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="函数结构",
                    message="所有函数都有文档字符串"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="函数结构",
                message=f"函数结构验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_modular_architecture(self):
        """验证模块化架构"""
        try:
            # 检查是否有清晰的模块分离
            manager_classes = [
                'ConfigManager', 'DeviceManager', 'LogManager', 'DataManager',
                'ModelManager', 'OptimizerManager', 'LossManager', 'CurriculumManager',
                'CheckpointManager'
            ]
            
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            class_names = [cls.name for cls in classes]
            
            found_managers = [name for name in manager_classes if name in class_names]
            
            if len(found_managers) >= 5:  # 期望至少有5个管理器类
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="模块化架构",
                    message=f"找到{len(found_managers)}个管理器类，模块化架构良好",
                    details=f"管理器类: {', '.join(found_managers)}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="模块化架构",
                    message=f"管理器类数量较少({len(found_managers)})，可能模块化不够充分",
                    suggestion="考虑将更多功能分离到专门的管理器类中"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="模块化架构",
                message=f"模块化架构验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_manager_classes(self):
        """验证管理器类实现"""
        try:
            # 定义期望的管理器类及其关键方法
            expected_managers = {
                'ConfigManager': ['load_config', 'validate_config'],
                'DeviceManager': ['setup_device', 'cleanup_distributed'],
                'LogManager': ['setup_logging', 'log_metrics'],
                'DataManager': ['setup_data', '_create_data_loaders'],
                'ModelManager': ['setup_model', '_create_base_model'],
                'OptimizerManager': ['setup_optimizer', 'step_scheduler'],
                'LossManager': ['setup_loss_functions', 'compute_loss'],
                'CurriculumManager': ['setup_curriculum', 'get_current_T_out'],
                'CheckpointManager': ['setup_checkpointing', 'save_checkpoint']
            }
            
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            
            for manager_name, expected_methods in expected_managers.items():
                manager_class = next((cls for cls in classes if cls.name == manager_name), None)
                
                if manager_class is None:
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="管理器类",
                        message=f"缺少管理器类: {manager_name}",
                        suggestion="实现缺失的管理器类"
                    ))
                    continue
                
                # 检查关键方法
                methods = [node.name for node in manager_class.body if isinstance(node, ast.FunctionDef)]
                missing_methods = [method for method in expected_methods if method not in methods]
                
                if missing_methods:
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="管理器类",
                        message=f"管理器类'{manager_name}'缺少关键方法: {', '.join(missing_methods)}",
                        suggestion="实现缺失的关键方法"
                    ))
                else:
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.SUCCESS,
                        category="管理器类",
                        message=f"管理器类'{manager_name}'实现了所有关键方法"
                    ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="管理器类",
                message=f"管理器类验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_separation_of_concerns(self):
        """验证关注点分离"""
        try:
            # 检查是否有类承担了过多职责
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            
            for cls in classes:
                methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
                
                # 检查方法数量
                if len(methods) > 15:  # 如果一个类有太多方法，可能职责过多
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="关注点分离",
                        message=f"类'{cls.name}'方法数量较多({len(methods)})，可能职责过多",
                        suggestion="考虑将类拆分为多个更小的、职责单一的类"
                    ))
                
                # 检查方法职责范围
                method_names = [method.name for method in methods]
                
                # 如果一个类同时处理配置、数据、模型等多个方面，可能职责不清
                responsibility_indicators = {
                    'config': any('config' in name.lower() for name in method_names),
                    'data': any(name.lower() in ['load_data', 'setup_data', 'create_dataloader'] for name in method_names),
                    'model': any(name.lower() in ['create_model', 'setup_model', 'build_model'] for name in method_names),
                    'training': any(name.lower() in ['train', 'fit', 'step'] for name in method_names),
                    'validation': any(name.lower() in ['validate', 'evaluate', 'test'] for name in method_names)
                }
                
                responsibilities = sum(1 for v in responsibility_indicators.values() if v)
                
                if responsibilities > 2 and cls.name not in ['RealDataARTrainer']:  # 主训练器可以有多个职责
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="关注点分离",
                        message=f"类'{cls.name}'承担了多个职责({responsibilities}个)",
                        details=f"检测到的职责: {[k for k,v in responsibility_indicators.items() if v]}",
                        suggestion="考虑将不同职责分离到专门的类中"
                    ))
            
            self.report.add_result(ValidationResult(
                level=ValidationLevel.SUCCESS,
                category="关注点分离",
                message="整体架构关注点分离良好"
            ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="关注点分离",
                message=f"关注点分离验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_config_structure(self):
        """验证配置文件结构"""
        try:
            config = yaml.safe_load(self.config_content)
            
            if not isinstance(config, dict):
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="配置文件结构",
                    message="配置文件不是有效的YAML字典格式"
                ))
                return
            
            # 检查必需的一级配置项
            required_top_level = [
                'experiment', 'data', 'model', 'training', 'loss',
                'observation', 'validation', 'performance_monitoring',
                'hardware', 'testing', 'paper_package'
            ]
            
            missing_top_level = []
            for key in required_top_level:
                if key not in config:
                    missing_top_level.append(key)
            
            if missing_top_level:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="配置文件结构",
                    message=f"缺少必需的一级配置项: {', '.join(missing_top_level)}",
                    suggestion="在配置文件中添加缺失的配置项"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="配置文件结构",
                    message="配置文件包含所有必需的一级配置项"
                ))
            
            # 检查配置项的深度和复杂性
            def check_depth(obj, max_depth=5, current_depth=0):
                if current_depth > max_depth:
                    return False
                if isinstance(obj, dict):
                    for value in obj.values():
                        if not check_depth(value, max_depth, current_depth + 1):
                            return False
                return True
            
            if not check_depth(config):
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="配置文件结构",
                    message="配置文件嵌套层级过深，可能影响可读性",
                    suggestion="考虑简化配置结构，减少嵌套层级"
                ))
            
        except yaml.YAMLError as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="配置文件结构",
                message=f"配置文件YAML格式错误: {str(e)}",
                details=str(e),
                suggestion="修复YAML语法错误"
            ))
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="配置文件结构",
                message=f"配置文件结构验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_config_completeness(self):
        """验证配置文件完整性"""
        try:
            config = yaml.safe_load(self.config_content)
            
            # 检查关键配置项的完整性
            critical_paths = [
                ('experiment', 'name'),
                ('experiment', 'seed'),
                ('data', 'data_path'),
                ('data', 'T_in'),
                ('data', 'T_out'),
                ('model', 'name'),
                ('training', 'epochs'),
                ('training', 'optimizer', 'lr'),
                ('validation', 'metrics')
            ]
            
            missing_paths = []
            for path in critical_paths:
                current = config
                try:
                    for key in path:
                        current = current[key]
                except (KeyError, TypeError):
                    missing_paths.append('.'.join(path))
            
            if missing_paths:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="配置文件完整性",
                    message=f"缺少关键配置路径: {', '.join(missing_paths)}",
                    suggestion="在配置文件中添加缺失的关键配置"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="配置文件完整性",
                    message="配置文件包含所有关键配置项"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="配置文件完整性",
                message=f"配置文件完整性验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_config_consistency(self):
        """验证配置文件一致性"""
        try:
            config = yaml.safe_load(self.config_content)
            
            # 检查配置值的一致性
            inconsistencies = []
            
            # 检查批次大小一致性
            if 'data' in config and 'dataloader' in config['data']:
                dl_config = config['data']['dataloader']
                if 'batch_size' in dl_config and 'val_batch_size' in dl_config:
                    if dl_config['batch_size'] < dl_config['val_batch_size']:
                        inconsistencies.append("训练批次大小小于验证批次大小")
            
            # 检查学习率范围
            if 'training' in config and 'optimizer' in config['training']:
                lr = config['training']['optimizer'].get('lr', 0)
                if lr <= 0 or lr > 1:
                    inconsistencies.append(f"学习率值({lr})超出合理范围")
            
            # 检查epochs设置
            if 'training' in config:
                epochs = config['training'].get('epochs', 0)
                if epochs < 1:
                    inconsistencies.append("训练轮数必须至少为1")
            
            if inconsistencies:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="配置文件一致性",
                    message=f"发现配置不一致: {', '.join(inconsistencies)}",
                    suggestion="检查并修正配置值的一致性"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="配置文件一致性",
                    message="配置值一致性良好"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="配置文件一致性",
                message=f"配置文件一致性验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_error_handling(self):
        """验证错误处理"""
        try:
            # 检查try-except块的使用
            try_blocks = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.Try)]
            
            if len(try_blocks) < 5:  # 期望有足够的错误处理
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="错误处理",
                    message=f"try-except块数量较少({len(try_blocks)})，可能错误处理不够充分",
                    suggestion="在关键操作周围添加适当的错误处理"
                ))
            
            # 检查是否有通用的异常捕获
            for try_block in try_blocks:
                for handler in try_block.handlers:
                    if handler.type is None:  # 捕获所有异常的except:
                        self.report.add_result(ValidationResult(
                            level=ValidationLevel.WARNING,
                            category="错误处理",
                            message="发现捕获所有异常的except块，可能隐藏重要错误",
                            suggestion="使用具体的异常类型而不是裸except:"
                        ))
            
            # 检查日志记录
            log_calls = []
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if 'log' in node.func.attr.lower():
                            log_calls.append(node.func.attr)
            
            if len(log_calls) < 10:  # 期望有足够的日志记录
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="错误处理",
                    message=f"日志调用数量较少({len(log_calls)})，可能日志记录不够充分",
                    suggestion="在错误处理中添加适当的日志记录"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="错误处理",
                    message="错误处理和日志记录充分"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="错误处理",
                message=f"错误处理验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_logging(self):
        """验证日志记录"""
        try:
            # 检查日志配置
            if 'setup_logger' not in self.script_content:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="日志记录",
                    message="未找到setup_logger函数，可能日志配置不够完善",
                    suggestion="实现完整的日志配置函数"
                ))
            
            # 检查日志级别使用
            log_levels = ['debug', 'info', 'warning', 'error', 'critical']
            found_levels = []
            
            for level in log_levels:
                if level in self.script_content.lower():
                    found_levels.append(level)
            
            if len(found_levels) < 3:  # 期望使用多种日志级别
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="日志记录",
                    message=f"使用的日志级别较少({len(found_levels)}): {', '.join(found_levels)}",
                    suggestion="使用适当的日志级别来区分不同重要性的信息"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="日志记录",
                    message="使用了多种日志级别"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="日志记录",
                message=f"日志记录验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_documentation(self):
        """验证文档"""
        try:
            # 检查模块文档字符串
            module_docstring = ast.get_docstring(self.ast_tree)
            if not module_docstring:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="文档",
                    message="缺少模块级文档字符串",
                    suggestion="在文件开头添加模块文档字符串"
                ))
            
            # 检查类文档字符串
            classes = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.ClassDef)]
            classes_without_docstring = []
            
            for cls in classes:
                if not ast.get_docstring(cls):
                    classes_without_docstring.append(cls.name)
            
            if classes_without_docstring:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="文档",
                    message=f"缺少文档字符串的类: {', '.join(classes_without_docstring[:5])}",
                    details=f"共{len(classes_without_docstring)}个类缺少文档字符串",
                    suggestion="为所有类添加文档字符串"
                ))
            
            # 检查内联注释
            lines = self.script_content.split('\n')
            commented_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            if commented_lines < len(lines) * 0.05:  # 期望至少有5%的注释行
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="文档",
                    message=f"注释行比例较低({commented_lines/len(lines)*100:.1f}%)",
                    suggestion="添加更多内联注释来解释复杂的逻辑"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="文档",
                    message="代码注释充分"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="文档",
                message=f"文档验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_type_hints(self):
        """验证类型提示"""
        try:
            # 检查函数参数和返回值的类型提示
            functions = [node for node in ast.walk(self.ast_tree) if isinstance(node, ast.FunctionDef)]
            
            funcs_without_type_hints = []
            for func in functions:
                # 检查参数类型提示
                has_arg_annotations = any(arg.annotation for arg in func.args.args)
                has_return_annotation = func.returns is not None
                
                if not has_arg_annotations and not has_return_annotation:
                    funcs_without_type_hints.append(func.name)
            
            if funcs_without_type_hints:
                percentage = len(funcs_without_type_hints) / len(functions) * 100
                if percentage > 50:  # 如果超过50%的函数没有类型提示
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="类型提示",
                        message=f"较多函数缺少类型提示({percentage:.1f}%)",
                        details=f"缺少类型提示的函数: {', '.join(funcs_without_type_hints[:5])}",
                        suggestion="为函数参数和返回值添加类型提示"
                    ))
                else:
                    self.report.add_result(ValidationResult(
                        level=ValidationLevel.INFO,
                        category="类型提示",
                        message=f"部分函数缺少类型提示({percentage:.1f}%)"
                    ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="类型提示",
                    message="所有函数都有类型提示"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="类型提示",
                message=f"类型提示验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_performance_considerations(self):
        """验证性能考虑"""
        try:
            # 检查是否有性能相关的代码
            performance_patterns = [
                'torch.no_grad', 'autocast', 'GradScaler', 'pin_memory',
                'num_workers', 'persistent_workers', 'prefetch_factor',
                'torch.cuda', 'memory_efficient', 'gradient_checkpointing'
            ]
            
            found_patterns = []
            for pattern in performance_patterns:
                if pattern in self.script_content:
                    found_patterns.append(pattern)
            
            if len(found_patterns) >= 5:  # 期望找到多个性能优化模式
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="性能考虑",
                    message=f"找到{len(found_patterns)}个性能优化模式",
                    details=f"性能模式: {', '.join(found_patterns[:5])}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="性能考虑",
                    message=f"性能优化模式较少({len(found_patterns)})，可能性能考虑不够充分",
                    suggestion="添加更多性能优化，如内存管理、并行处理等"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="性能考虑",
                message=f"性能考虑验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_security_practices(self):
        """验证安全实践"""
        try:
            # 检查潜在的安全问题
            security_issues = []
            
            # 检查是否有硬编码的敏感信息
            if 'password' in self.script_content.lower() or 'secret' in self.script_content.lower():
                security_issues.append("发现可能的硬编码敏感信息")
            
            # 检查文件路径处理
            if 'open(' in self.script_content and 'input(' not in self.script_content:
                # 检查是否有适当的文件存在性检查
                if 'os.path.exists' not in self.script_content:
                    security_issues.append("文件操作可能缺少存在性检查")
            
            # 检查eval/exec的使用
            if 'eval(' in self.script_content or 'exec(' in self.script_content:
                security_issues.append("使用了eval或exec，可能存在安全风险")
            
            if security_issues:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="安全实践",
                    message=f"发现安全实践问题: {', '.join(security_issues)}",
                    suggestion="修复安全实践问题"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="安全实践",
                    message="未发现明显的安全实践问题"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="安全实践",
                message=f"安全实践验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_resource_management(self):
        """验证资源管理"""
        try:
            # 检查资源管理相关代码
            resource_patterns = [
                'with open', 'try:', 'finally:', 'close', 'cleanup',
                'destroy', 'release', 'free', 'del '
            ]
            
            found_patterns = []
            for pattern in resource_patterns:
                if pattern in self.script_content:
                    found_patterns.append(pattern)
            
            if len(found_patterns) >= 3:  # 期望有足够的资源管理
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="资源管理",
                    message=f"找到{len(found_patterns)}个资源管理模式",
                    details=f"资源管理模式: {', '.join(found_patterns)}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="资源管理",
                    message=f"资源管理模式较少({len(found_patterns)})，可能资源管理不够充分",
                    suggestion="添加适当的资源清理和释放代码"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="资源管理",
                message=f"资源管理验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_backward_compatibility(self):
        """验证向后兼容性"""
        try:
            # 检查是否有向后兼容性考虑
            compatibility_indicators = [
                'compatibility', 'backward', 'legacy', 'deprecated',
                'version', 'compat'
            ]
            
            found_indicators = []
            for indicator in compatibility_indicators:
                if indicator in self.script_content.lower():
                    found_indicators.append(indicator)
            
            if found_indicators:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.INFO,
                    category="向后兼容性",
                    message=f"发现向后兼容性考虑: {', '.join(found_indicators)}",
                    details="代码考虑了向后兼容性"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.INFO,
                    category="向后兼容性",
                    message="未明确发现向后兼容性考虑",
                    suggestion="考虑添加版本兼容性检查"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="向后兼容性",
                message=f"向后兼容性验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_pytorch_compatibility(self):
        """验证PyTorch兼容性"""
        try:
            # 检查PyTorch版本兼容性
            pytorch_patterns = [
                'torch.__version__', 'torch_version', 'pytorch_version',
                'torch.cuda.is_available', 'torch.distributed'
            ]
            
            found_patterns = []
            for pattern in pytorch_patterns:
                if pattern in self.script_content:
                    found_patterns.append(pattern)
            
            if found_patterns:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="PyTorch兼容性",
                    message=f"找到{len(found_patterns)}个PyTorch兼容性模式",
                    details=f"PyTorch模式: {', '.join(found_patterns)}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="PyTorch兼容性",
                    message="未明确发现PyTorch版本兼容性考虑",
                    suggestion="考虑添加PyTorch版本检查和兼容性处理"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="PyTorch兼容性",
                message=f"PyTorch兼容性验证失败: {str(e)}",
                details=str(e)
            ))
    
    def _validate_python_compatibility(self):
        """验证Python兼容性"""
        try:
            # 检查Python版本兼容性
            python_patterns = [
                'sys.version', 'python_version', 'py_version',
                'from __future__', 'typing', 'dataclass'
            ]
            
            found_patterns = []
            for pattern in python_patterns:
                if pattern in self.script_content:
                    found_patterns.append(pattern)
            
            if found_patterns:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.SUCCESS,
                    category="Python兼容性",
                    message=f"找到{len(found_patterns)}个Python兼容性模式",
                    details=f"Python模式: {', '.join(found_patterns)}"
                ))
            else:
                self.report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Python兼容性",
                    message="未明确发现Python版本兼容性考虑",
                    suggestion="考虑添加Python版本检查和兼容性处理"
                ))
            
        except Exception as e:
            self.report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                category="Python兼容性",
                message=f"Python兼容性验证失败: {str(e)}",
                details=str(e)
            ))

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证重构的训练脚本")
    parser.add_argument("--script", type=str, 
                       default="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/tools/training/train_real_data_ar_refactored.py",
                       help="重构脚本路径")
    parser.add_argument("--config", type=str,
                       default="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/configs/ar_training_refactored_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--output", type=str, help="输出报告文件路径")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="输出格式")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = ScriptValidator(args.script, args.config)
    
    # 执行验证
    report = validator.validate_all()
    
    # 打印报告
    if args.format == "text":
        report.print_report()
    elif args.format == "json":
        summary = report.get_summary()
        results = [
            {
                'level': result.level.value,
                'category': result.category,
                'message': result.message,
                'details': result.details,
                'suggestion': result.suggestion
            }
            for result in report.results
        ]
        
        json_report = {
            'summary': summary,
            'results': results
        }
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            print(f"验证报告已保存到: {args.output}")
        else:
            print(json.dumps(json_report, indent=2, ensure_ascii=False))
    
    # 返回适当的退出码
    sys.exit(0 if report.failed == 0 else 1)

if __name__ == "__main__":
    main()