#!/usr/bin/env python3
"""
PDEBench模型接口统一性验证工具

严格遵循黄金法则：
3. 统一接口：所有模型forward(x[B,C_in,H,W])→y[B,C_out,H,W]

验证目标：
- 检查所有模型的forward方法签名一致性
- 验证输入输出张量形状
- 确保模型初始化参数统一
- 生成接口验证报告
"""

import os
import sys
import json
import logging
import inspect
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Type
from collections import defaultdict

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 项目导入
try:
    from models import *
    from models.base import BaseModel
except ImportError as e:
    logging.warning(f"Import error: {e}. Some models may not be available.")


class ModelInterfaceValidator:
    """模型接口统一性验证器
    
    验证所有模型是否遵循统一接口规范：
    1. 初始化签名：__init__(in_channels, out_channels, img_size, **kwargs)
    2. 前向传播：forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    3. 输入输出形状一致性
    4. 参数类型检查
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 验证配置
        self.validator_config = config.get('model_interface_validator', {})
        self.test_batch_size = self.validator_config.get('test_batch_size', 2)
        self.test_img_size = self.validator_config.get('test_img_size', 64)
        self.test_in_channels = self.validator_config.get('test_in_channels', 4)
        self.test_out_channels = self.validator_config.get('test_out_channels', 1)
        
        # 结果存储
        self.validation_results = {}
        self.failed_models = []
        self.passed_models = []
        
        logging.info(f"Model interface validator config: batch_size={self.test_batch_size}, "
                    f"img_size={self.test_img_size}, in_ch={self.test_in_channels}, out_ch={self.test_out_channels}")
    
    def discover_models(self) -> Dict[str, Type[nn.Module]]:
        """发现所有可用的模型类"""
        models = {}
        
        # 从models模块导入所有模型
        models_module = importlib.import_module('models')
        
        for name in dir(models_module):
            obj = getattr(models_module, name)
            
            # 检查是否是模型类
            if (inspect.isclass(obj) and 
                issubclass(obj, nn.Module) and 
                obj != nn.Module and 
                obj != BaseModel and
                not name.startswith('_')):
                
                models[name] = obj
                logging.info(f"Discovered model: {name}")
        
        return models
    
    def validate_init_signature(self, model_class: Type[nn.Module]) -> Dict[str, Any]:
        """验证模型初始化签名"""
        result = {
            'passed': False,
            'signature': None,
            'required_params': [],
            'optional_params': [],
            'errors': []
        }
        
        try:
            # 获取__init__方法签名
            init_signature = inspect.signature(model_class.__init__)
            result['signature'] = str(init_signature)
            
            # 检查必需参数
            required_params = ['in_channels', 'out_channels', 'img_size']
            found_required = []
            
            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue
                
                if param_name in required_params:
                    found_required.append(param_name)
                    result['required_params'].append({
                        'name': param_name,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    })
                else:
                    result['optional_params'].append({
                        'name': param_name,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    })
            
            # 检查是否有**kwargs
            has_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD 
                for param in init_signature.parameters.values()
            )
            
            # 验证必需参数
            missing_required = set(required_params) - set(found_required)
            if missing_required:
                result['errors'].append(f"Missing required parameters: {missing_required}")
            
            if not has_kwargs:
                result['errors'].append("Missing **kwargs parameter for extensibility")
            
            # 如果没有错误，则通过
            result['passed'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Failed to inspect __init__ signature: {e}")
        
        return result
    
    def validate_forward_signature(self, model_class: Type[nn.Module]) -> Dict[str, Any]:
        """验证forward方法签名"""
        result = {
            'passed': False,
            'signature': None,
            'errors': []
        }
        
        try:
            # 获取forward方法签名
            forward_signature = inspect.signature(model_class.forward)
            result['signature'] = str(forward_signature)
            
            # 检查参数
            params = list(forward_signature.parameters.keys())
            
            if len(params) < 2:  # self + x
                result['errors'].append("forward method should have at least 'self' and 'x' parameters")
            elif params[1] != 'x':
                result['errors'].append(f"First parameter should be 'x', got '{params[1]}'")
            
            # 如果没有错误，则通过
            result['passed'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Failed to inspect forward signature: {e}")
        
        return result
    
    def validate_model_instantiation(self, model_class: Type[nn.Module]) -> Dict[str, Any]:
        """验证模型实例化"""
        result = {
            'passed': False,
            'model_instance': None,
            'errors': []
        }
        
        try:
            # 尝试创建模型实例
            model = model_class(
                in_channels=self.test_in_channels,
                out_channels=self.test_out_channels,
                img_size=self.test_img_size
            )
            
            result['model_instance'] = model
            result['passed'] = True
            
        except Exception as e:
            result['errors'].append(f"Failed to instantiate model: {e}")
        
        return result
    
    def validate_forward_pass(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """验证前向传播"""
        result = {
            'passed': False,
            'input_shape': None,
            'output_shape': None,
            'shape_correct': False,
            'errors': []
        }
        
        try:
            model.eval()
            
            # 创建测试输入
            input_tensor = torch.randn(
                self.test_batch_size, 
                self.test_in_channels, 
                self.test_img_size, 
                self.test_img_size
            )
            
            result['input_shape'] = list(input_tensor.shape)
            
            # 前向传播
            with torch.no_grad():
                output = model(input_tensor)
            
            result['output_shape'] = list(output.shape)
            
            # 验证输出形状
            expected_shape = [
                self.test_batch_size,
                self.test_out_channels,
                self.test_img_size,
                self.test_img_size
            ]
            
            if list(output.shape) == expected_shape:
                result['shape_correct'] = True
                result['passed'] = True
            else:
                result['errors'].append(
                    f"Output shape mismatch. Expected {expected_shape}, got {list(output.shape)}"
                )
            
        except Exception as e:
            result['errors'].append(f"Forward pass failed: {e}")
        
        return result
    
    def validate_single_model(self, model_name: str, model_class: Type[nn.Module]) -> Dict[str, Any]:
        """验证单个模型"""
        logging.info(f"Validating model: {model_name}")
        
        result = {
            'model_name': model_name,
            'model_class': str(model_class),
            'overall_passed': False,
            'init_signature': {},
            'forward_signature': {},
            'instantiation': {},
            'forward_pass': {},
            'errors': []
        }
        
        try:
            # 1. 验证初始化签名
            result['init_signature'] = self.validate_init_signature(model_class)
            
            # 2. 验证forward签名
            result['forward_signature'] = self.validate_forward_signature(model_class)
            
            # 3. 验证模型实例化
            result['instantiation'] = self.validate_model_instantiation(model_class)
            
            # 4. 验证前向传播（如果实例化成功）
            if result['instantiation']['passed']:
                model_instance = result['instantiation']['model_instance']
                result['forward_pass'] = self.validate_forward_pass(model_instance, model_name)
            else:
                result['forward_pass'] = {
                    'passed': False,
                    'errors': ['Skipped due to instantiation failure']
                }
            
            # 总体通过状态
            result['overall_passed'] = all([
                result['init_signature']['passed'],
                result['forward_signature']['passed'],
                result['instantiation']['passed'],
                result['forward_pass']['passed']
            ])
            
            if result['overall_passed']:
                self.passed_models.append(model_name)
                logging.info(f"✅ {model_name} passed all validations")
            else:
                self.failed_models.append(model_name)
                logging.warning(f"❌ {model_name} failed validation")
            
        except Exception as e:
            result['errors'].append(f"Validation error: {e}")
            self.failed_models.append(model_name)
            logging.error(f"❌ {model_name} validation error: {e}")
        
        return result
    
    def run_validation(self, output_dir: Path) -> Dict[str, Any]:
        """运行完整的模型接口验证"""
        logging.info("Starting model interface validation...")
        
        # 发现所有模型
        models = self.discover_models()
        
        if not models:
            logging.error("No models found!")
            return {'error': 'No models found'}
        
        logging.info(f"Found {len(models)} models to validate")
        
        # 验证每个模型
        for model_name, model_class in models.items():
            try:
                result = self.validate_single_model(model_name, model_class)
                self.validation_results[model_name] = result
            except Exception as e:
                logging.error(f"Failed to validate {model_name}: {e}")
                self.validation_results[model_name] = {
                    'model_name': model_name,
                    'overall_passed': False,
                    'errors': [f"Validation failed: {e}"]
                }
                self.failed_models.append(model_name)
        
        # 生成汇总结果
        summary = self.generate_summary()
        
        # 保存结果
        self.save_results(output_dir, summary)
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成验证汇总"""
        total_models = len(self.validation_results)
        passed_count = len(self.passed_models)
        failed_count = len(self.failed_models)
        
        # 统计各类错误
        error_stats = defaultdict(int)
        for result in self.validation_results.values():
            if not result.get('overall_passed', False):
                # 收集所有错误
                for check_name in ['init_signature', 'forward_signature', 'instantiation', 'forward_pass']:
                    check_result = result.get(check_name, {})
                    for error in check_result.get('errors', []):
                        error_stats[error] += 1
        
        summary = {
            'total_models': total_models,
            'passed_models': passed_count,
            'failed_models': failed_count,
            'pass_rate': passed_count / total_models if total_models > 0 else 0.0,
            'passed_model_list': self.passed_models,
            'failed_model_list': self.failed_models,
            'error_statistics': dict(error_stats),
            'overall_passed': failed_count == 0,
            'validation_config': {
                'test_batch_size': self.test_batch_size,
                'test_img_size': self.test_img_size,
                'test_in_channels': self.test_in_channels,
                'test_out_channels': self.test_out_channels
            }
        }
        
        return summary
    
    def save_results(self, output_dir: Path, summary: Dict[str, Any]):
        """保存验证结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        detailed_results_path = output_dir / 'model_interface_detailed.json'
        with open(detailed_results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # 保存汇总结果
        summary_path = output_dir / 'model_interface_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 生成报告
        self.generate_report(output_dir, summary)
        
        logging.info(f"Model interface validation results saved to {output_dir}")
    
    def generate_report(self, output_dir: Path, summary: Dict[str, Any]):
        """生成模型接口验证报告"""
        report_path = output_dir / 'model_interface_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# PDEBench模型接口统一性验证报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- 验证模型总数: {summary['total_models']}\n")
            f.write(f"- 通过验证模型数: {summary['passed_models']}\n")
            f.write(f"- 失败验证模型数: {summary['failed_models']}\n")
            f.write(f"- 通过率: {summary['pass_rate']:.1%}\n")
            f.write(f"- 测试配置: 批次大小={summary['validation_config']['test_batch_size']}, "
                   f"图像尺寸={summary['validation_config']['test_img_size']}, "
                   f"输入通道={summary['validation_config']['test_in_channels']}, "
                   f"输出通道={summary['validation_config']['test_out_channels']}\n\n")
            
            # 总体结果
            f.write("## 总体结果\n\n")
            f.write(f"- **接口统一性验证**: {'✅ 通过' if summary['overall_passed'] else '❌ 失败'}\n\n")
            
            # 通过的模型
            if summary['passed_model_list']:
                f.write("## 通过验证的模型\n\n")
                for model_name in summary['passed_model_list']:
                    f.write(f"- ✅ {model_name}\n")
                f.write("\n")
            
            # 失败的模型
            if summary['failed_model_list']:
                f.write("## 失败验证的模型\n\n")
                for model_name in summary['failed_model_list']:
                    f.write(f"- ❌ {model_name}\n")
                f.write("\n")
            
            # 错误统计
            if summary['error_statistics']:
                f.write("## 错误统计\n\n")
                f.write("| 错误类型 | 出现次数 |\n")
                f.write("|----------|----------|\n")
                for error, count in summary['error_statistics'].items():
                    f.write(f"| {error} | {count} |\n")
                f.write("\n")
            
            # 详细验证结果
            f.write("## 详细验证结果\n\n")
            for model_name, result in self.validation_results.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **总体状态**: {'✅ 通过' if result.get('overall_passed', False) else '❌ 失败'}\n")
                
                # 各项检查结果
                checks = ['init_signature', 'forward_signature', 'instantiation', 'forward_pass']
                check_names = ['初始化签名', '前向传播签名', '模型实例化', '前向传播测试']
                
                for check, check_name in zip(checks, check_names):
                    check_result = result.get(check, {})
                    status = '✅ 通过' if check_result.get('passed', False) else '❌ 失败'
                    f.write(f"- **{check_name}**: {status}\n")
                    
                    if check_result.get('errors'):
                        f.write(f"  - 错误: {'; '.join(check_result['errors'])}\n")
                
                f.write("\n")
            
            # 结论和建议
            f.write("## 结论和建议\n\n")
            if summary['overall_passed']:
                f.write("✅ **所有模型都通过了接口统一性验证**\n\n")
                f.write("- 所有模型都遵循统一的初始化签名\n")
                f.write("- 所有模型都遵循统一的前向传播接口\n")
                f.write("- 系统满足黄金法则的统一接口要求\n")
            else:
                f.write("❌ **部分模型未通过接口统一性验证**\n\n")
                f.write("需要修复的问题：\n")
                
                for error, count in summary['error_statistics'].items():
                    f.write(f"- {error} (影响 {count} 个模型)\n")
                
                f.write("\n建议修复步骤：\n")
                f.write("1. 检查模型初始化参数是否包含 in_channels, out_channels, img_size\n")
                f.write("2. 确保模型初始化包含 **kwargs 参数\n")
                f.write("3. 验证 forward 方法签名正确\n")
                f.write("4. 测试模型实例化和前向传播\n")
        
        logging.info(f"Model interface report saved to {report_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="model_interface_validator")
def main(cfg: DictConfig) -> None:
    """主验证函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path(cfg.get('output_dir', 'model_interface_validation_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建验证器
    validator = ModelInterfaceValidator(cfg)
    
    # 运行验证
    logger.info("Starting model interface validation...")
    results = validator.run_validation(output_dir)
    
    # 打印结果
    logger.info("Model interface validation completed!")
    logger.info(f"Overall pass: {'YES' if results['overall_passed'] else 'NO'}")
    logger.info(f"Total models: {results['total_models']}")
    logger.info(f"Passed models: {results['passed_models']}")
    logger.info(f"Failed models: {results['failed_models']}")
    logger.info(f"Pass rate: {results['pass_rate']:.1%}")
    
    if results['failed_model_list']:
        logger.warning(f"Failed models: {results['failed_model_list']}")
    
    logger.info(f"Detailed results saved to: {output_dir}")
    
    # 返回退出码
    return 0 if results['overall_passed'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)