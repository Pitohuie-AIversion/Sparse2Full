#!/usr/bin/env python3
"""
模型配置验证脚本
验证所有模型配置文件的完整性和参数一致性

作者: PDEBench稀疏观测重建系统
日期: 2025-01-13
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConfigValidator:
    """模型配置验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.project_root = Path(__file__).parent
        self.model_configs_dir = self.project_root / "configs" / "model"
        self.base_config_path = self.project_root / "configs" / "train.yaml"
        
        # 定义所有可用模型
        self.available_models = [
            "unet",
            "unet_plus_plus", 
            "fno2d",
            "ufno_unet",
            "segformer",
            "unetformer",
            "segformer_unetformer",
            "mlp",
            "mlp_mixer",
            "liif",
            "swin_unet",
            "hybrid"
        ]
        
        # 验证结果
        self.validation_results = {}
        
    def load_base_config(self) -> Dict[str, Any]:
        """加载基础配置文件"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载基础配置: {self.base_config_path}")
            return config
        except Exception as e:
            logger.error(f"加载基础配置失败: {e}")
            raise
    
    def load_model_config(self, model_name: str) -> Tuple[Dict[str, Any], bool]:
        """加载模型配置文件
        
        Returns:
            Tuple[config, exists]: 配置字典和文件是否存在的标志
        """
        model_config_path = self.model_configs_dir / f"{model_name}.yaml"
        
        if not model_config_path.exists():
            logger.warning(f"模型配置文件不存在: {model_config_path}")
            return {}, False
            
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
            return model_config, True
        except Exception as e:
            logger.error(f"加载模型配置失败 {model_name}: {e}")
            return {}, False
    
    def validate_model_config_structure(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证模型配置结构"""
        validation_result = {
            'model_name': model_name,
            'structure_valid': True,
            'missing_fields': [],
            'warnings': [],
            'errors': []
        }
        
        # 必需字段检查
        required_fields = ['name', 'params']
        
        for field in required_fields:
            if field not in config:
                validation_result['missing_fields'].append(field)
                validation_result['structure_valid'] = False
                validation_result['errors'].append(f"缺少必需字段: {field}")
        
        # 检查模型名称一致性
        if 'name' in config and config['name'] != model_name:
            validation_result['warnings'].append(
                f"配置中的模型名称 '{config['name']}' 与文件名 '{model_name}' 不一致"
            )
        
        # 检查参数结构
        if 'params' in config:
            params = config['params']
            
            # 检查基本参数
            basic_params = ['in_channels', 'out_channels']
            for param in basic_params:
                if param not in params:
                    validation_result['warnings'].append(f"建议添加参数: {param}")
            
            # 检查图像尺寸参数
            if 'img_size' not in params and 'image_size' not in params:
                validation_result['warnings'].append("建议添加图像尺寸参数: img_size 或 image_size")
        
        return validation_result
    
    def validate_parameter_consistency(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """验证模型间参数一致性"""
        consistency_result = {
            'consistent': True,
            'inconsistencies': [],
            'recommendations': []
        }
        
        # 收集所有模型的基本参数
        common_params = {}
        
        for model_name, config in model_configs.items():
            if 'params' in config:
                params = config['params']
                
                # 检查输入输出通道数
                if 'in_channels' in params:
                    if 'in_channels' not in common_params:
                        common_params['in_channels'] = {}
                    common_params['in_channels'][model_name] = params['in_channels']
                
                if 'out_channels' in params:
                    if 'out_channels' not in common_params:
                        common_params['out_channels'] = {}
                    common_params['out_channels'][model_name] = params['out_channels']
                
                # 检查图像尺寸
                img_size = params.get('img_size') or params.get('image_size')
                if img_size:
                    if 'img_size' not in common_params:
                        common_params['img_size'] = {}
                    common_params['img_size'][model_name] = img_size
        
        # 检查参数一致性
        for param_name, param_values in common_params.items():
            unique_values = set(param_values.values())
            
            if len(unique_values) > 1:
                consistency_result['consistent'] = False
                inconsistency = {
                    'parameter': param_name,
                    'values': param_values,
                    'unique_values': list(unique_values)
                }
                consistency_result['inconsistencies'].append(inconsistency)
                
                # 生成建议
                most_common_value = max(unique_values, key=lambda x: list(param_values.values()).count(x))
                consistency_result['recommendations'].append(
                    f"建议将所有模型的 {param_name} 统一为: {most_common_value}"
                )
        
        return consistency_result
    
    def validate_training_compatibility(self, base_config: Dict[str, Any], model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """验证与训练配置的兼容性"""
        compatibility_result = {
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # 检查数据配置兼容性
        if 'data' in base_config:
            data_config = base_config['data']
            expected_channels = data_config.get('channels', 1)
            expected_img_size = data_config.get('img_size', 128)
            
            for model_name, model_config in model_configs.items():
                if 'params' in model_config:
                    params = model_config['params']
                    
                    # 检查输入通道数
                    model_in_channels = params.get('in_channels')
                    if model_in_channels and model_in_channels != expected_channels:
                        compatibility_result['compatible'] = False
                        compatibility_result['issues'].append(
                            f"模型 {model_name} 的输入通道数 ({model_in_channels}) "
                            f"与数据配置不匹配 ({expected_channels})"
                        )
                    
                    # 检查图像尺寸
                    model_img_size = params.get('img_size') or params.get('image_size')
                    if model_img_size and model_img_size != expected_img_size:
                        compatibility_result['warnings'].append(
                            f"模型 {model_name} 的图像尺寸 ({model_img_size}) "
                            f"与数据配置不同 ({expected_img_size})"
                        )
        
        return compatibility_result
    
    def check_model_implementation(self, model_name: str) -> Dict[str, Any]:
        """检查模型实现是否存在"""
        implementation_result = {
            'model_name': model_name,
            'implementation_exists': False,
            'model_file': None,
            'class_found': False,
            'import_path': None
        }
        
        # 检查模型文件
        models_dir = self.project_root / "models"
        possible_files = [
            models_dir / f"{model_name}.py",
            models_dir / f"{model_name.replace('_', '')}.py",
            models_dir / "architectures" / f"{model_name}.py"
        ]
        
        for model_file in possible_files:
            if model_file.exists():
                implementation_result['implementation_exists'] = True
                implementation_result['model_file'] = str(model_file)
                break
        
        # 检查 __init__.py 中的导入
        init_file = models_dir / "__init__.py"
        if init_file.exists():
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    init_content = f.read()
                
                # 查找模型类或函数
                model_variations = [
                    model_name,
                    model_name.upper(),
                    ''.join(word.capitalize() for word in model_name.split('_')),
                    f"create_{model_name}",
                    f"get_{model_name}"
                ]
                
                for variation in model_variations:
                    if variation in init_content:
                        implementation_result['class_found'] = True
                        implementation_result['import_path'] = f"models.{variation}"
                        break
                        
            except Exception as e:
                logger.warning(f"检查模型导入失败 {model_name}: {e}")
        
        return implementation_result
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info("开始验证模型配置...")
        
        # 加载基础配置
        base_config = self.load_base_config()
        
        # 验证结果
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.available_models),
            'models': {},
            'consistency_check': {},
            'compatibility_check': {},
            'implementation_check': {},
            'summary': {
                'config_files_found': 0,
                'valid_configs': 0,
                'missing_configs': 0,
                'implementation_found': 0,
                'overall_status': 'unknown'
            }
        }
        
        # 验证每个模型配置
        model_configs = {}
        
        for model_name in self.available_models:
            logger.info(f"验证模型: {model_name}")
            
            # 加载配置
            config, exists = self.load_model_config(model_name)
            
            if exists:
                validation_results['summary']['config_files_found'] += 1
                model_configs[model_name] = config
                
                # 验证配置结构
                structure_result = self.validate_model_config_structure(model_name, config)
                validation_results['models'][model_name] = structure_result
                
                if structure_result['structure_valid']:
                    validation_results['summary']['valid_configs'] += 1
            else:
                validation_results['summary']['missing_configs'] += 1
                validation_results['models'][model_name] = {
                    'model_name': model_name,
                    'structure_valid': False,
                    'missing_fields': [],
                    'warnings': [],
                    'errors': ['配置文件不存在']
                }
            
            # 检查模型实现
            impl_result = self.check_model_implementation(model_name)
            validation_results['implementation_check'][model_name] = impl_result
            
            if impl_result['implementation_exists']:
                validation_results['summary']['implementation_found'] += 1
        
        # 参数一致性检查
        if model_configs:
            consistency_result = self.validate_parameter_consistency(model_configs)
            validation_results['consistency_check'] = consistency_result
        
        # 训练兼容性检查
        if model_configs:
            compatibility_result = self.validate_training_compatibility(base_config, model_configs)
            validation_results['compatibility_check'] = compatibility_result
        
        # 计算总体状态
        total_models = len(self.available_models)
        valid_configs = validation_results['summary']['valid_configs']
        impl_found = validation_results['summary']['implementation_found']
        
        if valid_configs == total_models and impl_found == total_models:
            validation_results['summary']['overall_status'] = 'excellent'
        elif valid_configs >= total_models * 0.8 and impl_found >= total_models * 0.8:
            validation_results['summary']['overall_status'] = 'good'
        elif valid_configs >= total_models * 0.5:
            validation_results['summary']['overall_status'] = 'fair'
        else:
            validation_results['summary']['overall_status'] = 'poor'
        
        self.validation_results = validation_results
        return validation_results
    
    def save_validation_report(self, results: Dict[str, Any]):
        """保存验证报告"""
        # 保存JSON格式
        json_file = self.project_root / "runs" / "batch_training_results" / "model_config_validation.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"保存验证结果: {json_file}")
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
        
        # 生成Markdown报告
        md_file = json_file.with_suffix('.md')
        self.generate_markdown_report(results, md_file)
    
    def generate_markdown_report(self, results: Dict[str, Any], output_file: Path):
        """生成Markdown格式的验证报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# 模型配置验证报告\n\n")
                f.write(f"**生成时间**: {results['timestamp']}\n")
                f.write(f"**验证模型数量**: {len(self.available_models)}\n\n")
                
                # 总体状态
                summary = results['summary']
                status_emoji = {
                    'excellent': '🟢',
                    'good': '🟡', 
                    'fair': '🟠',
                    'poor': '🔴'
                }
                
                f.write("## 总体状态\n\n")
                f.write(f"{status_emoji.get(summary['overall_status'], '⚪')} **{summary['overall_status'].upper()}**\n\n")
                
                # 统计信息
                f.write("## 统计信息\n\n")
                total_models = len(self.available_models)
                f.write(f"- 📁 **配置文件**: {summary['config_files_found']}/{total_models} 找到\n")
                f.write(f"- ✅ **有效配置**: {summary['valid_configs']}/{total_models}\n")
                f.write(f"- ❌ **缺失配置**: {summary['missing_configs']}/{total_models}\n")
                f.write(f"- 🔧 **实现文件**: {summary['implementation_found']}/{total_models} 找到\n\n")
                
                # 模型详情
                f.write("## 模型配置详情\n\n")
                f.write("| 模型名称 | 配置状态 | 实现状态 | 问题数量 | 警告数量 |\n")
                f.write("|---------|---------|---------|---------|----------|\n")
                
                for model_name in self.available_models:
                    model_result = results['models'].get(model_name, {})
                    impl_result = results['implementation_check'].get(model_name, {})
                    
                    config_status = "✅" if model_result.get('structure_valid', False) else "❌"
                    impl_status = "✅" if impl_result.get('implementation_exists', False) else "❌"
                    error_count = len(model_result.get('errors', []))
                    warning_count = len(model_result.get('warnings', []))
                    
                    f.write(f"| {model_name} | {config_status} | {impl_status} | {error_count} | {warning_count} |\n")
                
                f.write("\n")
                
                # 一致性检查
                if 'consistency_check' in results:
                    consistency = results['consistency_check']
                    f.write("## 参数一致性检查\n\n")
                    
                    if consistency.get('consistent', True):
                        f.write("✅ **所有模型参数一致**\n\n")
                    else:
                        f.write("⚠️ **发现参数不一致**\n\n")
                        
                        for inconsistency in consistency.get('inconsistencies', []):
                            param_name = inconsistency['parameter']
                            f.write(f"### {param_name}\n\n")
                            
                            for model, value in inconsistency['values'].items():
                                f.write(f"- {model}: {value}\n")
                            f.write("\n")
                        
                        # 建议
                        if consistency.get('recommendations'):
                            f.write("### 建议\n\n")
                            for rec in consistency['recommendations']:
                                f.write(f"- {rec}\n")
                            f.write("\n")
                
                # 兼容性检查
                if 'compatibility_check' in results:
                    compatibility = results['compatibility_check']
                    f.write("## 训练兼容性检查\n\n")
                    
                    if compatibility.get('compatible', True):
                        f.write("✅ **所有模型与训练配置兼容**\n\n")
                    else:
                        f.write("❌ **发现兼容性问题**\n\n")
                        
                        for issue in compatibility.get('issues', []):
                            f.write(f"- ❌ {issue}\n")
                        
                        for warning in compatibility.get('warnings', []):
                            f.write(f"- ⚠️ {warning}\n")
                        
                        f.write("\n")
                
                # 详细错误和警告
                f.write("## 详细问题列表\n\n")
                
                for model_name in self.available_models:
                    model_result = results['models'].get(model_name, {})
                    errors = model_result.get('errors', [])
                    warnings = model_result.get('warnings', [])
                    
                    if errors or warnings:
                        f.write(f"### {model_name}\n\n")
                        
                        if errors:
                            f.write("**错误**:\n")
                            for error in errors:
                                f.write(f"- ❌ {error}\n")
                            f.write("\n")
                        
                        if warnings:
                            f.write("**警告**:\n")
                            for warning in warnings:
                                f.write(f"- ⚠️ {warning}\n")
                            f.write("\n")
                
                # 建议操作
                f.write("## 建议操作\n\n")
                
                if summary['missing_configs'] > 0:
                    f.write("### 缺失配置文件\n\n")
                    f.write("为以下模型创建配置文件:\n")
                    for model_name in self.available_models:
                        if not results['models'].get(model_name, {}).get('structure_valid', False):
                            f.write(f"- `configs/model/{model_name}.yaml`\n")
                    f.write("\n")
                
                if summary['implementation_found'] < len(self.available_models):
                    f.write("### 缺失实现文件\n\n")
                    f.write("为以下模型添加实现:\n")
                    for model_name in self.available_models:
                        impl_result = results['implementation_check'].get(model_name, {})
                        if not impl_result.get('implementation_exists', False):
                            f.write(f"- `models/{model_name}.py`\n")
                    f.write("\n")
                
            logger.info(f"生成Markdown报告: {output_file}")
            
        except Exception as e:
            logger.error(f"生成Markdown报告失败: {e}")


def main():
    """主函数"""
    validator = ModelConfigValidator()
    
    try:
        # 运行验证
        results = validator.run_validation()
        
        # 保存报告
        validator.save_validation_report(results)
        
        # 输出摘要
        summary = results['summary']
        total_models = len(validator.available_models)
        print(f"\n📋 模型配置验证完成!")
        print(f"📊 总体状态: {summary['overall_status'].upper()}")
        print(f"📁 配置文件: {summary['config_files_found']}/{total_models}")
        print(f"✅ 有效配置: {summary['valid_configs']}/{total_models}")
        print(f"🔧 实现文件: {summary['implementation_found']}/{total_models}")
        
        # 检查是否可以继续训练
        if summary['valid_configs'] >= total_models * 0.8:
            print("✅ 配置验证通过，可以开始批量训练!")
        else:
            print("⚠️ 配置问题较多，建议修复后再开始训练")
            
        print(f"📋 详细报告: runs/batch_training_results/model_config_validation.md")
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()