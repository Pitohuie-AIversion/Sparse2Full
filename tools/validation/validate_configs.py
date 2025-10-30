#!/usr/bin/env python3
"""
验证自动生成的配置文件
检查YAML格式、数据路径、模型参数等
"""

import os
import yaml
import h5py
from pathlib import Path
from typing import Dict, List, Any, Tuple

def validate_yaml_format(config_path: str) -> Tuple[bool, str]:
    """验证YAML格式是否正确"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return True, "YAML格式正确"
    except yaml.YAMLError as e:
        return False, f"YAML格式错误: {e}"
    except Exception as e:
        return False, f"文件读取错误: {e}"

def check_data_path(config: Dict[str, Any]) -> Tuple[bool, str]:
    """检查数据路径是否存在"""
    try:
        data_path = config.get('data', {}).get('data_path', '')
        if not data_path:
            return False, "未找到data_path配置"
        
        if not os.path.exists(data_path):
            return False, f"数据文件不存在: {data_path}"
        
        # 尝试打开HDF5文件
        try:
            with h5py.File(data_path, 'r') as f:
                keys = list(f.keys())
                return True, f"数据文件可访问，包含键: {keys[:5]}..."
        except Exception as e:
            return False, f"无法读取HDF5文件: {e}"
            
    except Exception as e:
        return False, f"检查数据路径时出错: {e}"

def validate_model_params(config: Dict[str, Any]) -> Tuple[bool, str]:
    """验证模型参数配置"""
    try:
        model_config = config.get('model', {})
        if not model_config:
            return False, "未找到model配置"
        
        # 检查必要参数
        required_params = ['name', 'in_channels', 'out_channels']
        missing_params = [p for p in required_params if p not in model_config]
        if missing_params:
            return False, f"缺少必要模型参数: {missing_params}"
        
        # 检查图像尺寸一致性
        model_size = model_config.get('image_size', 0)
        data_size = config.get('data', {}).get('image_size', 0)
        if model_size != data_size:
            return False, f"模型图像尺寸({model_size})与数据尺寸({data_size})不匹配"
        
        return True, "模型参数配置正确"
        
    except Exception as e:
        return False, f"验证模型参数时出错: {e}"

def validate_training_params(config: Dict[str, Any]) -> Tuple[bool, str]:
    """验证训练参数配置"""
    try:
        training_config = config.get('training', {})
        if not training_config:
            return False, "未找到training配置"
        
        # 检查学习率
        lr = training_config.get('learning_rate', 0)
        if lr <= 0 or lr > 1:
            return False, f"学习率设置不合理: {lr}"
        
        # 检查批处理大小
        batch_size = config.get('data', {}).get('batch_size', 0)
        if batch_size <= 0:
            return False, f"批处理大小设置不合理: {batch_size}"
        
        # 检查训练轮数
        max_epochs = training_config.get('max_epochs', 0)
        if max_epochs <= 0:
            return False, f"训练轮数设置不合理: {max_epochs}"
        
        return True, "训练参数配置正确"
        
    except Exception as e:
        return False, f"验证训练参数时出错: {e}"

def validate_single_config(config_path: str) -> Dict[str, Any]:
    """验证单个配置文件"""
    result = {
        'config_path': config_path,
        'config_name': Path(config_path).stem,
        'valid': True,
        'issues': []
    }
    
    print(f"\n🔍 验证配置: {Path(config_path).name}")
    
    # 1. 验证YAML格式
    yaml_valid, yaml_msg = validate_yaml_format(config_path)
    if not yaml_valid:
        result['valid'] = False
        result['issues'].append(f"YAML格式: {yaml_msg}")
        print(f"  ❌ YAML格式: {yaml_msg}")
        return result
    else:
        print(f"  ✅ YAML格式: {yaml_msg}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 检查数据路径
    data_valid, data_msg = check_data_path(config)
    if not data_valid:
        result['valid'] = False
        result['issues'].append(f"数据路径: {data_msg}")
        print(f"  ❌ 数据路径: {data_msg}")
    else:
        print(f"  ✅ 数据路径: {data_msg}")
    
    # 3. 验证模型参数
    model_valid, model_msg = validate_model_params(config)
    if not model_valid:
        result['valid'] = False
        result['issues'].append(f"模型参数: {model_msg}")
        print(f"  ❌ 模型参数: {model_msg}")
    else:
        print(f"  ✅ 模型参数: {model_msg}")
    
    # 4. 验证训练参数
    training_valid, training_msg = validate_training_params(config)
    if not training_valid:
        result['valid'] = False
        result['issues'].append(f"训练参数: {training_msg}")
        print(f"  ❌ 训练参数: {training_msg}")
    else:
        print(f"  ✅ 训练参数: {training_msg}")
    
    if result['valid']:
        print(f"  🎉 配置验证通过!")
    else:
        print(f"  ⚠️  发现 {len(result['issues'])} 个问题")
    
    return result

def main():
    """主函数"""
    print("🔧 开始验证自动生成的配置文件...")
    
    config_dir = Path("configs/auto_generated")
    if not config_dir.exists():
        print(f"❌ 配置目录不存在: {config_dir}")
        return
    
    # 获取所有YAML配置文件
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        print(f"❌ 未找到配置文件在: {config_dir}")
        return
    
    print(f"📁 找到 {len(config_files)} 个配置文件")
    
    results = []
    valid_configs = []
    invalid_configs = []
    
    # 验证每个配置文件
    for config_file in sorted(config_files):
        result = validate_single_config(str(config_file))
        results.append(result)
        
        if result['valid']:
            valid_configs.append(result['config_name'])
        else:
            invalid_configs.append(result)
    
    # 输出总结
    print(f"\n📊 验证结果总结:")
    print(f"  ✅ 有效配置: {len(valid_configs)}")
    print(f"  ❌ 无效配置: {len(invalid_configs)}")
    
    if valid_configs:
        print(f"\n🎉 可以进行训练的配置:")
        for config_name in valid_configs:
            print(f"  - {config_name}")
    
    if invalid_configs:
        print(f"\n⚠️  需要修复的配置:")
        for config in invalid_configs:
            print(f"  - {config['config_name']}:")
            for issue in config['issues']:
                print(f"    • {issue}")
    
    # 提供训练命令示例
    if valid_configs:
        print(f"\n🚀 训练命令示例:")
        example_config = valid_configs[0]
        print(f"  # 单个配置训练:")
        print(f"  python train.py --config-path configs/auto_generated --config-name {example_config}")
        print(f"  ")
        print(f"  # 批量训练:")
        print(f"  python batch_train_selected_datasets.py")
    
    return len(invalid_configs) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)