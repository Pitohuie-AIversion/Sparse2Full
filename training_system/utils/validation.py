"""
配置验证工具模块

提供配置文件的验证功能
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> bool:
    """验证配置文件的基本结构
    
    Args:
        config: 配置字典
        required_keys: 必需的关键字列表
        
    Returns:
        如果配置有效返回True
        
    Raises:
        ValueError: 如果配置无效
    """
    if not isinstance(config, dict):
        raise ValueError("配置必须是字典类型")
    
    # 默认必需关键字
    if required_keys is None:
        required_keys = [
            'data', 'model', 'training', 'loss', 'logging'
        ]
    
    # 检查必需关键字
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"配置缺少必需的关键字: {missing_keys}")
    
    # 验证各个部分
    _validate_data_config(config.get('data', {}))
    _validate_model_config(config.get('model', {}))
    _validate_training_config(config.get('training', {}))
    _validate_loss_config(config.get('loss', {}))
    _validate_logging_config(config.get('logging', {}))
    
    return True


def _validate_data_config(data_config: Dict[str, Any]) -> None:
    """验证数据配置"""
    required_data_keys = ['name', 'path', 'input_size', 'output_size']
    missing_keys = [key for key in required_data_keys if key not in data_config]
    
    if missing_keys:
        raise ValueError(f"数据配置缺少必需的关键字: {missing_keys}")
    
    # 验证路径是否存在
    if 'path' in data_config:
        data_path = Path(data_config['path'])
        if not data_path.exists():
            raise ValueError(f"数据路径不存在: {data_path}")


def _validate_model_config(model_config: Dict[str, Any]) -> None:
    """验证模型配置"""
    required_model_keys = ['name', 'in_channels', 'out_channels']
    missing_keys = [key for key in required_model_keys if key not in model_config]
    
    if missing_keys:
        raise ValueError(f"模型配置缺少必需的关键字: {missing_keys}")
    
    # 验证数值参数
    if 'in_channels' in model_config and not isinstance(model_config['in_channels'], int):
        raise ValueError("in_channels 必须是整数")
    
    if 'out_channels' in model_config and not isinstance(model_config['out_channels'], int):
        raise ValueError("out_channels 必须是整数")


def _validate_training_config(training_config: Dict[str, Any]) -> None:
    """验证训练配置"""
    required_training_keys = ['epochs', 'batch_size', 'learning_rate']
    missing_keys = [key for key in required_training_keys if key not in training_config]
    
    if missing_keys:
        raise ValueError(f"训练配置缺少必需的关键字: {missing_keys}")
    
    # 验证数值参数
    if 'epochs' in training_config and training_config['epochs'] <= 0:
        raise ValueError("epochs 必须是正数")
    
    if 'batch_size' in training_config and training_config['batch_size'] <= 0:
        raise ValueError("batch_size 必须是正数")
    
    if 'learning_rate' in training_config and training_config['learning_rate'] <= 0:
        raise ValueError("learning_rate 必须是正数")


def _validate_loss_config(loss_config: Dict[str, Any]) -> None:
    """验证损失配置"""
    if not isinstance(loss_config, dict):
        raise ValueError("损失配置必须是字典类型")
    
    # 验证损失权重
    if 'weights' in loss_config:
        weights = loss_config['weights']
        if not isinstance(weights, dict):
            raise ValueError("损失权重必须是字典类型")
        
        for loss_name, weight in weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"损失权重 {loss_name} 必须是非负数")


def _validate_logging_config(logging_config: Dict[str, Any]) -> None:
    """验证日志配置"""
    if not isinstance(logging_config, dict):
        raise ValueError("日志配置必须是字典类型")
    
    # 验证日志级别
    if 'level' in logging_config:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if logging_config['level'] not in valid_levels:
            raise ValueError(f"无效的日志级别: {logging_config['level']}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML格式无效
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML格式错误: {e}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个配置（后者覆盖前者）
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config_hash(config: Dict[str, Any]) -> str:
    """获取配置的哈希值（用于实验追踪）
    
    Args:
        config: 配置字典
        
    Returns:
        配置哈希值
    """
    import hashlib
    import json
    
    # 将配置转换为排序后的JSON字符串
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    
    # 计算哈希值
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    return config_hash