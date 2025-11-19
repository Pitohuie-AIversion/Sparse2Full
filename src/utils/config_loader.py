"""
配置加载器
支持YAML配置文件加载和合并
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"配置文件加载成功: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
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

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否有效
    """
    for key in required_keys:
        if key not in config:
            logger.error(f"配置缺少必需的键: {key}")
            return False
    
    return True

def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 输出文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的输出文件格式: {config_path.suffix}")
        
        logger.info(f"配置文件已保存: {config_path}")
        
    except Exception as e:
        logger.error(f"配置文件保存失败: {e}")
        raise


class ConfigLoader:
    """配置加载器类
    提供与测试期望一致的面向对象接口。
    """

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """加载配置文件并返回字典"""
        return load_config(config_path)

    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        save_config(config, config_path)