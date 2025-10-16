"""配置管理工具

提供Hydra配置文件的加载和合并功能。
按照开发手册要求，支持分层配置和快照保存。
"""

import os
import platform
import subprocess
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import yaml
import torch


def load_config(config_path: str) -> DictConfig:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """合并多个配置
    
    Args:
        *configs: 配置对象列表
        
    Returns:
        合并后的配置
    """
    merged = OmegaConf.create({})
    
    for config in configs:
        merged = OmegaConf.merge(merged, config)
    
    return merged


def save_config_snapshot(config: DictConfig, save_path: str):
    """保存配置快照
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(config, f)


def get_environment_info() -> Dict[str, Any]:
    """获取环境信息
    
    Returns:
        环境信息字典
    """
    env_info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable': os.sys.executable,
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    # 获取GPU信息
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': gpu_props.name,
                'memory_total': gpu_props.total_memory,
                'memory_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
        env_info['pytorch']['gpu_info'] = gpu_info
    
    # 获取Git信息（如果可用）
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                         cwd=os.getcwd(), text=True).strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                           cwd=os.getcwd(), text=True).strip()
        env_info['git'] = {
            'commit_hash': git_hash,
            'branch': git_branch
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info['git'] = {'error': 'Git not available or not a git repository'}
    
    return env_info


def get_default_config() -> DictConfig:
    """获取默认配置
    
    Returns:
        默认配置对象
    """
    default_config = {
        'model': {
            'name': 'SwinUNet',
            'in_channels': 3,
            'out_channels': 3,
            'img_size': 256,
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 100,
            'warmup_steps': 1000,
        },
        'loss': {
            'rec_weight': 1.0,
            'spec_weight': 0.5,
            'dc_weight': 1.0,
        },
        'data': {
            'dataset': 'PDEBench',
            'data_root': './data',
            'img_size': 256,
        }
    }
    
    return OmegaConf.create(default_config)