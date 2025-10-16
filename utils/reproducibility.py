"""可复现性工具模块

提供随机种子设置、确定性模式配置和环境信息收集功能，
确保实验的可复现性，符合开发手册的黄金法则要求。

开发手册 - 黄金法则：
2. 可复现：同一YAML+种子，验证指标方差≤1e-4

使用方法：
    from utils.reproducibility import set_deterministic_mode, get_environment_info
    
    # 设置确定性模式
    set_deterministic_mode(True)
    
    # 获取环境信息
    env_info = get_environment_info()
"""

import os
import platform
import random
import sys
from typing import Dict, Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置随机种子
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic_mode(enabled: bool = True) -> None:
    """设置确定性模式
    
    Args:
        enabled: 是否启用确定性模式
    """
    if enabled:
        # PyTorch确定性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # 设置环境变量
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # 警告用户可能的性能影响
        import warnings
        warnings.warn(
            "Deterministic mode is enabled. This may reduce performance but ensures reproducibility.",
            UserWarning
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def get_environment_info() -> Dict[str, Any]:
    """获取环境信息
    
    Returns:
        env_info: 环境信息字典
    """
    env_info = {
        # Python环境
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        
        # PyTorch信息
        'torch_version': torch.__version__,
        'torch_cuda_available': torch.cuda.is_available(),
        'torch_cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'torch_cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        
        # CUDA设备信息
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        
        # NumPy信息
        'numpy_version': np.__version__,
        
        # 环境变量
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not_set'),
        
        # 确定性设置
        'torch_deterministic': torch.backends.cudnn.deterministic,
        'torch_benchmark': torch.backends.cudnn.benchmark,
    }
    
    # 添加CUDA设备详细信息
    if torch.cuda.is_available():
        cuda_devices = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            cuda_devices.append({
                'device_id': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'major': device_props.major,
                'minor': device_props.minor,
                'multi_processor_count': device_props.multi_processor_count
            })
        env_info['cuda_devices'] = cuda_devices
    
    return env_info


def verify_reproducibility(results1: Dict[str, float], results2: Dict[str, float], 
                          threshold: float = 1e-4) -> Dict[str, Any]:
    """验证可复现性
    
    Args:
        results1: 第一次运行结果
        results2: 第二次运行结果
        threshold: 差异阈值
        
    Returns:
        verification_result: 验证结果
    """
    verification_result = {
        'is_reproducible': True,
        'differences': {},
        'max_difference': 0.0,
        'threshold': threshold,
        'failed_metrics': []
    }
    
    # 检查共同的指标
    common_metrics = set(results1.keys()) & set(results2.keys())
    
    for metric in common_metrics:
        diff = abs(results1[metric] - results2[metric])
        verification_result['differences'][metric] = diff
        
        if diff > threshold:
            verification_result['is_reproducible'] = False
            verification_result['failed_metrics'].append(metric)
        
        verification_result['max_difference'] = max(
            verification_result['max_difference'], diff
        )
    
    return verification_result


def create_reproducible_config(base_config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """创建可复现的配置
    
    Args:
        base_config: 基础配置
        seed: 随机种子
        
    Returns:
        reproducible_config: 可复现配置
    """
    reproducible_config = base_config.copy()
    
    # 设置随机种子
    reproducible_config['seed'] = seed
    
    # 添加环境信息
    reproducible_config['environment_info'] = get_environment_info()
    
    # 确保确定性设置
    if 'training' not in reproducible_config:
        reproducible_config['training'] = {}
    
    reproducible_config['training']['deterministic'] = True
    reproducible_config['training']['benchmark'] = False
    
    return reproducible_config


def log_reproducibility_info(logger, seed: int) -> None:
    """记录可复现性信息
    
    Args:
        logger: 日志记录器
        seed: 随机种子
    """
    logger.info(f"Reproducibility settings:")
    logger.info(f"  Random seed: {seed}")
    logger.info(f"  Deterministic mode: {torch.backends.cudnn.deterministic}")
    logger.info(f"  Benchmark mode: {torch.backends.cudnn.benchmark}")
    logger.info(f"  PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not_set')}")
    logger.info(f"  CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not_set')}")
    
    # 记录环境信息
    env_info = get_environment_info()
    logger.info(f"Environment info:")
    logger.info(f"  Python: {env_info['python_version'].split()[0]}")
    logger.info(f"  PyTorch: {env_info['torch_version']}")
    logger.info(f"  CUDA available: {env_info['torch_cuda_available']}")
    if env_info['torch_cuda_available']:
        logger.info(f"  CUDA version: {env_info['torch_cuda_version']}")
        logger.info(f"  Device count: {env_info['cuda_device_count']}")