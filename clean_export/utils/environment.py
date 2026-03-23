"""环境信息获取工具

用于获取系统环境信息，支持一致性检查和实验记录。
"""

import os
import sys
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np


def get_environment_info() -> Dict[str, Any]:
    """获取环境信息
    
    Returns:
        环境信息字典
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        'numpy_version': np.__version__,
        'working_directory': os.getcwd(),
    }
    
    # 获取GPU信息
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': gpu_props.name,
                'total_memory': gpu_props.total_memory,
                'major': gpu_props.major,
                'minor': gpu_props.minor,
            })
        env_info['gpu_info'] = gpu_info
    
    # 获取Git信息（如果可用）
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        env_info['git_commit'] = git_commit
        
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        env_info['git_branch'] = git_branch
        
        # 检查是否有未提交的更改
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        env_info['git_dirty'] = bool(git_status)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git不可用或不在Git仓库中
        env_info['git_commit'] = None
        env_info['git_branch'] = None
        env_info['git_dirty'] = None
    
    return env_info


def get_system_resources() -> Dict[str, Any]:
    """获取系统资源信息
    
    Returns:
        系统资源信息字典
    """
    import psutil
    
    # CPU信息
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'cpu_percent': psutil.cpu_percent(interval=1),
    }
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_info = {
        'total': memory.total,
        'available': memory.available,
        'percent': memory.percent,
        'used': memory.used,
        'free': memory.free,
    }
    
    # 磁盘信息
    disk = psutil.disk_usage('/')
    disk_info = {
        'total': disk.total,
        'used': disk.used,
        'free': disk.free,
        'percent': (disk.used / disk.total) * 100,
    }
    
    return {
        'cpu': cpu_info,
        'memory': memory_info,
        'disk': disk_info,
        'timestamp': datetime.now().isoformat(),
    }


def format_bytes(bytes_value: int) -> str:
    """格式化字节数为人类可读格式
    
    Args:
        bytes_value: 字节数
        
    Returns:
        格式化后的字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def print_environment_info():
    """打印环境信息"""
    env_info = get_environment_info()
    
    print("=== Environment Information ===")
    print(f"Timestamp: {env_info['timestamp']}")
    print(f"Python: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']['system']} {env_info['platform']['release']}")
    print(f"PyTorch: {env_info['pytorch']['version']}")
    print(f"NumPy: {env_info['numpy_version']}")
    
    if env_info['pytorch']['cuda_available']:
        print(f"CUDA: {env_info['pytorch']['cuda_version']}")
        print(f"cuDNN: {env_info['pytorch']['cudnn_version']}")
        print(f"GPU Count: {env_info['pytorch']['device_count']}")
        
        if 'gpu_info' in env_info:
            for i, gpu in enumerate(env_info['gpu_info']):
                print(f"GPU {i}: {gpu['name']} ({format_bytes(gpu['total_memory'])})")
    
    if env_info['git_commit']:
        print(f"Git: {env_info['git_branch']}@{env_info['git_commit'][:8]}")
        if env_info['git_dirty']:
            print("  (有未提交的更改)")
    
    print(f"Working Directory: {env_info['working_directory']}")
    print("=" * 40)


if __name__ == "__main__":
    print_environment_info()
    
    try:
        resources = get_system_resources()
        print("\n=== System Resources ===")
        print(f"CPU: {resources['cpu']['physical_cores']} cores ({resources['cpu']['logical_cores']} logical)")
        print(f"CPU Usage: {resources['cpu']['cpu_percent']:.1f}%")
        print(f"Memory: {format_bytes(resources['memory']['used'])} / {format_bytes(resources['memory']['total'])} ({resources['memory']['percent']:.1f}%)")
        print(f"Disk: {format_bytes(resources['disk']['used'])} / {format_bytes(resources['disk']['total'])} ({resources['disk']['percent']:.1f}%)")
    except ImportError:
        print("psutil not available, skipping system resources")