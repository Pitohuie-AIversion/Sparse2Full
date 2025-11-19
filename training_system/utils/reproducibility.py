"""
可重复性工具模块

提供实验可重复性相关的工具函数
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = False) -> None:
    """设置随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法（可能会影响性能）
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("种子必须是正整数")
    
    # Python内置随机
    random.seed(seed)
    
    # Numpy随机
    np.random.seed(seed)
    
    # PyTorch随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置确定性行为
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        
        # 设置Python哈希种子
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 设置NumPy打印选项
        np.set_printoptions(precision=8, suppress=True)
        
        # 设置PyTorch打印选项
        torch.set_printoptions(precision=8)
    else:
        # 保持性能优先的设置
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def get_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """获取配置了种子的PyTorch生成器
    
    Args:
        seed: 随机种子，如果为None则使用全局种子
        
    Returns:
        PyTorch生成器
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int, seed: int) -> None:
    """DataLoader worker初始化函数
    
    Args:
        worker_id: worker ID
        seed: 基础随机种子
    """
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ReproducibilityConfig:
    """可重复性配置类"""
    
    def __init__(self, seed: int = 42, deterministic: bool = False):
        """初始化可重复性配置
        
        Args:
            seed: 随机种子
            deterministic: 是否使用确定性算法
        """
        self.seed = seed
        self.deterministic = deterministic
        self._original_states = {}
    
    def enable(self) -> None:
        """启用可重复性设置"""
        # 保存原始状态
        self._original_states['torch_deterministic'] = torch.backends.cudnn.deterministic
        self._original_states['torch_benchmark'] = torch.backends.cudnn.benchmark
        
        # 应用设置
        set_seed(self.seed, self.deterministic)
    
    def disable(self) -> None:
        """禁用可重复性设置（恢复到原始状态）"""
        if self._original_states:
            torch.backends.cudnn.deterministic = self._original_states.get('torch_deterministic', False)
            torch.backends.cudnn.benchmark = self._original_states.get('torch_benchmark', True)
    
    def __enter__(self):
        """上下文管理器进入"""
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disable()


def validate_reproducibility() -> bool:
    """验证当前环境是否支持可重复性
    
    Returns:
        如果环境支持可重复性返回True
    """
    try:
        # 检查PyTorch版本
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major < 1 or (major == 1 and minor < 8):
            print(f"警告: PyTorch版本 {torch_version} 可能不支持完整的可重复性")
            return False
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"CUDA版本: {cuda_version}")
        
        return True
        
    except Exception as e:
        print(f"验证可重复性时出错: {e}")
        return False