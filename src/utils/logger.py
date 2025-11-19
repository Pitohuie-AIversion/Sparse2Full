"""
日志设置模块
配置训练日志格式和输出
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
# 可选依赖：colorlog 不存在时回退到标准日志
try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except ImportError:  # pragma: no cover
    colorlog = None
    _HAS_COLORLOG = False

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    format_string: Optional[str] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径，如果为None则不写入文件
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: 控制台日志级别
        file_level: 文件日志级别
        format_string: 自定义日志格式
        
    Returns:
        根日志记录器
    """
    # 设置根日志记录器
    logger = logging.getLogger()
    # 兼容参数：若提供 level（如 logging.WARNING），优先生效
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 默认格式
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    
    # 控制台处理器（带颜色，若不可用则使用普通处理器）
    if _HAS_COLORLOG:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + format_string + "%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)

def log_system_info():
    """记录系统信息"""
    import platform
    import torch
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("系统信息")
    logger.info("=" * 60)
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"NumPy版本: {np.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    else:
        logger.info("CUDA不可用，使用CPU")
    
    logger.info("=" * 60)