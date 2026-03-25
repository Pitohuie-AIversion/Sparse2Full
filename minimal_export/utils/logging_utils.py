"""
日志工具

提供统一的日志配置功能
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    设置日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 日志格式字符串

    Returns:
        日志器
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 如果日志器已经有处理器，先清除它们
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(format_string)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
