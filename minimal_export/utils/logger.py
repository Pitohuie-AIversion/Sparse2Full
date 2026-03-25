"""日志记录模块

提供统一的日志记录功能
支持文件和控制台输出
兼容测试中的 `Logger` 类导入（包装 `setup_logger`）。
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str, log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Logger:
    """轻量包装类，兼容测试对 `utils.logger.Logger` 的期望。

    用法：
        logger = Logger(name, log_file).get()
        logger.info("message")
    """

    def __init__(
        self, name: str, log_file: Path | None = None, level: int = logging.INFO
    ):
        self._logger = setup_logger(name, log_file, level)

    def get(self) -> logging.Logger:
        return self._logger
