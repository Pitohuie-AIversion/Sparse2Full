"""
日志工具模块

提供统一的日志配置和管理功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: str = "default",
    output_dir: str = "logs",
    format_string: Optional[str] = None
) -> logging.Logger:
    """设置日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则只输出到控制台
        experiment_name: 实验名称
        output_dir: 输出目录
        format_string: 自定义日志格式
        
    Returns:
        配置好的logger实例
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认日志格式
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件handler
    if log_file is None:
        # 自动生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_path / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = Path(log_file)
        if not log_file.is_absolute():
            log_file = output_path / log_file
    
    # 确保日志文件目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 创建实验logger
    logger = logging.getLogger(experiment_name)
    logger.info(f"日志系统初始化完成")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"实验名称: {experiment_name}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的logger
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """实验日志管理器
    
    提供实验级别的日志管理和元数据记录
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验元数据
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "log_files": [],
            "checkpoints": [],
            "metrics": {}
        }
        
        # 初始化日志
        self.logger = setup_logging(
            experiment_name=experiment_name,
            output_dir=str(self.output_dir)
        )
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """记录实验开始
        
        Args:
            config: 实验配置
        """
        self.logger.info("=" * 60)
        self.logger.info(f"实验开始: {self.experiment_name}")
        self.logger.info("=" * 60)
        
        # 记录配置
        config_file = self.output_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.metadata["config_file"] = str(config_file)
        self.logger.info(f"配置已保存到: {config_file}")
    
    def log_experiment_end(self, final_metrics: Dict[str, Any]):
        """记录实验结束
        
        Args:
            final_metrics: 最终指标
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["final_metrics"] = final_metrics
        
        self.logger.info("=" * 60)
        self.logger.info(f"实验结束: {self.experiment_name}")
        self.logger.info(f"最终指标: {final_metrics}")
        self.logger.info("=" * 60)
        
        # 保存元数据
        metadata_file = self.output_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验元数据已保存到: {metadata_file}")
    
    def log_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """记录检查点
        
        Args:
            checkpoint_path: 检查点路径
            epoch: 轮次
            metrics: 指标
        """
        checkpoint_info = {
            "path": checkpoint_path,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self.logger.info(f"检查点已保存: {checkpoint_path} (epoch {epoch})")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录指标
        
        Args:
            metrics: 指标字典
            step: 步数或轮次
        """
        if step is not None:
            if str(step) not in self.metadata["metrics"]:
                self.metadata["metrics"][str(step)] = {}
            self.metadata["metrics"][str(step)].update(metrics)
        
        # 记录到日志文件
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step} - {metrics_str}")
        else:
            self.logger.info(f"Metrics - {metrics_str}")
    
    def get_logger(self) -> logging.Logger:
        """获取logger实例
        
        Returns:
            logger实例
        """
        return self.logger