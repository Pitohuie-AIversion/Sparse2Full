"""
资源监控工具

监控训练过程中的资源使用情况
"""

import logging
import threading
import time
from typing import Any

import psutil
import torch


class ResourceMonitor:
    """资源监控器"""

    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)

        # 资源使用记录
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)

                # 获取内存使用率
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)

                # 获取GPU使用率（如果可用）
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB

                    self.gpu_usage.append(gpu_percent)
                    self.gpu_memory_usage.append(gpu_memory)

                # 记录日志
                self.logger.info(
                    f"Resource Usage - CPU: {cpu_percent:.1f}%, "
                    f"Memory: {memory.percent:.1f}%"
                )

                if torch.cuda.is_available() and self.gpu_usage:
                    self.logger.info(
                        f"GPU Usage - GPU: {self.gpu_usage[-1]:.1f}%, "
                        f"Memory: {self.gpu_memory_usage[-1]:.2f}GB"
                    )

                # 等待下一个周期
                time.sleep(self.log_interval)

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.log_interval)

    def get_current_stats(self) -> dict[str, Any]:
        """获取当前资源统计"""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        }

        if torch.cuda.is_available():
            stats.update(
                {
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_memory_total_gb": torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / 1024**3,
                }
            )

        return stats

    def get_peak_usage(self) -> dict[str, float]:
        """获取峰值使用情况"""
        peak_stats = {}

        if self.cpu_usage:
            peak_stats["peak_cpu_percent"] = max(self.cpu_usage)

        if self.memory_usage:
            peak_stats["peak_memory_percent"] = max(self.memory_usage)

        if self.gpu_usage:
            peak_stats["peak_gpu_percent"] = max(self.gpu_usage)

        if self.gpu_memory_usage:
            peak_stats["peak_gpu_memory_gb"] = max(self.gpu_memory_usage)

        return peak_stats

    def log_training_resources(self, epoch: int, batch_idx: int):
        """记录训练过程中的资源使用情况"""
        stats = self.get_current_stats()

        self.logger.info(
            f"Epoch {epoch}, Batch {batch_idx} - "
            f"CPU: {stats['cpu_percent']:.1f}%, "
            f"Memory: {stats['memory_percent']:.1f}%"
        )

        if torch.cuda.is_available():
            self.logger.info(
                f"GPU Memory: {stats['gpu_memory_allocated_gb']:.2f}/"
                f"{stats['gpu_memory_total_gb']:.2f}GB "
                f"({stats['gpu_memory_allocated_gb']/stats['gpu_memory_total_gb']*100:.1f}%)"
            )
