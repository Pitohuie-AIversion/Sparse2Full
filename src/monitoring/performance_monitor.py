"""
性能监控器模块
实时监控训练过程中的性能指标和资源使用情况
"""

import time
import psutil
import GPUtil
import threading
import queue
import json
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float = 0.0
    epoch: int = 0
    step: int = 0
    
    # 训练指标
    loss: float = 0.0
    learning_rate: float = 0.0
    batch_time: float = 0.0
    data_loading_time: float = 0.0
    
    # 系统资源
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    
    # GPU资源
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_temperature: float = 0.0
    
    # 训练速度
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # 模型统计
    model_parameters: int = 0
    gradient_norm: float = 0.0
    
    # 额外信息
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """
    实时性能监控器
    监控训练过程中的性能指标和资源使用情况
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        monitoring_interval: float = 1.0,
        enable_tensorboard: bool = True,
        enable_json_logging: bool = True,
        max_history_size: int = 10000
    ):
        """
        初始化性能监控器
        
        Args:
            log_dir: 日志目录路径
            monitoring_interval: 监控间隔（秒）
            enable_tensorboard: 是否启用TensorBoard日志
            enable_json_logging: 是否启用JSON日志记录
            max_history_size: 历史数据最大保存数量
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.monitoring_interval = monitoring_interval
        self.enable_tensorboard = enable_tensorboard
        self.enable_json_logging = enable_json_logging
        self.max_history_size = max_history_size
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        self.metrics_history = deque(maxlen=max_history_size)
        
        # 系统信息
        self.system_info = self._get_system_info()
        self.gpu_available = self._check_gpu_availability()
        
        # TensorBoard写入器
        self.tb_writer = None
        if enable_tensorboard and self.log_dir:
            self.tb_writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
        
        # JSON日志文件
        self.json_log_file = None
        if enable_json_logging and self.log_dir:
            self.json_log_file = self.log_dir / 'performance_metrics.jsonl'
            self.json_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 性能统计
        self.performance_stats = {
            'loss_history': deque(maxlen=1000),
            'batch_time_history': deque(maxlen=100),
            'throughput_history': deque(maxlen=100),
            'resource_usage_history': deque(maxlen=1000)
        }
        
        # 回调函数
        self.callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        logger.info(f"性能监控器初始化完成: log_dir={self.log_dir}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import platform
            import cpuinfo
            
            cpu_info = cpuinfo.get_cpu_info()
            memory = psutil.virtual_memory()
            
            return {
                'platform': platform.platform(),
                'processor': cpu_info.get('brand_raw', 'Unknown'),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'total_memory_gb': memory.total / (1024**3),
                'python_version': platform.python_version()
            }
        except Exception as e:
            logger.warning(f"获取系统信息失败: {e}")
            return {}
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            if torch.cuda.is_available():
                return True
            
            # 尝试使用GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
            
        except Exception as e:
            logger.warning(f"GPU可用性检查失败: {e}")
            return False
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        metrics = {}
        
        try:
            # CPU使用率
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            
            # GPU使用情况（如果可用）
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        metrics['gpu_utilization'] = gpu.load * 100
                        metrics['gpu_memory_used_gb'] = gpu.memoryUsed / 1024
                        metrics['gpu_memory_total_gb'] = gpu.memoryTotal / 1024
                        metrics['gpu_temperature'] = gpu.temperature
                except Exception as e:
                    logger.debug(f"GPU指标收集失败: {e}")
                    
                    # 备用方案：使用PyTorch
                    if torch.cuda.is_available():
                        metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                        metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
                        metrics['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        metrics['gpu_temperature'] = 0  # PyTorch不直接提供温度信息
                        
        except Exception as e:
            logger.warning(f"系统指标收集失败: {e}")
        
        return metrics
    
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("性能监控循环开始")
        
        while self.is_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                
                # 创建性能指标对象
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    **system_metrics
                )
                
                # 添加到队列和历史记录
                self.metrics_queue.put(metrics)
                self.metrics_history.append(metrics)
                
                # 执行回调函数
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")
                
                # 记录详细日志（每10秒一次）
                if len(self.metrics_history) % int(10 / self.monitoring_interval) == 0:
                    self._log_detailed_metrics(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("性能监控循环结束")
    
    def _log_detailed_metrics(self, metrics: PerformanceMetrics):
        """记录详细性能指标"""
        logger.info(
            f"性能监控 - "
            f"CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB/{metrics.memory_total_gb:.1f}GB), "
            f"GPU: {metrics.gpu_utilization:.1f}% ({metrics.gpu_memory_used_gb:.1f}GB/{metrics.gpu_memory_total_gb:.1f}GB)"
        )
    
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            logger.warning("性能监控已经在运行")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("性能监控已停止")
    
    def record_training_metrics(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float = 0.0,
        batch_time: float = 0.0,
        data_loading_time: float = 0.0,
        model_parameters: int = 0,
        gradient_norm: float = 0.0,
        samples_per_second: float = 0.0,
        tokens_per_second: float = 0.0,
        **extra_metrics
    ):
        """记录训练指标"""
        # 获取最新的系统指标
        current_metrics = None
        if not self.metrics_queue.empty():
            current_metrics = self.metrics_queue.get()
        else:
            # 创建新的指标对象
            current_metrics = PerformanceMetrics(
                timestamp=time.time(),
                epoch=epoch,
                step=step
            )
        
        # 更新训练指标
        current_metrics.epoch = epoch
        current_metrics.step = step
        current_metrics.loss = loss
        current_metrics.learning_rate = learning_rate
        current_metrics.batch_time = batch_time
        current_metrics.data_loading_time = data_loading_time
        current_metrics.model_parameters = model_parameters
        current_metrics.gradient_norm = gradient_norm
        current_metrics.samples_per_second = samples_per_second
        current_metrics.tokens_per_second = tokens_per_second
        current_metrics.extra_metrics = extra_metrics
        
        # 更新性能统计
        self.performance_stats['loss_history'].append(loss)
        self.performance_stats['batch_time_history'].append(batch_time)
        self.performance_stats['throughput_history'].append(samples_per_second)
        self.performance_stats['resource_usage_history'].append({
            'cpu': current_metrics.cpu_percent,
            'memory': current_metrics.memory_percent,
            'gpu': current_metrics.gpu_utilization
        })
        
        # 记录到TensorBoard
        if self.tb_writer:
            self._log_to_tensorboard(current_metrics)
        
        # 记录到JSON文件
        if self.enable_json_logging and self.json_log_file:
            self._log_to_json(current_metrics)
        
        # 添加到历史记录
        self.metrics_history.append(current_metrics)
        
        logger.debug(f"训练指标记录完成: epoch={epoch}, step={step}, loss={loss:.6f}")
    
    def _log_to_tensorboard(self, metrics: PerformanceMetrics):
        """记录到TensorBoard"""
        try:
            global_step = metrics.epoch * 10000 + metrics.step  # 简单的全局步数计算
            
            # 训练指标
            self.tb_writer.add_scalar('Training/Loss', metrics.loss, global_step)
            self.tb_writer.add_scalar('Training/Learning_Rate', metrics.learning_rate, global_step)
            self.tb_writer.add_scalar('Training/Samples_Per_Second', metrics.samples_per_second, global_step)
            self.tb_writer.add_scalar('Training/Batch_Time', metrics.batch_time, global_step)
            self.tb_writer.add_scalar('Training/Data_Loading_Time', metrics.data_loading_time, global_step)
            
            # 系统资源
            self.tb_writer.add_scalar('System/CPU_Percent', metrics.cpu_percent, global_step)
            self.tb_writer.add_scalar('System/Memory_Percent', metrics.memory_percent, global_step)
            self.tb_writer.add_scalar('System/GPU_Utilization', metrics.gpu_utilization, global_step)
            self.tb_writer.add_scalar('System/GPU_Memory_Used_GB', metrics.gpu_memory_used_gb, global_step)
            
            # 模型统计
            self.tb_writer.add_scalar('Model/Parameters', metrics.model_parameters, global_step)
            self.tb_writer.add_scalar('Model/Gradient_Norm', metrics.gradient_norm, global_step)
            
            # 额外指标
            for key, value in metrics.extra_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Extra/{key}', value, global_step)
            
        except Exception as e:
            logger.error(f"TensorBoard日志记录失败: {e}")
    
    def _log_to_json(self, metrics: PerformanceMetrics):
        """记录到JSON文件"""
        try:
            log_entry = {
                'timestamp': metrics.timestamp,
                'epoch': metrics.epoch,
                'step': metrics.step,
                'loss': metrics.loss,
                'learning_rate': metrics.learning_rate,
                'batch_time': metrics.batch_time,
                'data_loading_time': metrics.data_loading_time,
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'memory_used_gb': metrics.memory_used_gb,
                'memory_total_gb': metrics.memory_total_gb,
                'gpu_utilization': metrics.gpu_utilization,
                'gpu_memory_used_gb': metrics.gpu_memory_used_gb,
                'gpu_memory_total_gb': metrics.gpu_memory_total_gb,
                'gpu_temperature': metrics.gpu_temperature,
                'samples_per_second': metrics.samples_per_second,
                'tokens_per_second': metrics.tokens_per_second,
                'model_parameters': metrics.model_parameters,
                'gradient_norm': metrics.gradient_norm,
                'extra_metrics': metrics.extra_metrics
            }
            
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"JSON日志记录失败: {e}")
    
    def add_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """移除回调函数"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]  # 最近100个指标
        
        summary = {
            'total_samples': len(self.metrics_history),
            'monitoring_duration': self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            'system_info': self.system_info,
            'gpu_available': self.gpu_available
        }
        
        if recent_metrics:
            # 平均性能指标
            summary['avg_loss'] = np.mean([m.loss for m in recent_metrics if m.loss > 0])
            summary['avg_batch_time'] = np.mean([m.batch_time for m in recent_metrics if m.batch_time > 0])
            summary['avg_throughput'] = np.mean([m.samples_per_second for m in recent_metrics if m.samples_per_second > 0])
            summary['avg_cpu_usage'] = np.mean([m.cpu_percent for m in recent_metrics])
            summary['avg_memory_usage'] = np.mean([m.memory_percent for m in recent_metrics])
            summary['avg_gpu_usage'] = np.mean([m.gpu_utilization for m in recent_metrics])
            
            # 峰值性能
            summary['min_loss'] = np.min([m.loss for m in recent_metrics if m.loss > 0])
            summary['max_throughput'] = np.max([m.samples_per_second for m in recent_metrics if m.samples_per_second > 0])
            summary['max_cpu_usage'] = np.max([m.cpu_percent for m in recent_metrics])
            summary['max_memory_usage'] = np.max([m.memory_percent for m in recent_metrics])
            summary['max_gpu_usage'] = np.max([m.gpu_utilization for m in recent_metrics])
        
        return summary
    
    def save_metrics_report(self, filepath: Optional[str] = None) -> str:
        """保存性能报告"""
        if not filepath and self.log_dir:
            filepath = str(self.log_dir / 'performance_report.json')
        elif not filepath:
            filepath = 'performance_report.json'
        
        summary = self.get_performance_summary()
        
        # 添加详细的历史数据
        if self.metrics_history:
            summary['detailed_history'] = [
                {
                    'timestamp': m.timestamp,
                    'epoch': m.epoch,
                    'step': m.step,
                    'loss': m.loss,
                    'learning_rate': m.learning_rate,
                    'batch_time': m.batch_time,
                    'samples_per_second': m.samples_per_second,
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'gpu_utilization': m.gpu_utilization
                }
                for m in self.metrics_history
            ]
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"性能报告已保存: {filepath}")
        return filepath
    
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        logger.info("性能监控器清理完成")