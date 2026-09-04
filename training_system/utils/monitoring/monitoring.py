"""
训练监控和验证管道
提供实时监控、指标跟踪、资源监控和可视化功能
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.metrics_history = defaultdict(list)
        self.resource_history = defaultdict(list)
        
        # 创建可视化目录
        self.fig_dir = output_dir / "figs"
        self.fig_dir.mkdir(exist_ok=True)
        
        # 初始化TensorBoard
        try:
            from src.monitoring import TensorBoardLogger
            self.tb_logger = TensorBoardLogger(output_dir / "tensorboard")
            self.tb_writer = self.tb_logger.writer
        except Exception:
            self.tb_logger = None
            self.tb_writer = SummaryWriter(str(output_dir / "tensorboard"))
        
        # 监控配置
        self.log_interval = getattr(config, 'log_interval', 10)
        self.save_interval = getattr(config, 'save_interval', 100)
        self.plot_interval = getattr(config, 'plot_interval', 500)
        
        # 性能指标
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = getattr(config, 'early_stop_patience', 10)
        
        logger.info(f"训练监控器初始化完成，输出目录: {output_dir}")
    
    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float], 
                   phase: str = 'train') -> None:
        """记录指标"""
        timestamp = time.time()
        
        # 记录到历史
        for key, value in metrics.items():
            metric_key = f"{phase}_{key}"
            self.metrics_history[metric_key].append({
                'epoch': epoch,
                'step': step,
                'value': value,
                'timestamp': timestamp
            })
            
            # 记录到TensorBoard
            self.tb_writer.add_scalar(f"{phase}/{key}", value, step)
        
        # 控制台输出
        if step % self.log_interval == 0:
            metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"[{phase.upper()}] Epoch {epoch}, Step {step}: {metric_str}")
        
        # 检查最佳验证指标
        if phase == 'val' and 'loss' in metrics:
            if metrics['loss'] < self.best_val_metric:
                self.best_val_metric = metrics['loss']
                self.patience_counter = 0
                logger.info(f"🎯 新的最佳验证指标: {self.best_val_metric:.6f}")
            else:
                self.patience_counter += 1
    
    def log_resources(self, step: int, resources: Dict[str, float]) -> None:
        """记录资源使用情况"""
        timestamp = time.time()
        
        for key, value in resources.items():
            self.resource_history[key].append({
                'step': step,
                'value': value,
                'timestamp': timestamp
            })
            
            # 记录到TensorBoard
            self.tb_writer.add_scalar(f"resources/{key}", value, step)
    
    def should_early_stop(self) -> bool:
        """检查是否应该早停"""
        if self.patience_counter >= self.early_stop_patience:
            logger.info(f"早停触发，耐心计数器: {self.patience_counter}")
            return True
        return False
    
    def plot_metrics(self, epoch: int, step: int) -> None:
        """绘制指标图表"""
        if step % self.plot_interval != 0:
            return
        
        try:
            # 训练指标
            train_metrics = {k: v for k, v in self.metrics_history.items() 
                           if k.startswith('train_')}
            val_metrics = {k: v for k, v in self.metrics_history.items() 
                         if k.startswith('val_')}
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Epoch {epoch}, Step {step}')
            
            # 绘制损失曲线
            self._plot_metric_curves(axes[0, 0], train_metrics, val_metrics, 'loss')
            axes[0, 0].set_title('Loss Curves')
            
            # 绘制其他指标
            self._plot_metric_curves(axes[0, 1], train_metrics, val_metrics, 'mae')
            axes[0, 1].set_title('MAE Curves')
            
            # 绘制资源使用
            self._plot_resource_usage(axes[1, 0], step)
            axes[1, 0].set_title('Resource Usage')
            
            # 绘制学习率
            self._plot_learning_rate(axes[1, 1], step)
            axes[1, 1].set_title('Learning Rate Schedule')
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = self.fig_dir / f"training_progress_step_{step}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练进度图已保存: {plot_file}")
            
        except Exception as e:
            logger.error(f"绘制指标图表失败: {e}")
    
    def _plot_metric_curves(self, ax, train_metrics: Dict, val_metrics: Dict, 
                             metric_name: str) -> None:
        """绘制指标曲线"""
        train_key = f'train_{metric_name}'
        val_key = f'val_{metric_name}'
        
        if train_key in train_metrics:
            train_data = train_metrics[train_key]
            steps = [item['step'] for item in train_data]
            values = [item['value'] for item in train_data]
            ax.plot(steps, values, label=f'Train {metric_name}', alpha=0.7)
        
        if val_key in val_metrics:
            val_data = val_metrics[val_key]
            steps = [item['step'] for item in val_data]
            values = [item['value'] for item in val_data]
            ax.plot(steps, values, label=f'Val {metric_name}', alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_resource_usage(self, ax, current_step: int) -> None:
        """绘制资源使用情况"""
        resource_keys = ['gpu_memory', 'cpu_memory', 'gpu_utilization']
        
        for key in resource_keys:
            if key in self.resource_history:
                data = self.resource_history[key]
                steps = [item['step'] for item in data if item['step'] <= current_step]
                values = [item['value'] for item in data if item['step'] <= current_step]
                
                if steps and values:
                    ax.plot(steps, values, label=key.replace('_', ' ').title())
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_rate(self, ax, current_step: int) -> None:
        """绘制学习率"""
        if 'learning_rate' in self.metrics_history:
            lr_data = self.metrics_history['learning_rate']
            steps = [item['step'] for item in lr_data if item['step'] <= current_step]
            values = [item['value'] for item in lr_data if item['step'] <= current_step]
            
            if steps and values:
                ax.plot(steps, values, 'g-', label='Learning Rate')
                ax.set_xlabel('Step')
                ax.set_ylabel('Learning Rate')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    def save_metrics(self) -> None:
        """保存指标历史"""
        metrics_file = self.output_dir / "metrics_history.json"
        
        # 转换历史数据为可序列化格式
        serializable_history = {}
        for key, data in self.metrics_history.items():
            serializable_history[key] = [
                {
                    'epoch': item['epoch'],
                    'step': item['step'],
                    'value': float(item['value']),
                    'timestamp': item['timestamp']
                }
                for item in data
            ]
        
        with open(metrics_file, "w") as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"指标历史已保存: {metrics_file}")
    
    def close(self) -> None:
        """关闭监控器"""
        self.save_metrics()
        self.tb_writer.close()
        logger.info("训练监控器已关闭")


class ResourceMonitor:
    """资源使用监控器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.initial_memory = self._get_gpu_memory()
        
        logger.info(f"资源监控器初始化，设备: {device}")
    
    def get_resource_stats(self) -> Dict[str, float]:
        """获取当前资源统计"""
        stats = {}
        
        # GPU内存使用
        if self.device.type == 'cuda':
            stats['gpu_memory'] = self._get_gpu_memory()
            stats['gpu_utilization'] = self._get_gpu_utilization()
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
        
        # CPU内存使用
        import psutil
        memory_info = psutil.virtual_memory()
        stats['cpu_memory'] = memory_info.percent
        stats['cpu_utilization'] = psutil.cpu_percent(interval=1)
        
        return stats
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存使用"""
        if self.device.type != 'cuda':
            return 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
            info = pynvml.nvmlDeviceGetMemoryInfo()
            return (info.used / info.total) * 100
        except ImportError:
            # 备用方法
            return (torch.cuda.memory_allocated(self.device) / torch.cuda.get_device_properties(self.device).total_memory) * 100
        except Exception:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        if self.device.type != 'cuda':
            return 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates()
            return utilization.gpu
        except Exception:
            return 0.0


class ValidationPipeline:
    """验证管道"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # 验证配置
        self.val_interval = getattr(config, 'val_interval', 100)
        self.val_samples = getattr(config, 'val_samples', 10)
        
        logger.info(f"验证管道初始化，验证间隔: {self.val_interval}")
    
    def validate_model(self, model: nn.Module, val_loader, criterion, 
                        epoch: int, step: int) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        
        total_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= self.val_samples:
                    break
                
                # 前向传播
                inputs, targets = self._prepare_batch(batch)
                outputs = model(inputs)
                
                # 计算损失
                loss_dict = criterion(outputs, targets)
                
                # 累积指标
                for key, value in loss_dict.items():
                    total_metrics[f'val_{key}'] += value.item()
                
                # 计算额外指标
                additional_metrics = self._compute_additional_metrics(outputs, targets)
                for key, value in additional_metrics.items():
                    total_metrics[f'val_{key}'] += value
                
                num_batches += 1
        
        # 计算平均值
        avg_metrics = {}
        for key, value in total_metrics.items():
            avg_metrics[key] = value / num_batches
        
        model.train()
        
        logger.info(f"验证完成 - Epoch {epoch}, Step {step}")
        return avg_metrics
    
    def _prepare_batch(self, batch):
        """准备批次数据"""
        # 根据批次结构提取输入和目标
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            # 假设batch包含所有必要信息
            inputs = batch.get('input', batch.get('inputs'))
            targets = batch.get('target', batch.get('targets'))
        
        if inputs is None or targets is None:
            raise ValueError("无法从批次中提取输入和目标数据")
        
        return inputs.to(self.device), targets.to(self.device)
    
    def _compute_additional_metrics(self, outputs: torch.Tensor, 
                                   targets: torch.Tensor) -> Dict[str, float]:
        """计算额外指标"""
        metrics = {}
        
        # 相对L2误差
        rel_l2 = torch.norm(outputs - targets) / torch.norm(targets)
        metrics['rel_l2'] = rel_l2.item()
        
        # MAE
        mae = torch.mean(torch.abs(outputs - targets))
        metrics['mae'] = mae.item()
        
        # PSNR (假设数据在0-1范围)
        mse = torch.mean((outputs - targets) ** 2)
        if mse > 0:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'] = psnr.item()
        else:
            metrics['psnr'] = float('inf')
        
        return metrics
    
    def check_data_consistency(self, model: nn.Module, sample_batch) -> bool:
        """检查数据一致性"""
        try:
            model.eval()
            
            with torch.no_grad():
                inputs, targets = self._prepare_batch(sample_batch)
                outputs = model(inputs)
                
                # 检查输出形状
                if outputs.shape != targets.shape:
                    logger.error(f"输出形状不匹配: {outputs.shape} vs {targets.shape}")
                    return False
                
                # 检查数值稳定性
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.error("输出包含NaN或Inf值")
                    return False
                
                # 检查值域
                output_range = torch.max(outputs) - torch.min(outputs)
                if output_range < 1e-8:
                    logger.warning("输出值域过小，可能存在死神经元")
            
            model.train()
            return True
            
        except Exception as e:
            logger.error(f"数据一致性检查失败: {e}")
            return False


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.counter = 0
        self.best_state_dict = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y - min_delta
        else:
            self.is_better = lambda x, y: x > y + min_delta
    
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        """检查是否应该早停"""
        if self.best_value is None:
            self.best_value = current_value
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        elif self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                    logger.info("恢复到最佳权重")
        
        return self.early_stop
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'best_value': self.best_value,
            'counter': self.counter,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.best_value = state_dict['best_value']
        self.counter = state_dict['counter']
        self.early_stop = state_dict['early_stop']