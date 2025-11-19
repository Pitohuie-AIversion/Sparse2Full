"""
多GPU分布式训练管理器
支持PyTorch DDP和FSDP分布式训练策略
基于技术方案实现高性能分布式训练
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision, CPUOffload, BackwardPrefetch
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy, enable_wrap, wrap
)
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import gc
from pathlib import Path
import json

# 导入硬件优化器
from src.optimizers.hardware_profiler import HardwareProfiler
from src.optimizers.gpu_optimizer import GPUOptimizer
from src.optimizers.mixed_precision_trainer import MixedPrecisionTrainer

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """分布式训练配置"""
    # 基础配置
    backend: str = "nccl"  # "nccl", "gloo", "mpi"
    strategy: str = "ddp"  # "ddp", "fsdp", "deepspeed"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # 训练配置
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # FSDP配置
    fsdp_wrap_policy: str = "size_based"  # "size_based", "transformer_layer", "custom"
    fsdp_mixed_precision: bool = True
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "backward_pre"  # "backward_pre", "backward_post"
    fsdp_forward_prefetch: bool = True
    fsdp_use_orig_params: bool = True
    
    # 性能优化
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # 通信优化
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    # 内存优化
    clear_cache_frequency: int = 100
    gc_frequency: int = 500
    
    # 监控配置
    log_frequency: int = 10
    profile_memory: bool = True

class DistributedTrainingManager:
    """
    多GPU分布式训练管理器
    支持DDP和FSDP策略，自动硬件优化
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.is_main_process = False
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # 性能监控
        self.training_stats = {
            'total_steps': 0,
            'total_time': 0.0,
            'communication_time': 0.0,
            'memory_usage': [],
            'throughput_samples_per_sec': [],
            'gpu_utilization': []
        }
        
        # 硬件优化器
        self.hardware_profiler = HardwareProfiler()
        self.gpu_optimizer = GPUOptimizer()
        
        logger.info(f"分布式训练管理器初始化: strategy={config.strategy}, "
                   f"world_size={config.world_size}")
    
    def initialize_distributed(self):
        """初始化分布式环境"""
        try:
            # 设置环境变量
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
            
            self.is_initialized = True
            self.is_main_process = (self.config.rank == 0)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
            
            # 设置CUDA设备
            torch.cuda.set_device(self.device)
            
            logger.info(f"分布式环境初始化成功: rank={self.config.rank}, "
                       f"world_size={self.config.world_size}, device={self.device}")
            
            # 同步所有进程
            dist.barrier()
            
        except Exception as e:
            logger.error(f"分布式初始化失败: {e}")
            raise RuntimeError(f"分布式初始化失败: {e}")
    
    def setup_model(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> tuple:
        """
        设置模型用于分布式训练
        
        Args:
            model: 要训练的模型
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            (分布式模型, 优化器, 调度器)
        """
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        # 移动模型到设备
        model = model.to(self.device)
        
        # 根据策略包装模型
        if self.config.strategy == "ddp":
            model = self._setup_ddp_model(model)
        elif self.config.strategy == "fsdp":
            model = self._setup_fsdp_model(model)
        else:
            raise ValueError(f"不支持的分布式策略: {self.config.strategy}")
        
        # 设置优化器和调度器
        if optimizer is not None:
            # 重新创建优化器，确保参数正确
            optimizer_state = optimizer.state_dict()
            optimizer = type(optimizer)(
                model.parameters(),
                **{k: v for k, v in optimizer.defaults.items() if k != 'params'}
            )
            optimizer.load_state_dict(optimizer_state)
        
        # 设置混合精度训练
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        logger.info(f"模型设置完成: strategy={self.config.strategy}, "
                   f"device={self.device}")
        
        return model, optimizer, scheduler
    
    def _setup_ddp_model(self, model: nn.Module) -> nn.Module:
        """设置DDP模型"""
        ddp_config = {
            'device_ids': [self.config.local_rank],
            'output_device': self.config.local_rank,
            'find_unused_parameters': self.config.find_unused_parameters,
            'gradient_as_bucket_view': self.config.gradient_as_bucket_view,
            'static_graph': self.config.static_graph,
            'broadcast_buffers': self.config.broadcast_buffers,
            'bucket_cap_mb': self.config.bucket_cap_mb
        }
        
        # 应用GPU优化
        model = self.gpu_optimizer.optimize_model(model)
        
        return DDP(model, **ddp_config)
    
    def _setup_fsdp_model(self, model: nn.Module) -> nn.Module:
        """设置FSDP模型"""
        # 配置混合精度
        if self.config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        else:
            mixed_precision = None
        
        # 配置CPU卸载
        cpu_offload = CPUOffload(offload_params=self.config.fsdp_cpu_offload) if self.config.fsdp_cpu_offload else None
        
        # 配置反向预取
        backward_prefetch = {
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST
        }.get(self.config.fsdp_backward_prefetch, BackwardPrefetch.BACKWARD_PRE)
        
        # 自动包装策略
        if self.config.fsdp_wrap_policy == "size_based":
            auto_wrap_policy = size_based_auto_wrap_policy
        else:
            auto_wrap_policy = None
        
        # FSDP配置
        fsdp_config = {
            'mixed_precision': mixed_precision,
            'cpu_offload': cpu_offload,
            'backward_prefetch': backward_prefetch,
            'forward_prefetch': self.config.fsdp_forward_prefetch,
            'use_orig_params': self.config.fsdp_use_orig_params,
            'auto_wrap_policy': auto_wrap_policy
        }
        
        # 应用GPU优化
        model = self.gpu_optimizer.optimize_model(model)
        
        return FSDP(model, **fsdp_config)
    
    def train_step(self, data_loader, loss_fn: Callable, 
                  update_model: bool = True) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            data_loader: 数据加载器
            loss_fn: 损失函数
            update_model: 是否更新模型参数
            
        Returns:
            训练统计信息
        """
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        step_time = 0.0
        comm_time = 0.0
        
        start_time = time.time()
        
        # 获取一个批次的数据
        try:
            batch = next(iter(data_loader))
        except StopIteration:
            return {
                'loss': 0.0,
                'samples_per_sec': 0.0,
                'step_time': 0.0,
                'communication_time': 0.0
            }
        
        # 准备数据
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device, non_blocking=True)
            targets = batch[1].to(self.device, non_blocking=True)
        else:
            inputs = batch.to(self.device, non_blocking=True)
            targets = None
        
        batch_size = inputs.size(0)
        
        # 前向传播
        comm_start = time.time()
        
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
        
        comm_time += time.time() - comm_start
        
        # 反向传播
        if update_model:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # 梯度同步
                comm_start = time.time()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                comm_time += time.time() - comm_start
            else:
                loss.backward()
                
                # 梯度同步
                comm_start = time.time()
                self.optimizer.step()
                comm_time += time.time() - comm_start
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 清零梯度
            self.optimizer.zero_grad()
        
        step_time = time.time() - start_time
        
        # 同步损失值
        loss_tensor = torch.tensor(loss.item()).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.config.world_size
        
        # 更新统计信息
        total_loss += avg_loss
        total_samples += batch_size
        
        # 性能监控
        samples_per_sec = batch_size / step_time
        
        self.training_stats['total_steps'] += 1
        self.training_stats['total_time'] += step_time
        self.training_stats['communication_time'] += comm_time
        self.training_stats['throughput_samples_per_sec'].append(samples_per_sec)
        
        # 内存监控
        if self.config.profile_memory and self.training_stats['total_steps'] % self.config.log_frequency == 0:
            memory_stats = self._get_memory_stats()
            self.training_stats['memory_usage'].append(memory_stats)
            
            if self.is_main_process:
                logger.info(f"内存使用: {memory_stats}")
        
        # 清理缓存
        if self.training_stats['total_steps'] % self.config.clear_cache_frequency == 0:
            torch.cuda.empty_cache()
        
        # 垃圾回收
        if self.training_stats['total_steps'] % self.config.gc_frequency == 0:
            gc.collect()
        
        return {
            'loss': avg_loss,
            'samples_per_sec': samples_per_sec,
            'step_time': step_time,
            'communication_time': comm_time
        }
    
    def validate(self, val_loader, metrics_fn: Callable) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            metrics_fn: 评估函数
            
        Returns:
            验证指标
        """
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        self.model.eval()
        
        total_metrics = {}
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 准备数据
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)
                else:
                    inputs = batch.to(self.device, non_blocking=True)
                    targets = None
                
                batch_size = inputs.size(0)
                
                # 前向传播
                if self.config.mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        metrics = metrics_fn(outputs, targets) if targets is not None else metrics_fn(outputs)
                else:
                    outputs = self.model(inputs)
                    metrics = metrics_fn(outputs, targets) if targets is not None else metrics_fn(outputs)
                
                # 累积指标
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value * batch_size
                else:
                    if 'loss' not in total_metrics:
                        total_metrics['loss'] = 0.0
                    total_metrics['loss'] += metrics * batch_size
                
                total_samples += batch_size
        
        # 同步所有进程的指标
        for key in total_metrics:
            metric_tensor = torch.tensor(total_metrics[key]).to(self.device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            total_metrics[key] = metric_tensor.item() / (total_samples * self.config.world_size)
        
        return total_metrics
    
    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """保存检查点"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_stats': self.training_stats,
            'config': self.config,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载缩放器状态
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 加载训练统计
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        logger.info(f"检查点已加载: {filepath}")
        
        return checkpoint
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        
        # GPU内存统计
        memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # GPU利用率
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.config.local_rank)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_stats['gpu_utilization'] = utilization.gpu
                memory_stats['memory_utilization'] = utilization.memory
            except ImportError:
                pass
        
        return memory_stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.is_main_process:
            return {}
        
        summary = {
            'total_steps': self.training_stats['total_steps'],
            'total_time': self.training_stats['total_time'],
            'average_step_time': self.training_stats['total_time'] / max(self.training_stats['total_steps'], 1),
            'total_communication_time': self.training_stats['communication_time'],
            'average_throughput': np.mean(self.training_stats['throughput_samples_per_sec']) if self.training_stats['throughput_samples_per_sec'] else 0.0,
            'peak_memory_gb': max([stats.get('gpu_allocated', 0) for stats in self.training_stats['memory_usage']]) if self.training_stats['memory_usage'] else 0.0,
            'gpu_utilization': np.mean([stats.get('gpu_utilization', 0) for stats in self.training_stats['memory_usage']]) if self.training_stats['memory_usage'] else 0.0
        }
        
        return summary
    
    def cleanup(self):
        """清理分布式环境"""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("分布式环境已清理")

# 分布式训练辅助函数
def spawn_distributed_training(train_fn: Callable, config: DistributedConfig, *args, **kwargs):
    """
    启动分布式训练
    
    Args:
        train_fn: 训练函数
        config: 分布式配置
        *args: 传递给训练函数的参数
        **kwargs: 传递给训练函数的关键字参数
    """
    if config.world_size <= 1:
        # 单GPU训练
        config.rank = 0
        config.local_rank = 0
        manager = DistributedTrainingManager(config)
        manager.initialize_distributed()
        train_fn(manager, *args, **kwargs)
        manager.cleanup()
    else:
        # 多GPU训练
        mp.spawn(
            _distributed_worker,
            args=(train_fn, config, args, kwargs),
            nprocs=config.world_size,
            join=True
        )

def _distributed_worker(rank: int, train_fn: Callable, config: DistributedConfig, args: tuple, kwargs: dict):
    """分布式工作进程"""
    config.rank = rank
    config.local_rank = rank % torch.cuda.device_count()
    
    manager = DistributedTrainingManager(config)
    manager.initialize_distributed()
    
    try:
        train_fn(manager, *args, **kwargs)
    finally:
        manager.cleanup()