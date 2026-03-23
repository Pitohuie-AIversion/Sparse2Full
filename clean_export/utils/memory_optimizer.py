#!/usr/bin/env python3
"""内存优化工具

提供时序AR模型的内存优化策略，支持更长的时序预测
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import gc
import psutil
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """内存统计信息"""
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0
    gpu_usage_ratio: float = 0.0
    cpu_usage_gb: float = 0.0
    cpu_total_gb: float = 0.0
    cpu_usage_ratio: float = 0.0


class MemoryOptimizer:
    """内存优化器
    
    提供多种内存优化策略：
    1. 梯度检查点 (Gradient Checkpointing)
    2. 序列分块处理 (Sequence Chunking)
    3. 动态批次大小调整 (Dynamic Batch Sizing)
    4. 内存清理 (Memory Cleanup)
    5. 混合精度训练 (Mixed Precision)
    """
    
    def __init__(self, 
                 device: torch.device,
                 memory_threshold: float = 0.85,
                 cleanup_frequency: int = 10,
                 enable_gradient_checkpointing: bool = True,
                 enable_sequence_chunking: bool = True,
                 chunk_size: int = 5,
                 min_batch_size: int = 1):
        """初始化内存优化器
        
        Args:
            device: 计算设备
            memory_threshold: GPU内存使用率阈值 (0-1)
            cleanup_frequency: 内存清理频率 (每N步)
            enable_gradient_checkpointing: 是否启用梯度检查点
            enable_sequence_chunking: 是否启用序列分块
            chunk_size: 序列分块大小
            min_batch_size: 最小批次大小
        """
        self.device = device
        self.memory_threshold = memory_threshold
        self.cleanup_frequency = cleanup_frequency
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_sequence_chunking = enable_sequence_chunking
        self.chunk_size = chunk_size
        self.min_batch_size = min_batch_size
        
        self.step_count = 0
        self.oom_count = 0
        self.current_batch_size = None
        self.original_batch_size = None
        
        logger.info(f"内存优化器初始化:")
        logger.info(f"  设备: {device}")
        logger.info(f"  内存阈值: {memory_threshold:.1%}")
        logger.info(f"  梯度检查点: {enable_gradient_checkpointing}")
        logger.info(f"  序列分块: {enable_sequence_chunking} (chunk_size={chunk_size})")
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        stats = MemoryStats()
        
        # GPU内存统计
        if self.device.type == 'cuda' and torch.cuda.is_available():
            stats.gpu_allocated_gb = torch.cuda.memory_allocated(self.device) / 1024**3
            stats.gpu_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
            stats.gpu_total_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            stats.gpu_usage_ratio = stats.gpu_allocated_gb / stats.gpu_total_gb
        
        # CPU内存统计
        memory_info = psutil.virtual_memory()
        stats.cpu_usage_gb = memory_info.used / 1024**3
        stats.cpu_total_gb = memory_info.total / 1024**3
        stats.cpu_usage_ratio = memory_info.percent / 100.0
        
        return stats
    
    def log_memory_stats(self, prefix: str = ""):
        """记录内存统计信息"""
        stats = self.get_memory_stats()
        
        if self.device.type == 'cuda':
            logger.info(f"{prefix}GPU内存: {stats.gpu_allocated_gb:.2f}GB / {stats.gpu_total_gb:.2f}GB "
                       f"({stats.gpu_usage_ratio:.1%})")
        
        logger.info(f"{prefix}CPU内存: {stats.cpu_usage_gb:.2f}GB / {stats.cpu_total_gb:.2f}GB "
                   f"({stats.cpu_usage_ratio:.1%})")
    
    def cleanup_memory(self, force: bool = False):
        """清理内存"""
        if force or self.step_count % self.cleanup_frequency == 0:
            # 清理Python垃圾回收
            gc.collect()
            
            # 清理CUDA缓存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if force:
                logger.debug("强制内存清理完成")
    
    def check_memory_pressure(self) -> bool:
        """检查内存压力"""
        if self.device.type == 'cuda':
            stats = self.get_memory_stats()
            return stats.gpu_usage_ratio > self.memory_threshold
        return False
    
    def enable_gradient_checkpointing_for_model(self, model: nn.Module):
        """为模型启用梯度检查点"""
        if not self.enable_gradient_checkpointing:
            return
        
        def enable_checkpointing(module):
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
            elif hasattr(module, 'enable_gradient_checkpointing'):
                module.enable_gradient_checkpointing()
            else:
                # 对于自定义模块，尝试设置标志
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
        
        model.apply(enable_checkpointing)
        logger.info("已为模型启用梯度检查点")
    
    @contextmanager
    def memory_efficient_forward(self, model: nn.Module, enable_amp: bool = True):
        """内存高效的前向传播上下文管理器"""
        try:
            # 启用梯度检查点
            if self.enable_gradient_checkpointing:
                self.enable_gradient_checkpointing_for_model(model)
            
            # 使用混合精度
            if enable_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    yield
            else:
                yield
                
        finally:
            # 清理内存
            self.cleanup_memory()
    
    def chunk_sequence_forward(self, 
                              model: nn.Module, 
                              input_sequence: torch.Tensor,
                              target_sequence: torch.Tensor = None,
                              loss_fn: callable = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """分块处理长序列
        
        Args:
            model: AR模型
            input_sequence: 输入序列 [B, T, C, H, W]
            target_sequence: 目标序列 [B, T, C, H, W] (可选)
            loss_fn: 损失函数 (可选)
            
        Returns:
            outputs: 预测输出 [B, T, C, H, W]
            loss: 损失值 (如果提供了target_sequence和loss_fn)
        """
        if not self.enable_sequence_chunking:
            # 不使用分块，直接前向传播
            outputs = model(input_sequence)
            loss = None
            if target_sequence is not None and loss_fn is not None:
                loss = loss_fn(outputs, target_sequence)
            return outputs, loss
        
        batch_size, seq_len, channels, height, width = input_sequence.shape
        
        # 确保输入需要梯度（如果在训练模式）
        if model.training and not input_sequence.requires_grad:
            input_sequence = input_sequence.requires_grad_(True)
        
        if seq_len <= self.chunk_size:
            # 序列长度小于分块大小，直接处理
            outputs = model(input_sequence)
            loss = None
            if target_sequence is not None and loss_fn is not None:
                loss = loss_fn(outputs, target_sequence)
            return outputs, loss
        
        # 分块处理
        logger.debug(f"使用分块处理，序列长度: {seq_len}, 分块大小: {self.chunk_size}")
        
        outputs_list = []
        losses = []
        
        for start_idx in range(0, seq_len, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            
            # 获取当前分块
            chunk_input = input_sequence[:, start_idx:end_idx]
            
            # 前向传播
            with self.memory_efficient_forward(model):
                chunk_output = model(chunk_input)
                outputs_list.append(chunk_output)
                
                # 计算损失
                if target_sequence is not None and loss_fn is not None:
                    chunk_target = target_sequence[:, start_idx:end_idx]
                    chunk_loss = loss_fn(chunk_output, chunk_target)
                    losses.append(chunk_loss)
            
            # 清理中间结果
            del chunk_input, chunk_output
            if target_sequence is not None:
                del chunk_target
            
            # 定期清理内存
            if (start_idx // self.chunk_size) % 2 == 0:
                self.cleanup_memory()
        
        # 合并输出
        outputs = torch.cat(outputs_list, dim=1)
        
        # 计算平均损失
        loss = None
        if losses:
            loss = torch.stack(losses).mean()
        
        # 清理临时列表
        del outputs_list, losses
        
        return outputs, loss
    
    def adaptive_batch_size_training(self, 
                                   model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   optimizer: torch.optim.Optimizer,
                                   loss_fn: callable,
                                   max_retries: int = 3) -> Dict[str, Any]:
        """自适应批次大小训练
        
        在遇到OOM时自动减小批次大小并重试
        """
        if self.original_batch_size is None:
            self.original_batch_size = dataloader.batch_size
            self.current_batch_size = self.original_batch_size
        
        results = {
            'success': False,
            'batch_size_used': self.current_batch_size,
            'oom_occurred': False,
            'retries': 0
        }
        
        for retry in range(max_retries):
            try:
                # 尝试训练一个批次
                batch = next(iter(dataloader))
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 前向传播
                with self.memory_efficient_forward(model):
                    if self.enable_sequence_chunking:
                        outputs, loss = self.chunk_sequence_forward(
                            model, input_seq, target_seq, loss_fn
                        )
                    else:
                        outputs = model(input_seq)
                        loss = loss_fn(outputs, target_seq)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                results['success'] = True
                results['loss'] = loss.item()
                break
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_count += 1
                    results['oom_occurred'] = True
                    results['retries'] = retry + 1
                    
                    logger.warning(f"OOM错误 (第{retry+1}次): {e}")
                    
                    # 清理内存
                    self.cleanup_memory(force=True)
                    
                    # 减小批次大小
                    if self.current_batch_size > self.min_batch_size:
                        self.current_batch_size = max(
                            self.min_batch_size, 
                            self.current_batch_size // 2
                        )
                        logger.warning(f"减小批次大小至: {self.current_batch_size}")
                        
                        # 重新创建数据加载器（这里需要外部实现）
                        results['new_batch_size'] = self.current_batch_size
                    else:
                        logger.error("已达到最小批次大小，无法继续减小")
                        break
                else:
                    # 非OOM错误，直接抛出
                    raise e
        
        return results
    
    def get_optimal_chunk_size(self, 
                              model: nn.Module,
                              sample_input: torch.Tensor,
                              max_chunk_size: int = 20) -> int:
        """自动确定最优分块大小
        
        通过二分搜索找到不会导致OOM的最大分块大小
        """
        logger.info("正在确定最优分块大小...")
        
        batch_size, seq_len, channels, height, width = sample_input.shape
        
        # 二分搜索
        left, right = 1, min(max_chunk_size, seq_len)
        optimal_size = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # 测试当前分块大小
                test_input = sample_input[:, :mid].clone()
                
                with torch.no_grad():
                    with self.memory_efficient_forward(model):
                        _ = model(test_input)
                
                # 成功，尝试更大的分块
                optimal_size = mid
                left = mid + 1
                
                # 清理测试数据
                del test_input
                self.cleanup_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM，尝试更小的分块
                    right = mid - 1
                    self.cleanup_memory(force=True)
                else:
                    raise e
        
        logger.info(f"确定最优分块大小: {optimal_size}")
        return optimal_size
    
    def step(self):
        """步骤计数器"""
        self.step_count += 1
        
        # 定期清理内存
        if self.step_count % self.cleanup_frequency == 0:
            self.cleanup_memory()
            
            # 记录内存统计
            if self.step_count % (self.cleanup_frequency * 5) == 0:
                self.log_memory_stats(f"步骤 {self.step_count} - ")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        stats = self.get_memory_stats()
        
        return {
            'memory_stats': {
                'gpu_usage_gb': stats.gpu_allocated_gb,
                'gpu_usage_ratio': stats.gpu_usage_ratio,
                'cpu_usage_gb': stats.cpu_usage_gb,
                'cpu_usage_ratio': stats.cpu_usage_ratio
            },
            'optimization_stats': {
                'total_steps': self.step_count,
                'oom_count': self.oom_count,
                'current_batch_size': self.current_batch_size,
                'original_batch_size': self.original_batch_size,
                'chunk_size': self.chunk_size,
                'gradient_checkpointing': self.enable_gradient_checkpointing,
                'sequence_chunking': self.enable_sequence_chunking
            },
            'settings': {
                'memory_threshold': self.memory_threshold,
                'cleanup_frequency': self.cleanup_frequency,
                'min_batch_size': self.min_batch_size
            }
        }


class LongSequenceTrainer:
    """长序列训练器
    
    专门用于训练长时序AR模型的训练器
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 max_sequence_length: int = 50,
                 memory_optimizer_config: Dict[str, Any] = None):
        """初始化长序列训练器
        
        Args:
            model: AR模型
            optimizer: 优化器
            device: 计算设备
            max_sequence_length: 最大序列长度
            memory_optimizer_config: 内存优化器配置
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_sequence_length = max_sequence_length
        
        # 初始化内存优化器
        config = memory_optimizer_config or {}
        self.memory_optimizer = MemoryOptimizer(
            device=device,
            **config
        )
        
        # 启用梯度检查点
        self.memory_optimizer.enable_gradient_checkpointing_for_model(model)
        
        logger.info(f"长序列训练器初始化完成，最大序列长度: {max_sequence_length}")
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   loss_fn: callable,
                   gradient_clip_val: float = 1.0) -> Dict[str, Any]:
        """训练步骤"""
        input_seq = batch['input_sequence'].to(self.device)
        target_seq = batch['target_sequence'].to(self.device)
        
        # 使用内存优化的前向传播
        with self.memory_optimizer.memory_efficient_forward(self.model):
            if self.memory_optimizer.enable_sequence_chunking:
                outputs, loss = self.memory_optimizer.chunk_sequence_forward(
                    self.model, input_seq, target_seq, loss_fn
                )
            else:
                outputs = self.model(input_seq)
                loss = loss_fn(outputs, target_seq)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
        
        self.optimizer.step()
        
        # 更新内存优化器
        self.memory_optimizer.step()
        
        return {
            'loss': loss.item(),
            'memory_stats': self.memory_optimizer.get_memory_stats()
        }
    
    def validate_step(self, 
                     batch: Dict[str, torch.Tensor],
                     loss_fn: callable) -> Dict[str, Any]:
        """验证步骤"""
        input_seq = batch['input_sequence'].to(self.device)
        target_seq = batch['target_sequence'].to(self.device)
        
        with torch.no_grad():
            with self.memory_optimizer.memory_efficient_forward(self.model, enable_amp=False):
                if self.memory_optimizer.enable_sequence_chunking:
                    outputs, loss = self.memory_optimizer.chunk_sequence_forward(
                        self.model, input_seq, target_seq, loss_fn
                    )
                else:
                    outputs = self.model(input_seq)
                    loss = loss_fn(outputs, target_seq)
        
        return {
            'loss': loss.item(),
            'outputs': outputs,
            'targets': target_seq
        }