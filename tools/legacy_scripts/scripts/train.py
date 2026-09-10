#!/usr/bin/env python3
"""
高性能PDEBench训练脚本主程序
基于技术方案实现针对AMD EPYC 9654 CPU和NVIDIA L40 GPU优化的训练系统
"""

import os
import sys
import yaml
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # 允许在未安装 OmegaConf 的环境下运行

# 导入自定义模块
from src.optimizers.hardware_profiler import HardwareProfiler
from src.optimizers.numa_manager import NUMAMemoryManager
from src.optimizers.gpu_optimizer import GPUOptimizer
from src.optimizers.mixed_precision_trainer import MixedPrecisionTrainer
from src.optimizers.distributed_trainer import DistributedTrainingManager, DistributedConfig
from src.data.optimized_pipeline import OptimizedDataPipeline
from src.models.swin_temporal_nar import SwinTemporalNAR, SwinTemporalConfig, create_swin_temporal_nar
from src.monitoring.performance_monitor import PerformanceMonitor
from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import setup_logging

# 设置日志
logger = logging.getLogger(__name__)

class PDEBenchTrainer:
    """
    PDEBench高性能训练器
    集成所有优化组件的统一训练接口
    """
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = None
        self.hardware_profiler = None
        self.numa_manager = None
        self.gpu_optimizer = None
        self.mixed_precision_trainer = None
        self.distributed_manager = None
        self.performance_monitor = None
        self.data_pipeline = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        logger.info(f"PDEBench训练器初始化，配置文件: {config_path}")
    
    def setup_environment(self):
        """设置训练环境"""
        logger.info("开始设置训练环境...")
        
        # 1. 加载配置
        self.config = load_config(self.config_path)
        logger.info("配置文件加载完成")
        
        # 2. 设置日志
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(
            log_file=log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log",
            log_level=self.config['logging']['level']
        )
        
        # 3. 设置随机种子
        self._set_random_seeds(self.config['training']['seed'])
        
        # 4. 硬件检测与优化
        self._setup_hardware_optimization()
        
        # 5. 设置TensorBoard
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(log_dir=log_dir / 'tensorboard')
        
        logger.info("训练环境设置完成")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"随机种子设置完成: {seed}")
    
    def _setup_hardware_optimization(self):
        """设置硬件优化"""
        logger.info("开始硬件优化设置...")
        
        # 硬件检测
        self.hardware_profiler = HardwareProfiler()
        hardware_config = self.hardware_profiler.detect_hardware()
        
        # NUMA优化
        if self.config['hardware']['numa_optimization']:
            self.numa_manager = NUMAMemoryManager()
            self.numa_manager.initialize()
            logger.info("NUMA优化已启用")
        
        # GPU优化
        if torch.cuda.is_available():
            self.gpu_optimizer = GPUOptimizer()
            logger.info("GPU优化已启用")
        
        # 混合精度训练
        if self.config['training']['mixed_precision']:
            mp_config = self.config['mixed_precision']
            self.mixed_precision_trainer = MixedPrecisionTrainer(mp_config)
            logger.info("混合精度训练已启用")
        
        logger.info(f"硬件优化设置完成: {hardware_config}")
    
    def setup_data_pipeline(self):
        """设置数据流水线"""
        logger.info("开始设置数据流水线...")
        
        data_config = self.config['data']
        
        # 创建数据流水线
        self.data_pipeline = OptimizedDataPipeline(data_config)
        
        # 加载数据集
        train_dataset = self._create_dataset('train')
        val_dataset = self._create_dataset('validation')
        
        # 创建数据加载器
        self.train_loader = self.data_pipeline.create_dataloader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = self.data_pipeline.create_dataloader(
            val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"数据流水线设置完成: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
    
    def _create_dataset(self, split: str) -> Dataset:
        """创建数据集"""
        # 这里需要根据实际的PDEBench数据集格式进行实现
        # 暂时返回一个模拟数据集用于测试
        from src.data.pdebench_dataset import PDEBenchDataset
        return PDEBenchDataset(
            data_dir=self.config['data']['data_dir'],
            split=split,
            transform=None
        )
    
    def setup_model(self):
        """设置模型"""
        logger.info("开始设置模型...")
        
        model_config = self.config['model']
        
        # 创建模型配置
        swin_config = SwinTemporalConfig(
            img_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            in_channels=model_config['in_channels'],
            embed_dim=model_config['embed_dim'],
            depths=model_config['depths'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            future_steps=model_config['future_steps'],
            prediction_type=model_config['prediction_type'],
            temporal_config=model_config.get('temporal_config', {})
        )
        
        # 创建模型
        self.model = create_swin_temporal_nar(swin_config)
        
        # 应用GPU优化
        if self.gpu_optimizer:
            self.model = self.gpu_optimizer.optimize_model(self.model)
        
        logger.info(f"模型设置完成: {self.model.get_model_info()}")
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        logger.info("开始设置优化器和调度器...")
        
        optim_config = self.config['optimizer']
        scheduler_config = self.config['scheduler']
        
        # 创建优化器
        if optim_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optim_config['lr'],
                betas=(optim_config['beta1'], optim_config['beta2']),
                weight_decay=optim_config['weight_decay'],
                eps=optim_config['eps']
            )
        elif optim_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['lr'],
                betas=(optim_config['beta1'], optim_config['beta2']),
                weight_decay=optim_config['weight_decay'],
                eps=optim_config['eps']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optim_config['type']}")
        
        # 创建学习率调度器
        if scheduler_config['type'] == 'onecycle':
            total_steps = self.config['training']['epochs'] * len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config['max_lr'],
                total_steps=total_steps,
                pct_start=scheduler_config['pct_start'],
                anneal_strategy=scheduler_config['anneal_strategy'],
                div_factor=scheduler_config['div_factor'],
                final_div_factor=scheduler_config['final_div_factor']
            )
        elif scheduler_config['type'] == 'cosine':
            # 支持按步数设置 T_max，以匹配 OneCycle 的行为
            # 如果未显式指定，默认使用 total_steps
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                total_steps = self.config['training']['epochs'] * len(self.train_loader)
            else:
                # 兜底：无法获取 train_loader 时退化为按 epoch 近似
                total_steps = max(1, self.config['training']['epochs'])

            t_max_unit = scheduler_config.get('t_max_unit', 'steps')
            if t_max_unit == 'steps':
                t_max_value = total_steps
            else:
                t_max_value = scheduler_config.get('T_max', self.config['training']['epochs'])

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max_value,
                eta_min=scheduler_config.get('eta_min', 0.0)
            )
        else:
            logger.warning(f"不支持的调度器类型: {scheduler_config['type']}，使用恒定学习率")
            self.scheduler = None
        
        logger.info(f"优化器和调度器设置完成: {optim_config['type']}, {scheduler_config['type']}")
    
    def setup_distributed_training(self):
        """设置分布式训练"""
        if self.config['distributed']['enabled']:
            logger.info("开始设置分布式训练...")
            
            dist_config = DistributedConfig(
                strategy=self.config['distributed']['strategy'],
                world_size=self.config['distributed']['world_size'],
                backend=self.config['distributed']['backend'],
                mixed_precision=self.config['training']['mixed_precision'],
                batch_size_per_gpu=self.config['training']['batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps']
            )
            
            self.distributed_manager = DistributedTrainingManager(dist_config)
            
            # 设置模型、优化器和调度器
            self.model, self.optimizer, self.scheduler = self.distributed_manager.setup_model(
                self.model, self.optimizer, self.scheduler
            )
            
            logger.info("分布式训练设置完成")
        else:
            # 单机训练
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            logger.info(f"单机训练模式，设备: {device}")
    
    def setup_performance_monitor(self):
        """设置性能监控"""
        logger.info("开始设置性能监控...")
        
        monitor_config = self.config['monitoring']
        
        self.performance_monitor = PerformanceMonitor(
            log_dir=self.config['logging']['log_dir'],
            log_frequency=monitor_config['log_frequency'],
            profile_memory=monitor_config['profile_memory'],
            profile_compute=monitor_config['profile_compute'],
            profile_communication=monitor_config['profile_communication']
        )
        
        logger.info("性能监控设置完成")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        # 步级统计（用于更真实的聚合）
        agg_stats = {
            'samples_per_sec_sum': 0.0,
            'gpu_util_sum': 0.0,
            'mem_gb_sum': 0.0,
            'log_count': 0
        }
        
        # 性能监控开始
        self.performance_monitor.start_epoch(epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 性能监控开始步骤
            self.performance_monitor.start_step()
            
            # 准备数据（统一输入/目标构建）
            inputs, targets = self._prepare_batch(batch)
            
            # 前向传播
            if self.config['training']['mixed_precision'] and self.mixed_precision_trainer:
                loss = self._mixed_precision_forward_backward(inputs, targets)
            else:
                loss = self._standard_forward_backward(inputs, targets)
            
            # 更新统计
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            self.global_step += 1
            
            # 性能监控结束步骤
            step_stats = self.performance_monitor.end_step()
            
            # 日志记录
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                self._log_training_progress(epoch, batch_idx, loss.item(), step_stats)
                # 聚合真实记录次数
                agg_stats['samples_per_sec_sum'] += float(step_stats.get('samples_per_sec', 0.0))
                agg_stats['gpu_util_sum'] += float(step_stats.get('gpu_utilization', 0.0))
                agg_stats['mem_gb_sum'] += float(step_stats.get('memory_usage_gb', 0.0))
                agg_stats['log_count'] += 1
            
            # TensorBoard记录
            if self.writer and batch_idx % self.config['logging']['tensorboard_frequency'] == 0:
                self._log_tensorboard('train', {
                    'loss': loss.item(),
                    'learning_rate': self._get_current_lr(),
                    **step_stats
                })
        
        # 性能监控结束epoch
        epoch_stats = self.performance_monitor.end_epoch()
        
        avg_loss = total_loss / total_samples
        
        # 使用真实记录次数进行更稳健的聚合（若不可用，回退到 epoch_stats）
        if agg_stats['log_count'] > 0:
            avg_samples_per_sec = agg_stats['samples_per_sec_sum'] / agg_stats['log_count']
            avg_gpu_util = agg_stats['gpu_util_sum'] / agg_stats['log_count']
            avg_mem_gb = agg_stats['mem_gb_sum'] / agg_stats['log_count']
        else:
            avg_samples_per_sec = epoch_stats.get('samples_per_sec', 0.0)
            avg_gpu_util = epoch_stats.get('gpu_utilization', 0.0)
            avg_mem_gb = epoch_stats.get('memory_usage_gb', 0.0)

        return {
            'loss': avg_loss,
            'epoch_time': epoch_stats['epoch_time'],
            'samples_per_sec': avg_samples_per_sec,
            'gpu_utilization': avg_gpu_util,
            'memory_usage_gb': avg_mem_gb
        }
    
    def _prepare_batch(self, batch):
        """准备批次数据（统一的输入/目标构建，满足 forward(x[B,C_in,H,W])→y[B,C_out,H,W]）"""
        device = next(self.model.parameters()).device

        # 字典批次：期望包含 baseline/coords/mask/target
        if isinstance(batch, dict):
            inputs = self._build_model_input(batch)
            targets = self._build_targets(batch, like=inputs)
        elif isinstance(batch, (list, tuple)):
            # 传统 (inputs, targets)
            inputs, targets = batch
        else:
            # 单张输入，可能是自监督
            inputs = batch
            targets = None

        # 移动到设备
        inputs = inputs.to(device, non_blocking=True)
        if targets is not None:
            targets = targets.to(device, non_blocking=True)

        # 健壮性校验：通道数与模型期望一致
        expected_in = self._get_expected_in_channels()
        if inputs.ndim != 4:
            raise ValueError(f"模型输入必须为4维[B,C,H,W]，但得到 {inputs.shape}")
        if inputs.size(1) != expected_in:
            raise ValueError(
                f"输入通道 {inputs.size(1)} 与模型期望 {expected_in} 不一致"
            )

        return inputs, targets

    def _get_expected_in_channels(self) -> int:
        """获取模型期望的输入通道数"""
        try:
            return int(self.config['model']['in_channels'])
        except Exception:
            # 尝试从模型属性获取
            return getattr(self.model, 'in_channels', None) or 1

    def _get_expected_out_channels(self) -> int:
        """获取模型期望的输出通道数"""
        return int(self.config['model'].get('out_channels', 1))

    def _select_time_step(self, tensor: torch.Tensor, time_index: Optional[int] = None) -> torch.Tensor:
        """选择 5D 张量的时间步 [B,T,C,H,W] → [B,C,H,W]"""
        if tensor.ndim == 5:
            t_idx = -1 if time_index is None else time_index
            return tensor[:, t_idx]
        return tensor

    def _match_spatial(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """将 src 的空间尺寸匹配到 ref（双线性插值，适用于坐标/掩码/目标）"""
        if src.shape[-2:] != ref.shape[-2:]:
            src = torch.nn.functional.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return src

    def _adjust_channels(self, x: torch.Tensor, required_channels: int) -> torch.Tensor:
        """裁剪或零填充通道到 required_channels"""
        c = x.size(1)
        if c == required_channels:
            return x
        elif c > required_channels:
            return x[:, :required_channels]
        else:
            pad = torch.zeros(x.size(0), required_channels - c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)

    def _build_model_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """构建模型输入 [baseline, coords, mask] 按通道拼接到期望的 in_channels"""
        expected_in = self._get_expected_in_channels()

        baseline = batch.get('baseline') or batch.get('observation') or batch.get('inputs')
        coords = batch.get('coords')
        mask = batch.get('mask')

        if baseline is None:
            raise KeyError("批次缺少 'baseline' 或 'observation' 作为输入的基础")

        # 时间选择与维度规范化
        baseline = self._select_time_step(baseline, time_index=self.config['data'].get('input_time_index'))
        if baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)  # [1,C,H,W]
        if baseline.ndim != 4:
            raise ValueError(f"baseline 维度应为4或5，当前 {baseline.shape}")

        parts = []
        B, _, H, W = baseline.shape
        ref = baseline

        # 坐标与掩码（可选），匹配空间维
        coord_ch = 0
        mask_ch = 0
        if coords is not None:
            if coords.ndim == 3:
                coords = coords.unsqueeze(0)
            coords = self._select_time_step(coords, time_index=None)
            coords = self._match_spatial(coords, ref)
            coord_ch = coords.size(1)
            parts.append(coords)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            mask = self._select_time_step(mask, time_index=None)
            mask = self._match_spatial(mask, ref)
            mask_ch = mask.size(1)
            parts.append(mask)

        # baseline 通道调整以满足总通道 = expected_in
        required_baseline_ch = max(1, expected_in - (coord_ch + mask_ch))
        baseline = self._adjust_channels(baseline, required_baseline_ch)

        # 拼接 [baseline, coords?, mask?]
        input_parts = [baseline]
        # 维持顺序 [baseline, coords, mask]
        if coords is not None:
            input_parts.append(coords)
        if mask is not None:
            input_parts.append(mask)

        inputs = torch.cat(input_parts, dim=1)

        # 最终健壮性校验
        if inputs.size(1) != expected_in:
            raise ValueError(
                f"构建的输入通道 {inputs.size(1)} 与期望 {expected_in} 不一致"
            )

        return inputs

    def _build_targets(self, batch: Dict[str, torch.Tensor], like: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """构建目标 [B,C_out,H,W]，与输入空间对齐，按需选择时间步"""
        target = batch.get('target') or batch.get('targets')
        if target is None:
            return None

        target = self._select_time_step(target, time_index=self.config['data'].get('target_time_index'))
        # 维度标准化
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if target.ndim != 4:
            raise ValueError(f"target 维度应为4或5，当前 {target.shape}")

        # 与 inputs 空间对齐
        if like is not None:
            target = self._match_spatial(target, like)

        # 可选通道调整到期望输出通道
        expected_out = self._get_expected_out_channels()
        target = self._adjust_channels(target, expected_out)

        return target
    
    def _mixed_precision_forward_backward(self, inputs, targets):
        """混合精度前向传播和反向传播"""
        try:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
        except Exception as e:
            # 提供更有用的错误上下文
            raise RuntimeError(
                f"前向失败: inputs={tuple(inputs.shape)} targets={(tuple(targets.shape) if targets is not None else None)}\n{e}"
            )
        
        # 反向传播
        if self.distributed_manager:
            # 分布式训练
            loss = loss / self.config['training']['gradient_accumulation_steps']
            self.distributed_manager.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.distributed_manager.scaler.step(self.optimizer)
                self.distributed_manager.scaler.update()
                self.optimizer.zero_grad()
        else:
            # 单机训练
            self.mixed_precision_trainer.backward(loss, self.optimizer)
        
        return loss
    
    def _standard_forward_backward(self, inputs, targets):
        """标准前向传播和反向传播"""
        try:
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets)
        except Exception as e:
            raise RuntimeError(
                f"前向失败: inputs={tuple(inputs.shape)} targets={(tuple(targets.shape) if targets is not None else None)}\n{e}"
            )
        
        # 反向传播
        if self.distributed_manager:
            # 分布式训练
            loss = loss / self.config['training']['gradient_accumulation_steps']
            loss.backward()
            
            if (self.global_step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            # 单机训练
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss
    
    def _compute_loss(self, outputs, targets):
        """计算损失"""
        if targets is None:
            # 自监督损失
            return torch.mean(outputs ** 2)
        
        # 监督损失
        criterion = nn.MSELoss()
        return criterion(outputs, targets)
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._prepare_batch(batch)
                
                if self.config['training']['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    def _get_current_lr(self):
        """获取当前学习率"""
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def _log_training_progress(self, epoch, batch_idx, loss, step_stats):
        """记录训练进度"""
        progress = (batch_idx + 1) / len(self.train_loader) * 100
        lr = self._get_current_lr()
        
        logger.info(
            f"Epoch [{epoch+1}/{self.config['training']['epochs']}] "
            f"[{progress:.1f}%] "
            f"Loss: {loss:.6f} "
            f"LR: {lr:.2e} "
            f"Samples/sec: {step_stats.get('samples_per_sec', 0):.1f} "
            f"GPU: {step_stats.get('gpu_utilization', 0):.1f}% "
            f"Mem: {step_stats.get('memory_usage_gb', 0):.1f}GB"
        )
    
    def _log_tensorboard(self, phase, metrics):
        """记录到TensorBoard"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{key}", value, self.global_step)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self._safe_config_snapshot()
        }
        
        # 保存最新检查点
        checkpoint_path = Path(self.config['training']['checkpoint_dir']) / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = Path(self.config['training']['checkpoint_dir']) / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存: {best_path}")

    def _safe_config_snapshot(self):
        """安全快照配置，避免对 Hydra DictConfig 进行不安全拷贝"""
        cfg = self.config
        try:
            if OmegaConf is not None and hasattr(cfg, 'get') and not isinstance(cfg, dict):
                # 可能是 DictConfig
                return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
        # 退化：深拷贝字典/对象
        try:
            return deepcopy(cfg)
        except Exception:
            return cfg
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"检查点已加载: {checkpoint_path}")
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    pass  # OneCycleLR在每个step更新
                else:
                    self.scheduler.step()
            
            # 记录epoch统计
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1} 完成: "
                f"训练损失: {train_metrics['loss']:.6f}, "
                f"验证损失: {val_metrics['loss']:.6f}, "
                f"用时: {epoch_time:.1f}s, "
                f"GPU利用率: {train_metrics['gpu_utilization']:.1f}%, "
                f"内存使用: {train_metrics['memory_usage_gb']:.1f}GB"
            )
            
            # TensorBoard记录
            if self.writer:
                self._log_tensorboard('epoch', {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': self._get_current_lr(),
                    'epoch_time': epoch_time
                })
            
            # 检查最佳模型
            is_best = val_metrics['loss'] < self.best_metric
            if is_best:
                self.best_metric = val_metrics['loss']
            
            # 保存检查点
            if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, is_best)
            
            # 重置计时器
            start_time = time.time()
        
        logger.info("训练完成！")
        
        # 保存最终模型
        self.save_checkpoint(self.config['training']['epochs'] - 1)
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")
        
        # 关闭分布式训练
        if self.distributed_manager:
            self.distributed_manager.cleanup()
        
        # 关闭NUMA管理器
        if self.numa_manager:
            self.numa_manager.cleanup()
        
        # 关闭性能监控
        if self.performance_monitor:
            self.performance_monitor.cleanup()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        logger.info("资源清理完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDEBench高性能训练脚本")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--validate', action='store_true', help='仅验证模式')
    parser.add_argument('--benchmark', action='store_true', help='性能基准测试模式')
    
    args = parser.parse_args()
    
    try:
        # 创建训练器
        trainer = PDEBenchTrainer(args.config)
        
        # 设置环境
        trainer.setup_environment()
        
        # 性能基准测试模式
        if args.benchmark:
            logger.info("性能基准测试模式")
            trainer.setup_data_pipeline()
            trainer.setup_model()
            trainer.setup_optimizer_and_scheduler()
            trainer.setup_performance_monitor()
            # 运行基准测试
            from src.benchmark.benchmark_runner import run_benchmark
            run_benchmark(trainer)
            return
        
        # 验证模式
        if args.validate:
            logger.info("验证模式")
            trainer.setup_data_pipeline()
            trainer.setup_model()
            trainer.setup_distributed_training()
            
            if args.resume:
                trainer.load_checkpoint(args.resume)
            
            val_metrics = trainer.validate()
            logger.info(f"验证结果: {val_metrics}")
            return
        
        # 正常训练模式
        logger.info("训练模式")
        trainer.setup_data_pipeline()
        trainer.setup_model()
        trainer.setup_optimizer_and_scheduler()
        trainer.setup_distributed_training()
        trainer.setup_performance_monitor()
        
        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
        # 清理资源
        trainer.cleanup()
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()