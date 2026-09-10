"""主训练脚本 (Train Entrypoint)

Sparse2Full 物理场重构主训练主控入口。
采用外观模式 (Facade) 编排数据流水线、模型构建、批量预处理与训练资产管理。
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import hydra
from omegaconf import DictConfig, OmegaConf

from models import create_model
from ops.losses import compute_total_loss, compute_ar_total_loss, compute_loss_weights_schedule
from utils.metrics import compute_all_metrics
from utils.logger import setup_logger

from training import (
    CurriculumScheduler,
    BatchProcessor,
    EngineBuilder,
    DataOrchestrator,
    TrainingArtifactManager
)

# 保持向后兼容性：允许外部直接从 train 导入 CurriculumScheduler
__all__ = ['Trainer', 'CurriculumScheduler', 'main']


class Trainer:
    """训练器类 (Facade)
    
    统一编排完整的训练流程，包括：
    - 数据与模型初始化
    - 批次流水线处理与物理守恒校验
    - 优化器与混合精度管理
    - Epoch 训练与验证控制流
    - 检查点持久化、样本可视化与论文交付包归档
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        
        # 设置随机种子与确定性
        self._set_random_seed(config.experiment.seed)
        
        # 创建输出目录与基础日志
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('train', self.output_dir / 'train.log')
        self.logger.info(f"Training started with config:\n{OmegaConf.to_yaml(config)}")
        
        # 1. 初始化核心解耦子系统
        self.batch_processor = BatchProcessor(config, self.logger)
        self.curriculum_scheduler = CurriculumScheduler(config)
        self.artifact_manager = TrainingArtifactManager(self.output_dir, config, self.logger)
        self.artifact_manager.save_env_fingerprint()
        
        # 2. 初始化数据模块
        self._init_data()
        
        # 3. 初始化模型
        self._init_model()
        
        # 4. 初始化优化器、调度器与混合精度
        self._init_optimizer()
        self._init_scheduler()
        self._init_amp()
        
        # 5. 训练状态跟踪
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics: Dict[str, Any] = {}
        self.early_stop_counter = 0
        self.train_time = 0.0
        self.val_time = 0.0

    @property
    def checkpoint_manager(self):
        """向后兼容属性：检查点管理器"""
        return self.artifact_manager.checkpoint_manager

    @property
    def tb_logger(self):
        """向后兼容属性：TensorBoard 日志器"""
        return self.artifact_manager.tb_logger

    @property
    def tb_writer(self):
        """向后兼容属性：TensorBoard Writer"""
        return self.artifact_manager.tb_writer

    @property
    def use_wandb(self):
        """向后兼容属性：WandB 启用状态"""
        return self.artifact_manager.use_wandb

    def _set_random_seed(self, seed: int) -> None:
        """设置随机种子与 CuDNN 确定性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        reproducibility = tr_cfg.get('reproducibility', {}) if hasattr(tr_cfg, 'get') else {}
        if reproducibility.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = reproducibility.get('benchmark', True)

    def _init_data(self) -> None:
        """初始化数据模块"""
        (
            self.data_module,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.norm_stats
        ) = DataOrchestrator.setup_data_module(self.config, self.logger)

        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        if tr_cfg.get('verify_data_consistency', True):
            self._verify_data_consistency()

    def _verify_data_consistency(self) -> None:
        """验证验证集数据一致性"""
        DataOrchestrator.verify_consistency(
            self.val_loader, self.config, self.output_dir, self.logger
        )

    def _init_model(self) -> None:
        """初始化模型与分布式包装"""
        self.logger.info("Initializing model...")
        model_params = {k: v for k, v in self.config.model.items() if k != 'name'}
        
        if 'params' in model_params:
            params = dict(model_params['params'])
            if 'kwargs' in params:
                kwargs = dict(params['kwargs'])
                del params['kwargs']
                kwargs.update(params)
                params = kwargs
            model_params = params
        
        # 确保 img_size 存在
        if 'img_size' not in model_params:
            if hasattr(self.config.data, 'img_size'):
                model_params['img_size'] = self.config.data.img_size
            elif hasattr(self.config.data, 'image_size'):
                model_params['img_size'] = self.config.data.image_size
            else:
                model_params['img_size'] = 512
        
        # 处理 ListConfig
        from omegaconf import ListConfig
        for key, value in model_params.items():
            if isinstance(value, ListConfig):
                model_params[key] = list(value)
        
        self.model = create_model(self.config.model.name, **model_params)
        self.model = self.model.to(self.device)
        
        model_info = self.model.get_model_info()
        self.logger.info(f"Model info: {model_info}")
        
        if hasattr(self.model, 'compute_flops'):
            flops = self.model.compute_flops()
            self.logger.info(f"Model FLOPs: {flops/1e9:.2f}G")
        
        batch_size = int(self.config.data.dataloader.batch_size) if hasattr(self.config.data, 'dataloader') else 16
        memory_info = self.model.get_memory_usage(batch_size)
        self.logger.info(f"Estimated memory usage: {memory_info}")
        
        # 多卡 DataParallel
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        dist_cfg = tr_cfg.get('distributed', {}) if hasattr(tr_cfg, 'get') else {}
        if torch.cuda.device_count() > 1 and dist_cfg.get('enabled', False):
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        # 梯度检查点
        if tr_cfg.get('gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                self.logger.info("Gradient checkpointing enabled")

    def _init_optimizer(self) -> None:
        """初始化优化器"""
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        self.optimizer = EngineBuilder.build_optimizer(self.model, tr_cfg, self.logger)

    def _init_scheduler(self) -> None:
        """初始化学习率调度器"""
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        self.scheduler, self.warmup_scheduler = EngineBuilder.build_scheduler(
            self.optimizer, tr_cfg, self.train_loader, self.logger
        )

    def _init_amp(self) -> None:
        """初始化混合精度"""
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        model_name = getattr(self.config.model, 'name', '')
        self.scaler, self.use_amp = EngineBuilder.build_amp_scaler(
            tr_cfg, model_name, self.logger
        )

    def _build_model_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """向后兼容代理：构建模型输入"""
        return self.batch_processor.build_model_input(batch, self.model)

    def _prepare_target(self, target: torch.Tensor, pred_shape: Tuple[int, ...]) -> torch.Tensor:
        """向后兼容代理：校验与处理目标物理场"""
        return self.batch_processor.prepare_target(target, pred_shape)

    def _save_checkpoint(self, val_results: Dict[str, float], is_best: bool) -> None:
        """向后兼容代理：保存检查点"""
        self.artifact_manager.save_checkpoint(
            epoch=self.current_epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            val_results=val_results,
            best_val_loss=self.best_val_loss,
            is_best=is_best,
            global_step=self.global_step
        )

    def _save_training_samples(self, epoch: int) -> None:
        """向后兼容代理：保存训练样本可视化"""
        try:
            val_batch = next(iter(self.val_loader))
            self.artifact_manager.save_training_samples(
                epoch=epoch,
                val_batch=val_batch,
                model=self.model,
                batch_processor=self.batch_processor,
                device=self.device
            )
        except Exception as e:
            self.logger.warning(f"Failed to fetch sample batch for visualization: {e}")

    def _log_epoch_results(self, train_results: Dict[str, float], val_results: Dict[str, float]) -> None:
        """向后兼容代理：记录 Epoch 统计"""
        lr = self.optimizer.param_groups[0]['lr']
        self.artifact_manager.log_epoch_results(
            epoch=self.current_epoch,
            train_results=train_results,
            val_results=val_results,
            lr=lr
        )

    def _cleanup(self) -> None:
        """向后兼容代理：清理资源并生成交付包"""
        self.artifact_manager.close(
            best_val_loss=self.best_val_loss,
            best_val_metrics=self.best_val_metrics,
            train_time=self.train_time,
            val_time=self.val_time,
            model=self.model
        )

    def train_epoch(self) -> Dict[str, float]:
        """执行单轮训练循环"""
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        epoch_metrics: Dict[str, float] = {}
        metrics_log_count = 0
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        self.logger.info(f"DEBUG: Starting epoch {self.current_epoch}, total batches: {num_batches}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            is_ar_model = bool(getattr(self.model, 'is_ar_model', False))
            
            with autocast(enabled=self.use_amp):
                if is_ar_model:
                    input_seq = batch.get('baseline_seq', batch['baseline'])
                    target_seq = batch.get('target_seq', batch['target'])
                    if input_seq.dim() == 4:
                        input_seq = input_seq.unsqueeze(1)
                    if target_seq.dim() == 4:
                        target_seq = target_seq.unsqueeze(1)
                    
                    tout = target_seq.shape[1]
                    pred_seq = self.model(input_seq, T_out=tout, teacher=target_seq)
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                else:
                    model_input = self._build_model_input(batch)
                    pred = self.model(model_input)
                    
                    # 计算课程损失权重
                    tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
                    total_epochs = int(getattr(tr_cfg, 'epochs', getattr(tr_cfg, 'max_epochs', 100)) or 100)
                    loss_weights = compute_loss_weights_schedule(self.current_epoch, total_epochs, self.config.loss)
                    
                    try:
                        config_with_weights = OmegaConf.copy(self.config)
                        for k, v in loss_weights.items():
                            config_with_weights.loss[k] = v
                    except Exception:
                        config_with_weights = self.config
                    
                    target = self._prepare_target(batch['target'], pred.shape)
                    losses = compute_total_loss(
                        pred_z=pred,
                        target_z=target,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=config_with_weights,
                        loss_weights_override=loss_weights if config_with_weights is self.config else None
                    )

            # 反向传播与梯度裁剪
            tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
            grad_clip = float(tr_cfg.get('grad_clip_norm', 0.0) if hasattr(tr_cfg, 'get') else 0.0)
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(losses['total_loss']).backward()
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
            
            # 学习率 warmup 步进
            sched_cfg = tr_cfg.get('scheduler', {}) if hasattr(tr_cfg, 'get') else {}
            warmup_epochs = int(sched_cfg.get('warmup_epochs', 0) if hasattr(sched_cfg, 'get') else 0)
            if self.warmup_scheduler is not None and self.current_epoch < warmup_epochs:
                self.warmup_scheduler.step()
            
            # 累积损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += float(value.item() if hasattr(value, 'item') else value)
            
            # 定期计算评估指标与日志记录
            log_interval = int(tr_cfg.get('log_interval', 50) if hasattr(tr_cfg, 'get') else 50) or 50
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    if is_ar_model:
                        metrics = compute_all_metrics(pred_seq[:, -1], target_seq[:, -1])
                    else:
                        metrics = compute_all_metrics(pred, batch['target'])
                    
                    for key, value in metrics.items():
                        if key not in epoch_metrics:
                            epoch_metrics[key] = 0.0
                        epoch_metrics[key] += float(value.item() if hasattr(value, 'item') else value)
                metrics_log_count += 1
                
                lr = self.optimizer.param_groups[0]['lr']
                batch_total_time = time.time() - batch_start_time
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {losses['total_loss'].item():.6f} | LR: {lr:.2e} | Time: {batch_total_time:.3f}s"
                )
                
                if self.tb_logger is not None and self.tb_logger.enabled:
                    step = self.current_epoch * num_batches + batch_idx
                    self.tb_logger.log_scalars({'loss': losses['total_loss'], 'lr': lr}, step, prefix='train')
                    sub_losses = {k: v for k, v in losses.items() if k != 'total_loss'}
                    self.tb_logger.log_scalars(sub_losses, step, prefix='train')
            
            self.global_step += 1

        # 计算 Epoch 平均值
        for key in epoch_losses:
            epoch_losses[key] /= max(1, num_batches)
        
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, metrics_log_count)
        
        self.train_time += time.time() - start_time
        return {**epoch_losses, **epoch_metrics}

    def validate_epoch(self) -> Dict[str, float]:
        """执行验证集评估循环"""
        self.model.eval()
        epoch_losses: Dict[str, float] = {}
        epoch_metrics: Dict[str, float] = {}
        num_batches = len(self.val_loader)
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                is_ar_model = bool(getattr(self.model, 'is_ar_model', False))
                
                if is_ar_model:
                    input_seq = batch.get('baseline_seq', batch['baseline'])
                    target_seq = batch.get('target_seq', batch['target'])
                    if input_seq.dim() == 4:
                        input_seq = input_seq.unsqueeze(1)
                    if target_seq.dim() == 4:
                        target_seq = target_seq.unsqueeze(1)
                    
                    tout = target_seq.shape[1]
                    pred_seq = self.model(input_seq, T_out=tout)
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                    metrics = compute_all_metrics(pred_seq[:, -1], target_seq[:, -1])
                else:
                    model_input = self._build_model_input(batch)
                    pred = self.model(model_input)
                    target = self._prepare_target(batch['target'], pred.shape)
                    losses = compute_total_loss(
                        pred_z=pred,
                        target_z=target,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                    metrics = compute_all_metrics(pred, target)
                
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    val_scalar = value.mean().item() if hasattr(value, 'mean') else (value.item() if hasattr(value, 'item') else float(value))
                    epoch_losses[key] += val_scalar

                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    val_scalar = value.mean().item() if hasattr(value, 'mean') else (value.item() if hasattr(value, 'item') else float(value))
                    epoch_metrics[key] += val_scalar

        for key in epoch_losses:
            epoch_losses[key] /= max(1, num_batches)
        
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_batches)
        
        self.val_time += time.time() - start_time
        return {**epoch_losses, **epoch_metrics}

    def train(self) -> None:
        """执行完整训练生命周期循环"""
        self.logger.info("Starting training...")
        tr_cfg = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        total_epochs = int(getattr(tr_cfg, 'epochs', getattr(tr_cfg, 'max_epochs', 100)) or 100)
        
        try:
            for epoch in range(total_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                train_results = self.train_epoch()
                val_results = self.validate_epoch()
                
                # 学习率调度
                if self.scheduler is not None:
                    import torch.optim.lr_scheduler as lrs
                    if isinstance(self.scheduler, lrs.ReduceLROnPlateau):
                        self.scheduler.step(val_results['total_loss'])
                    else:
                        sched_cfg = tr_cfg.get('scheduler', {}) if hasattr(tr_cfg, 'get') else {}
                        warmup_epochs = int(sched_cfg.get('warmup_epochs', 0) if hasattr(sched_cfg, 'get') else 0)
                        if self.warmup_scheduler is None or epoch >= warmup_epochs:
                            self.scheduler.step()
                
                self._log_epoch_results(train_results, val_results)
                
                # 检查点与早停
                is_best = val_results['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_results['total_loss']
                    self.best_val_metrics = val_results.copy()
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                self._save_checkpoint(val_results, is_best)
                
                es_cfg = tr_cfg.get('early_stopping', {}) if hasattr(tr_cfg, 'get') else {}
                if bool(es_cfg.get('enabled', False)) and self.early_stop_counter >= int(es_cfg.get('patience', 10)):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # 周期性绘制样本
                save_interval = int(tr_cfg.get('save_interval', 20) if hasattr(tr_cfg, 'get') else 20)
                if epoch % max(1, save_interval) == 0:
                    self._save_training_samples(epoch)
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training encountered unhandled exception: {e}")
            raise
        finally:
            try:
                self._verify_data_consistency()
            except Exception as e:
                self.logger.warning(f"Final consistency check skipped: {e}")
            self._cleanup()
        
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Total training time: {self.train_time:.2f}s | Total validation time: {self.val_time:.2f}s")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """Hydra CLI 主入口"""
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
