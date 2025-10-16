"""训练脚本

PDEBench稀疏观测重建系统主训练入口
支持Hydra配置管理，包含完整训练循环
严格按照开发手册的黄金法则实现
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置UTF-8编码，解决Windows下的编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    # Windows系统下设置控制台编码
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # 如果都失败，继续执行

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets import PDEBenchDataModule
from models.base import create_model
from ops.losses import compute_total_loss, compute_loss_weights_schedule
from ops.degradation import verify_degradation_consistency
from utils.metrics import compute_all_metrics
from utils.checkpoint import CheckpointManager
from utils.logger import setup_logger
from utils.visualization import PDEBenchVisualizer


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.curriculum_config = config.training.get('curriculum_learning', {})
        
    def get_current_task_params(self, epoch: int) -> Dict[str, Any]:
        """获取当前epoch的任务参数"""
        params = {}
        
        if not self.curriculum_config.enabled:
            return params
        
        # SR任务调度
        if self.curriculum_config.sr_schedule.enabled:
            for stage in self.curriculum_config.sr_schedule.stages:
                if stage.epochs[0] <= epoch < stage.epochs[1]:
                    params['scale_factor'] = stage.scale_factor
                    break
        
        # Crop任务调度
        if self.curriculum_config.crop_schedule.enabled:
            for stage in self.curriculum_config.crop_schedule.stages:
                if stage.epochs[0] <= epoch < stage.epochs[1]:
                    params['crop_ratio'] = stage.crop_ratio
                    break
        
        return params
    
    def get_loss_weights(self, epoch: int, total_epochs: int, base_weights: Dict[str, float]) -> Dict[str, float]:
        """获取当前epoch的损失权重"""
        if not self.curriculum_config.enabled or not self.curriculum_config.loss_weight_schedule.enabled:
            return base_weights
        
        weights = base_weights.copy()
        progress = epoch / total_epochs
        
        # DC损失权重调度
        if 'data_consistency' in self.curriculum_config.loss_weight_schedule:
            dc_config = self.curriculum_config.loss_weight_schedule.data_consistency
            if dc_config.schedule_type == 'linear':
                weight = dc_config.start_weight + (dc_config.end_weight - dc_config.start_weight) * progress
                weights['data_consistency'] = weight
        
        # 频谱损失权重调度
        if 'spectral' in self.curriculum_config.loss_weight_schedule:
            spec_config = self.curriculum_config.loss_weight_schedule.spectral
            if spec_config.schedule_type == 'peak':
                peak_ratio = spec_config.peak_epoch_ratio
                if progress <= peak_ratio:
                    # 上升阶段
                    weight = spec_config.start_weight + (spec_config.peak_weight - spec_config.start_weight) * (progress / peak_ratio)
                else:
                    # 下降阶段
                    weight = spec_config.peak_weight + (spec_config.end_weight - spec_config.peak_weight) * ((progress - peak_ratio) / (1 - peak_ratio))
                weights['spectral'] = weight
        
        return weights


class Trainer:
    """训练器类
    
    负责完整的训练流程，包括：
    - 模型训练和验证
    - 损失计算和优化
    - 检查点管理
    - 日志记录和可视化
    - 数据一致性验证
    - 课程学习和AMP支持
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        
        # 设置随机种子
        self._set_random_seed(config.experiment.seed)
        
        # 创建输出目录
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger('train', self.output_dir / 'train.log')
        self.logger.info(f"Training started with config:\n{OmegaConf.to_yaml(config)}")
        
        # 保存配置快照
        config_path = self.output_dir / 'config_merged.yaml'
        OmegaConf.save(config, config_path)
        self.logger.info(f"Config saved to {config_path}")
        
        # 初始化组件
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_amp()
        self._init_curriculum()
        self._init_logging()
        self._init_checkpoint_manager()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.early_stop_counter = 0
        
        # 性能统计
        self.train_time = 0
        self.val_time = 0
        
    def _set_random_seed(self, seed: int) -> None:
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # 确保确定性（可能影响性能）
        if self.config.training.get('reproducibility', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self.config.training.get('reproducibility', {}).get('benchmark', True)
    
    def _init_data(self) -> None:
        """初始化数据模块"""
        self.logger.info("Initializing data module...")
        
        # 添加调试信息
        print(f"DEBUG: config.data = {self.config.data}")
        print(f"DEBUG: config.data type = {type(self.config.data)}")
        print(f"DEBUG: config.data keys = {list(self.config.data.keys())}")
        print(f"DEBUG: config.data.data_path = {self.config.data.get('data_path', 'NOT_FOUND')}")
        
        self.data_module = PDEBenchDataModule(self.config.data)
        self.data_module.setup()
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
        
        # 获取归一化统计量
        self.norm_stats = self.data_module.get_norm_stats()
        
        self.logger.info(f"Data loaded: train={len(self.train_loader)}, "
                        f"val={len(self.val_loader)}, test={len(self.test_loader)}")
        
        # 数据一致性验证
        if self.config.training.get('verify_data_consistency', True):
            self._verify_data_consistency()
    
    def _init_model(self) -> None:
        """初始化模型"""
        self.logger.info("Initializing model...")
        
        # 提取模型参数，排除name字段
        model_params = {k: v for k, v in self.config.model.items() if k != 'name'}
        
        # 如果有params字段，则使用params中的参数
        if 'params' in model_params:
            params = dict(model_params['params'])  # 转换为普通字典
            # 合并kwargs到主参数中，但主参数优先级更高
            if 'kwargs' in params:
                kwargs = dict(params['kwargs'])  # 转换为普通字典
                del params['kwargs']  # 删除kwargs键
                # 先更新kwargs，再更新主参数（主参数覆盖kwargs）
                kwargs.update(params)
                params = kwargs
            model_params = params
        
        # 处理Hydra配置中的ListConfig类型
        from omegaconf import ListConfig
        for key, value in model_params.items():
            if isinstance(value, ListConfig):
                model_params[key] = list(value)
        
        self.model = create_model(
            model_name=self.config.model.name,
            **model_params
        )
        self.model = self.model.to(self.device)
        
        # 模型信息
        model_info = self.model.get_model_info()
        self.logger.info(f"Model info: {model_info}")
        
        # 计算FLOPs
        if hasattr(self.model, 'compute_flops'):
            flops = self.model.compute_flops()
            self.logger.info(f"Model FLOPs: {flops/1e9:.2f}G")
        
        # 显存使用量
        memory_info = self.model.get_memory_usage(self.config.data.dataloader.batch_size)
        self.logger.info(f"Estimated memory usage: {memory_info}")
        
        # 分布式训练（如果需要）
        if torch.cuda.device_count() > 1 and self.config.training.get('distributed', {}).get('enabled', False):
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        # 梯度检查点（节省显存）
        if self.config.training.get('gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                self.logger.info("Gradient checkpointing enabled")
    
    def _init_optimizer(self) -> None:
        """初始化优化器"""
        optimizer_config = self.config.training.optimizer
        
        # 检查是否有name字段，如果没有则使用_target_
        if hasattr(optimizer_config, 'name'):
            optimizer_name = optimizer_config.name.lower()
        elif hasattr(optimizer_config, '_target_'):
            optimizer_name = optimizer_config._target_.split('.')[-1].lower()
        else:
            optimizer_name = 'adamw'  # 默认使用AdamW
        
        if optimizer_name == 'adamw':
            if hasattr(optimizer_config, 'params'):
                params = optimizer_config.params
            else:
                params = optimizer_config
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=params.get('lr', 0.001),
                weight_decay=params.get('weight_decay', 0.0001),
                betas=params.get('betas', (0.9, 0.999)),
                eps=params.get('eps', 1e-8)
            )
        elif optimizer_name == 'adam':
            if hasattr(optimizer_config, 'params'):
                params = optimizer_config.params
            else:
                params = optimizer_config
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=params.get('lr', 0.001),
                weight_decay=params.get('weight_decay', 0),
                betas=params.get('betas', (0.9, 0.999)),
                eps=params.get('eps', 1e-8)
            )
        elif optimizer_name == 'sgd':
            if hasattr(optimizer_config, 'params'):
                params = optimizer_config.params
            else:
                params = optimizer_config
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=params.get('lr', 0.001),
                weight_decay=params.get('weight_decay', 0),
                momentum=params.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        self.logger.info(f"Optimizer: {self.optimizer}")
    
    def _init_scheduler(self) -> None:
        """初始化学习率调度器"""
        scheduler_config = self.config.training.scheduler
        
        # 检查scheduler配置是否为None或空
        if scheduler_config is None or (hasattr(scheduler_config, 'name') and scheduler_config.name is None):
            self.scheduler = None
            self.warmup_scheduler = None
            return
        
        # 获取scheduler名称
        scheduler_name = getattr(scheduler_config, 'name', None)
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config.training.epochs),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.factor,
                patience=scheduler_config.patience
            )
        else:
            self.scheduler = None
        
        # Warmup调度器
        if scheduler_config.get('warmup_epochs', 0) > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=scheduler_config.get('warmup_start_factor', 0.1),
                total_iters=scheduler_config.warmup_epochs
            )
        else:
            self.warmup_scheduler = None
        
        self.logger.info(f"Scheduler: {self.scheduler}")
    
    def _init_amp(self) -> None:
        """初始化混合精度训练"""
        self.use_amp = self.config.training.get('use_amp', False)
        
        # 检查模型是否支持AMP（FNO相关模型有复数操作问题）
        model_name = self.config.model.name.lower()
        fno_models = ['fno2d', 'hybrid', 'ufno_unet', 'u-fno']
        
        if self.use_amp and model_name in fno_models:
            self.logger.warning(f"Model {model_name} has complex number operations, disabling AMP")
            self.use_amp = False
        
        if self.use_amp:
            amp_config = self.config.training.get('amp', {})
            self.scaler = GradScaler(
                init_scale=amp_config.get('init_scale', 65536.0),
                growth_factor=amp_config.get('growth_factor', 2.0),
                backoff_factor=amp_config.get('backoff_factor', 0.5),
                growth_interval=amp_config.get('growth_interval', 2000)
            )
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if model_name in fno_models:
                self.logger.info(f"AMP disabled for {model_name} (complex operations compatibility)")
    
    def _init_curriculum(self) -> None:
        """初始化课程学习"""
        self.curriculum_scheduler = CurriculumScheduler(self.config)
        
        if self.config.training.get('curriculum_learning', {}).get('enabled', False):
            self.logger.info("Curriculum learning enabled")
    
    def _init_logging(self) -> None:
        """初始化日志记录"""
        # TensorBoard
        if self.config.logging.get('use_tensorboard', True):
            self.tb_writer = SummaryWriter(self.output_dir / 'tensorboard')
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.logging.get('use_wandb', False):
            wandb.init(
                project=self.config.logging.get('wandb_project', 'pdebench-sparse2full'),
                name=self.config.experiment.name,
                config=OmegaConf.to_container(self.config, resolve=True),
                dir=str(self.output_dir)
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def _init_checkpoint_manager(self) -> None:
        """初始化检查点管理器"""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / 'checkpoints',
            max_checkpoints=self.config.training.get('checkpoint', {}).get('max_keep', 3),
            save_best=self.config.training.get('checkpoint', {}).get('save_best', True)
        )
    
    def _verify_data_consistency(self) -> None:
        """验证数据一致性"""
        self.logger.info("Verifying data consistency...")
        
        # 从验证集中采样一些数据进行验证
        sample_batch = next(iter(self.val_loader))
        
        try:
            # 提取必要的数据
            target = sample_batch['target']
            observation = sample_batch['observation']
            task_params = sample_batch.get('task_params', {
                'task': 'SR',  # 默认为SR任务
                'scale': 4,    # 默认缩放因子
                'sigma': 1.0,
                'kernel_size': 5,
                'boundary': 'mirror'
            })
            
            consistency_result = verify_degradation_consistency(
                target, observation, task_params
            )
            consistency_error = consistency_result['mse']
            
            tolerance = self.config.training.get('consistency_tolerance', 1e-8)
            if consistency_error < tolerance:
                self.logger.info(f"Data consistency verified: MSE = {consistency_error:.2e}")
            else:
                self.logger.warning(f"⚠ Data consistency check failed: MSE = {consistency_error:.2e}")
                if consistency_error > 1e-6:
                    raise ValueError(f"Data consistency error too large: {consistency_error}")
        
        except Exception as e:
            self.logger.error(f"Data consistency verification failed: {e}")
            raise

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {}
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 模型预测
                pred = self.model(batch['baseline'])
                
                # 计算损失权重（课程学习）
                loss_weights = compute_loss_weights_schedule(
                    self.current_epoch,
                    self.config.training.epochs,
            self.config.loss
                )
                
                # 更新配置中的损失权重
                config_with_weights = self.config.copy()
                config_with_weights.loss.update(loss_weights)
                
                # 计算损失
                losses = compute_total_loss(
                    pred_z=pred,
                    target_z=batch['target'],
                    obs_data=batch,
                    norm_stats=self.norm_stats,
                    config=config_with_weights
                )
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(losses['total_loss']).backward()
                
                # 梯度裁剪
                if self.config.training.get('grad_clip_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                
                # 梯度裁剪
                if self.config.training.get('grad_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # 学习率调度（warmup）
            if self.warmup_scheduler is not None and self.current_epoch < self.config.training.scheduler.get('warmup_epochs', 0):
                self.warmup_scheduler.step()
            
            # 累积损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()
            
            # 计算指标（每隔一定步数）
            if batch_idx % self.config.training.log_interval == 0:
                with torch.no_grad():
                    metrics = compute_all_metrics(pred, batch['target'])
                    for key, value in metrics.items():
                        if key not in epoch_metrics:
                            epoch_metrics[key] = 0
                        epoch_metrics[key] += value
            
            # 日志记录
            if batch_idx % self.config.training.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {losses['total_loss'].item():.6f} "
                    f"LR: {lr:.2e}"
                )
                
                # TensorBoard日志
                if self.tb_writer is not None:
                    step = self.current_epoch * num_batches + batch_idx
                    self.tb_writer.add_scalar('train/loss', losses['total_loss'].item(), step)
                    self.tb_writer.add_scalar('train/lr', lr, step)
                    for key, value in losses.items():
                        if key != 'total_loss':
                            self.tb_writer.add_scalar(f'train/{key}', value.item(), step)
            
            self.global_step += 1
        
        # 计算epoch平均值
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_metrics:
            epoch_metrics[key] /= (num_batches // self.config.training.log_interval + 1)
        
        self.train_time += time.time() - start_time
        
        return {**epoch_losses, **epoch_metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        epoch_losses = {}
        epoch_metrics = {}
        num_batches = len(self.val_loader)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 移动数据到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                pred = self.model(batch['baseline'])
                
                # 计算损失
                losses = compute_total_loss(
                    pred_z=pred,
                    target_z=batch['target'],
                    obs_data=batch,
                    norm_stats=self.norm_stats,
                    config=self.config
                )
                
                # 计算指标
                metrics = compute_all_metrics(pred, batch['target'])
                
                # 累积损失和指标
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()
                
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0
                    epoch_metrics[key] += value
        
        # 计算平均值
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        self.val_time += time.time() - start_time
        
        return {**epoch_losses, **epoch_metrics}
    
    def train(self) -> None:
        """主训练循环"""
        self.logger.info("Starting training...")
        
        # AMP Scaler已在_init_amp中初始化，这里不需要重复创建
        
        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch
                
                # 训练
                train_results = self.train_epoch()
                
                # 验证
                val_results = self.validate_epoch()
                
                # 学习率调度
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_results['total_loss'])
                    else:
                        if self.warmup_scheduler is None or epoch >= self.config.training.scheduler.get('warmup_epochs', 0):
                            self.scheduler.step()
                
                # 日志记录
                self._log_epoch_results(train_results, val_results)
                
                # 保存检查点
                is_best = val_results['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_results['total_loss']
                    self.best_val_metrics = val_results.copy()
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                self._save_checkpoint(val_results, is_best)
                
                # 早停
                if (self.config.training.get('early_stopping', {}).get('enabled', False) and
                self.early_stop_counter >= self.config.training.get('early_stopping', {}).get('patience', 10)):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # 保存训练样本（可视化）
                if epoch % self.config.training.get('save_interval', 20) == 0:
                    self._save_training_samples(epoch)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation metrics: {self.best_val_metrics}")
        self.logger.info(f"Total training time: {self.train_time:.2f}s")
        self.logger.info(f"Total validation time: {self.val_time:.2f}s")
    
    def _log_epoch_results(self, train_results: Dict[str, float], val_results: Dict[str, float]) -> None:
        """记录epoch结果"""
        # 控制台日志
        # 确保所有值都是标量
        train_loss = train_results['total_loss']
        val_loss = val_results['total_loss']
        val_rel_l2 = val_results.get('rel_l2', 0)
        
        # 如果是张量，取平均值
        if hasattr(train_loss, 'item'):
            train_loss = train_loss.item()
        if hasattr(val_loss, 'item'):
            val_loss = val_loss.item()
        if hasattr(val_rel_l2, 'mean'):
            val_rel_l2 = val_rel_l2.mean().item()
        elif hasattr(val_rel_l2, 'item'):
            try:
                val_rel_l2 = val_rel_l2.item()
            except RuntimeError:
                # 如果张量有多个元素，取平均值
                val_rel_l2 = val_rel_l2.mean().item()
        
        self.logger.info(
            f"Epoch {self.current_epoch:3d} - "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"Val Rel-L2: {val_rel_l2:.6f}"
        )
        
        # TensorBoard日志
        if self.tb_writer is not None:
            for key, value in train_results.items():
                # 确保value是标量
                if hasattr(value, 'mean'):
                    value = value.mean().item()
                elif hasattr(value, 'item'):
                    try:
                        value = value.item()
                    except RuntimeError:
                        value = value.mean().item()
                self.tb_writer.add_scalar(f'epoch_train/{key}', value, self.current_epoch)
            for key, value in val_results.items():
                # 确保value是标量
                if hasattr(value, 'mean'):
                    value = value.mean().item()
                elif hasattr(value, 'item'):
                    try:
                        value = value.item()
                    except RuntimeError:
                        value = value.mean().item()
                self.tb_writer.add_scalar(f'epoch_val/{key}', value, self.current_epoch)
            
            # 学习率
            lr = self.optimizer.param_groups[0]['lr']
            self.tb_writer.add_scalar('epoch_train/lr', lr, self.current_epoch)
        
        # Weights & Biases日志
        if self.use_wandb:
            log_dict = {}
            for key, value in train_results.items():
                log_dict[f'train/{key}'] = value
            for key, value in val_results.items():
                log_dict[f'val/{key}'] = value
            log_dict['epoch'] = self.current_epoch
            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
    def _save_checkpoint(self, val_results: Dict[str, float], is_best: bool) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'val_results': val_results,
            'config': self.config,
            'global_step': self.global_step
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        self.checkpoint_manager.save_checkpoint(checkpoint, is_best, self.current_epoch)
    
    def _save_training_samples(self, epoch: int) -> None:
        """保存训练样本可视化"""
        if not self.config.training.get('save_samples', True):
            return
        
        if epoch % self.config.training.get('plot_interval', 50) != 0:
            return
        
        try:
            # 获取一个验证批次
            val_batch = next(iter(self.val_loader))
            
            with torch.no_grad():
                pred = self.model(val_batch['baseline'].to(self.device))
            
            # 创建可视化器 - 使用统一的可视化工具
            save_dir = self.output_dir / 'samples' / f'epoch_{epoch:04d}'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            visualizer = PDEBenchVisualizer(str(save_dir))
            
            # 保存可视化
            max_samples = self.config.training.get('max_samples', 4)
            for i in range(min(max_samples, pred.shape[0])):
                visualizer.plot_field_comparison(
                    gt=val_batch['target'][i:i+1],
                    pred=pred[i:i+1].cpu(),
                    baseline=val_batch['baseline'][i:i+1],
                    save_name=f'sample_{i}_epoch_{epoch}'
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to save training samples: {e}")
    
    def _cleanup(self) -> None:
        """清理资源"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """主函数"""
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()