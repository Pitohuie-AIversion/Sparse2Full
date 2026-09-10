"""训练引擎构建器模块 (Engine Builder)

负责优化器 (Optimizer)、学习率调度器 (Scheduler/Warmup) 与混合精度 (AMP GradScaler) 的标准化初始化。
"""

import logging
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig


class EngineBuilder:
    """优化器、调度器与混合精度构建工厂"""

    @staticmethod
    def build_optimizer(
        model: nn.Module, 
        training_config: Any, 
        logger: Optional[logging.Logger] = None
    ) -> optim.Optimizer:
        """初始化优化器 (AdamW / Adam / SGD)"""
        logger = logger or logging.getLogger(__name__)
        optimizer_config = getattr(training_config, 'optimizer', {})
        
        # 检查是否有 name 字段，否则尝试 _target_
        if hasattr(optimizer_config, 'name') and optimizer_config.name is not None:
            optimizer_name = str(optimizer_config.name).lower()
        elif hasattr(optimizer_config, '_target_'):
            optimizer_name = str(optimizer_config._target_).split('.')[-1].lower()
        elif isinstance(optimizer_config, dict) and 'name' in optimizer_config:
            optimizer_name = str(optimizer_config['name']).lower()
        else:
            optimizer_name = 'adamw'
        
        # 提取参数
        params = getattr(optimizer_config, 'params', optimizer_config if isinstance(optimizer_config, dict) else {})
        if not isinstance(params, dict):
            params = dict(params) if hasattr(params, 'items') else {}

        lr = float(params.get('lr', 1e-3))
        weight_decay = float(params.get('weight_decay', 1e-4 if optimizer_name == 'adamw' else 0.0))
        betas = tuple(params.get('betas', (0.9, 0.999)))
        eps = float(params.get('eps', 1e-8))

        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_name == 'sgd':
            momentum = float(params.get('momentum', 0.9))
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer initialized: {optimizer.__class__.__name__} with lr={lr}, weight_decay={weight_decay}")
        return optimizer

    @staticmethod
    def build_scheduler(
        optimizer: optim.Optimizer,
        training_config: Any,
        train_loader: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Optional[optim.lr_scheduler._LRScheduler], Optional[optim.lr_scheduler.LinearLR]]:
        """初始化学习率调度器与 Warmup 调度器"""
        logger = logger or logging.getLogger(__name__)
        scheduler_config = getattr(training_config, 'scheduler', None)
        
        if scheduler_config is None or (hasattr(scheduler_config, 'name') and scheduler_config.name is None):
            return None, None
            
        scheduler_name = getattr(scheduler_config, 'name', None)
        if scheduler_name is None and isinstance(scheduler_config, dict):
            scheduler_name = scheduler_config.get('name')
            
        if scheduler_name is not None:
            scheduler_name = str(scheduler_name).lower()
        
        scheduler = None
        warmup_scheduler = None
        
        if scheduler_name in ('cosine', 'cosineannealinglr', 'cosine_warmup'):
            total_epochs = int(getattr(training_config, 'epochs', getattr(training_config, 'max_epochs', 100)) or 100)
            steps_per_epoch = len(train_loader) if train_loader is not None else 1
            total_steps = total_epochs * max(1, steps_per_epoch)

            params = getattr(scheduler_config, 'params', scheduler_config if isinstance(scheduler_config, dict) else {})
            eta_min = float(params.get('eta_min', 0.0) if hasattr(params, 'get') else getattr(params, 'eta_min', 0.0))

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=eta_min
            )
        elif scheduler_name in ('step', 'steplr'):
            params = getattr(scheduler_config, 'params', scheduler_config if isinstance(scheduler_config, dict) else {})
            step_size = int(params.get('step_size', 30) if hasattr(params, 'get') else getattr(params, 'step_size', 30))
            gamma = float(params.get('gamma', 0.1) if hasattr(params, 'get') else getattr(params, 'gamma', 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name in ('plateau', 'reducelronplateau'):
            params = getattr(scheduler_config, 'params', scheduler_config if isinstance(scheduler_config, dict) else {})
            mode = str(params.get('mode', 'min') if hasattr(params, 'get') else getattr(params, 'mode', 'min'))
            factor = float(params.get('factor', 0.5) if hasattr(params, 'get') else getattr(params, 'factor', 0.5))
            patience = int(params.get('patience', 10) if hasattr(params, 'get') else getattr(params, 'patience', 10))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        
        # Warmup 调度器
        sched_params = getattr(scheduler_config, 'params', scheduler_config if isinstance(scheduler_config, dict) else {})
        warmup_epochs = 0
        if hasattr(sched_params, 'get'):
            warmup_epochs = sched_params.get('warmup_epochs', 0)
        elif hasattr(sched_params, 'warmup_epochs'):
            warmup_epochs = sched_params.warmup_epochs
            
        if warmup_epochs and int(warmup_epochs) > 0:
            start_factor = float(sched_params.get('warmup_start_factor', 0.1) if hasattr(sched_params, 'get') else 0.1)
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=int(warmup_epochs)
            )
            logger.info(f"Warmup LinearLR enabled for {warmup_epochs} epochs (start_factor={start_factor})")
        
        logger.info(f"Scheduler initialized: {scheduler.__class__.__name__ if scheduler else 'None'}")
        return scheduler, warmup_scheduler

    @staticmethod
    def build_amp_scaler(
        training_config: Any,
        model_name: str = "",
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Optional[GradScaler], bool]:
        """初始化混合精度 GradScaler，防范复数运算模型崩溃"""
        logger = logger or logging.getLogger(__name__)
        use_amp = bool(getattr(training_config, 'use_amp', False))
        
        model_name_lower = str(model_name).lower()
        fno_models = ['fno2d', 'hybrid', 'ufno_unet', 'u-fno']
        if use_amp and any(m in model_name_lower for m in fno_models):
            logger.warning(f"Model {model_name} utilizes complex operations, automatically disabling AMP")
            use_amp = False
        
        if use_amp:
            amp_config = getattr(training_config, 'amp', {}) if hasattr(training_config, 'amp') else {}
            scaler = GradScaler(
                init_scale=float(amp_config.get('init_scale', 65536.0) if hasattr(amp_config, 'get') else 65536.0),
                growth_factor=float(amp_config.get('growth_factor', 2.0) if hasattr(amp_config, 'get') else 2.0),
                backoff_factor=float(amp_config.get('backoff_factor', 0.5) if hasattr(amp_config, 'get') else 0.5),
                growth_interval=int(amp_config.get('growth_interval', 2000) if hasattr(amp_config, 'get') else 2000)
            )
            logger.info("AMP mixed precision enabled with GradScaler")
            return scaler, True
        else:
            return None, False
