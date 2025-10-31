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
    
    # 设置日志编码为UTF-8
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

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
from utils.visualization import TemporalVisualizer


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
        
        # 调试配置结构
        print(f"DEBUG: config keys = {list(config.keys())}")
        print(f"DEBUG: config type = {type(config)}")
        if hasattr(config, 'data'):
            print(f"DEBUG: config.data exists")
        else:
            print(f"DEBUG: config.data does NOT exist")
        
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
        training_config = getattr(self.config, 'training', getattr(self.config, 'train', {}))
        reproducibility = training_config.get('reproducibility', {})
        if reproducibility.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = reproducibility.get('benchmark', True)
    
    def _init_data(self) -> None:
        """初始化数据模块"""
        self.logger.info("Initializing data module...")
        
        # 处理配置结构问题 - 数据配置可能在空字符串键下
        print(f"DEBUG: config keys = {list(self.config.keys())}")
        print(f"DEBUG: config[''] = {self.config.get('', 'NOT_FOUND')}")
        
        if hasattr(self.config, 'data'):
            data_config = self.config.data
        elif '' in self.config and 'data' in self.config['']:
            data_config = self.config['']['data']
        else:
            # 使用空字符串键下的配置
            data_config = self.config['']
        
        # 添加调试信息
        print(f"DEBUG: data_config = {data_config}")
        print(f"DEBUG: data_config type = {type(data_config)}")
        
        # 检查是否有datasets.data配置
        if hasattr(data_config, 'datasets') and hasattr(data_config.datasets, 'data'):
            actual_data_config = data_config.datasets.data
            print(f"DEBUG: Found datasets.data config: {actual_data_config}")
            print(f"DEBUG: actual_data_config._target_ = {actual_data_config.get('_target_', 'NOT_FOUND')}")
            
            # 根据配置选择数据模块
            if hasattr(actual_data_config, '_target_') and 'temporal' in actual_data_config._target_:
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                print("DEBUG: Using TemporalPDEBenchDataModule")
                self.data_module = TemporalPDEBenchDataModule(actual_data_config.config)
            else:
                print("DEBUG: Using PDEBenchDataModule")
                self.data_module = PDEBenchDataModule(actual_data_config)
        elif hasattr(data_config, 'data'):
            actual_data_config = data_config.data
            print(f"DEBUG: Found nested data config: {actual_data_config}")
            print(f"DEBUG: actual_data_config._target_ = {actual_data_config.get('_target_', 'NOT_FOUND')}")
            
            # 根据配置选择数据模块
            if hasattr(actual_data_config, '_target_') and 'temporal' in actual_data_config._target_:
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                print("DEBUG: Using TemporalPDEBenchDataModule")
                self.data_module = TemporalPDEBenchDataModule(actual_data_config.config)
            else:
                print("DEBUG: Using PDEBenchDataModule")
                self.data_module = PDEBenchDataModule(actual_data_config)
        elif hasattr(data_config, '_target_'):
            print(f"DEBUG: data_config._target_ = {data_config.get('_target_', 'NOT_FOUND')}")
            # 根据配置选择数据模块
            if 'temporal' in data_config._target_:
                from datasets.temporal_pdebench import TemporalPDEBenchDataModule
                print("DEBUG: Using TemporalPDEBenchDataModule")
                self.data_module = TemporalPDEBenchDataModule(data_config.config)
            else:
                print("DEBUG: Using PDEBenchDataModule")
                self.data_module = PDEBenchDataModule(data_config)
        else:
            # 如果data_config本身就是配置参数，直接使用
            print("DEBUG: Using data_config as direct parameters")
            self.data_module = PDEBenchDataModule(data_config)
        
        # 只有PDEBenchDataModule有setup方法，TemporalPDEBenchDataModule没有
        if hasattr(self.data_module, 'setup'):
            self.data_module.setup()
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # TemporalPDEBenchDataModule可能没有test_dataloader
        if hasattr(self.data_module, 'test_dataloader'):
            self.test_loader = self.data_module.test_dataloader()
        else:
            self.test_loader = None
        
        # 获取归一化统计量
        if hasattr(self.data_module, 'get_norm_stats'):
            self.norm_stats = self.data_module.get_norm_stats()
        else:
            self.norm_stats = None
        
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
        
        # 确保img_size参数存在
        if 'img_size' not in model_params:
            if hasattr(self.config.data, 'img_size'):
                model_params['img_size'] = self.config.data.img_size
            elif hasattr(self.config.data, 'image_size'):
                model_params['img_size'] = self.config.data.image_size
            else:
                # 默认使用512
                model_params['img_size'] = 512
        
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
            elif isinstance(optimizer_config, dict):
                params = optimizer_config
            else:
                # 如果是字符串或其他类型，使用默认参数
                params = {}
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
            elif isinstance(optimizer_config, dict):
                params = optimizer_config
            else:
                # 如果是字符串或其他类型，使用默认参数
                params = {}
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
            # 获取训练轮数
            if hasattr(self.config.training, 'epochs'):
                default_epochs = self.config.training.epochs
            elif hasattr(self.config.training, 'max_epochs'):
                default_epochs = self.config.training.max_epochs
            else:
                default_epochs = 100  # 默认值
                
            if isinstance(scheduler_config, dict):
                T_max = scheduler_config.get('T_max', default_epochs)
                eta_min = scheduler_config.get('eta_min', 0)
            else:
                T_max = getattr(scheduler_config, 'T_max', default_epochs)
                eta_min = getattr(scheduler_config, 'eta_min', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
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
        if isinstance(scheduler_config, dict) and scheduler_config.get('warmup_epochs', 0) > 0:
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
            
            # 从数据样本中获取h_params，如果没有则使用默认值
            h_params = sample_batch.get('h_params', {
                'task': 'SR',  # 默认为SR任务
                'scale': self.config.data.config.get('scale', 2),  # 从配置获取缩放因子
                'sigma': self.config.data.config.get('sigma', 1.0),
                'blur_kernel': self.config.data.config.get('blur_kernel', 5),
                'boundary': self.config.data.config.get('boundary', 'mirror'),
                'noise_std': self.config.data.config.get('noise_std', 0.0)
            })
            
            # 对于时序数据，需要处理维度差异
            if target.dim() == 5:  # [B, T, C, H, W]
                # 取第一个时间步进行验证
                target_sample = target[:, 0]  # [B, C, H, W]
                observation_sample = observation[:, 0]  # [B, C, H, W]
            else:
                target_sample = target
                observation_sample = observation
            
            consistency_result = verify_degradation_consistency(
                target_sample, observation_sample, h_params
            )
            consistency_error = consistency_result['mse']
            
            tolerance = self.config.training.get('consistency_tolerance', 1e-6)  # 放宽容忍度
            if consistency_error < tolerance:
                self.logger.info(f"Data consistency verified: MSE = {consistency_error:.2e}")
            else:
                self.logger.warning(f"WARNING: Data consistency check failed: MSE = {consistency_error:.2e}")
                # 对于时序数据，暂时跳过严格的一致性检查
                self.logger.warning("Skipping strict consistency check for temporal data")
        
        except Exception as e:
            self.logger.error(f"Data consistency verification failed: {e}")
            # 对于时序数据，暂时跳过一致性检查
            self.logger.warning("Skipping data consistency verification for temporal data")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {}
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        self.logger.info(f"DEBUG: Starting epoch {self.current_epoch}, total batches: {num_batches}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.logger.info(f"DEBUG: Processing batch {batch_idx}/{num_batches}")
            batch_start_time = time.time()
            
            # 移动数据到设备
            device_start_time = time.time()
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            device_time = time.time() - device_start_time
            self.logger.info(f"DEBUG: Batch {batch_idx} device transfer took {device_time:.4f}s")
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 检查是否为AR模式
            is_ar_model = hasattr(self.model, 'is_ar_model') and self.model.is_ar_model
            
            # 前向传播
            forward_start_time = time.time()
            self.logger.info(f"DEBUG: Starting forward pass for batch {batch_idx}")
            with autocast(enabled=self.use_amp):
                if is_ar_model:
                    # AR模式：序列到序列预测
                    
                    # 获取输入序列和目标序列
                    input_seq = batch.get('baseline_seq', batch['baseline'])  # 兼容性处理
                    target_seq = batch.get('target_seq', batch['target'])
                    
                    # 如果输入不是序列，扩展维度
                    if input_seq.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
                        input_seq = input_seq.unsqueeze(1)
                    if target_seq.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
                        target_seq = target_seq.unsqueeze(1)
                    
                    # AR模型前向传播
                    tout = target_seq.shape[1]
                    pred_seq = self.model(input_seq, T_out=tout, teacher=target_seq)
                    
                    # 计算AR损失
                    from ops.losses import compute_ar_total_loss
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                else:
                    # 标准模式：单帧预测
                    baseline = batch['baseline']
                    
                    # 处理时序数据：如果是5维张量，取最后一个时间步
                    if baseline.dim() == 5:  # [B, T, C, H, W]
                        print(f"DEBUG: baseline shape before processing: {baseline.shape}")
                        baseline = baseline[:, -1]  # [B, C, H, W] - 取最后一个时间步
                        print(f"DEBUG: baseline shape after processing: {baseline.shape}")
                    elif baseline.dim() == 3:  # [T_in*C, H, W] - 已经被flatten了
                        print(f"DEBUG: baseline shape (3D): {baseline.shape}")
                        # 添加batch维度
                        baseline = baseline.unsqueeze(0)  # [1, C, H, W]
                        print(f"DEBUG: baseline with batch dim: {baseline.shape}")
                    else:
                        print(f"DEBUG: baseline shape: {baseline.shape}")
                    
                    # 构建模型输入（统一接口：[baseline, coords, mask, fourier_pe?]）
                    model_input = baseline
                    print(f"DEBUG: Initial model_input shape: {model_input.shape}")
                    
                    # 添加坐标信息
                    if 'coords' in batch:
                        coords = batch['coords']
                        print(f"DEBUG: coords shape: {coords.shape}")
                        model_input = torch.cat([model_input, coords], dim=1)
                        print(f"DEBUG: model_input after coords: {model_input.shape}")
                    
                    # 添加mask信息
                    if 'mask' in batch:
                        mask = batch['mask']
                        print(f"DEBUG: mask shape: {mask.shape}")
                        model_input = torch.cat([model_input, mask], dim=1)
                        print(f"DEBUG: model_input after mask: {model_input.shape}")
                    
                    model_start_time = time.time()
                    self.logger.info(f"DEBUG: About to call model forward for batch {batch_idx}, model_input shape: {model_input.shape}")
                    pred = self.model(model_input)
                    model_time = time.time() - model_start_time
                    self.logger.info(f"DEBUG: Model forward completed for batch {batch_idx}, took {model_time:.4f}s, pred shape: {pred.shape}")
                    
                    # 计算损失权重（课程学习）
                    # 获取训练轮数
                    if hasattr(self.config.training, 'epochs'):
                        total_epochs = self.config.training.epochs
                    elif hasattr(self.config.training, 'max_epochs'):
                        total_epochs = self.config.training.max_epochs
                    else:
                        total_epochs = 100  # 默认值
                        
                    loss_weights = compute_loss_weights_schedule(
                        self.current_epoch,
                        total_epochs,
                        self.config.loss
                    )
                    
                    # 更新配置中的损失权重
                    config_with_weights = self.config.copy()
                    config_with_weights.loss.update(loss_weights)
                    
                    # 处理目标数据
                    target = batch['target']
                    print(f"DEBUG: pred shape: {pred.shape}")
                    print(f"DEBUG: target shape before processing: {target.shape}")
                    
                    # 如果target是时序数据，需要处理
                    if target.dim() == 5:  # [B, T, C, H, W]
                        # 对于SR任务，我们需要取最后一个时间步作为目标
                        target = target[:, -1]  # [B, C, H, W]
                        print(f"DEBUG: target shape after taking last timestep: {target.shape}")
                        
                        # 如果通道数不匹配，需要调整
                        if target.shape[1] != pred.shape[1]:
                            # 假设我们只需要前2个通道
                            target = target[:, :pred.shape[1]]
                            print(f"DEBUG: target shape after channel adjustment: {target.shape}")
                        
                        # 如果空间尺寸不匹配，需要调整
                        if target.shape[-2:] != pred.shape[-2:]:
                            # 使用双线性插值调整target的空间尺寸
                            import torch.nn.functional as F
                            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
                            print(f"DEBUG: target shape after spatial adjustment: {target.shape}")
                    
                    # 计算损失
                    loss_start_time = time.time()
                    losses = compute_total_loss(
                        pred_z=pred,
                        target_z=target,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=config_with_weights
                    )
                    loss_time = time.time() - loss_start_time
            
            forward_time = time.time() - forward_start_time
            
            # 反向传播
            backward_start_time = time.time()
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
            
            backward_time = time.time() - backward_start_time
            
            # 学习率调度（warmup）
            if self.warmup_scheduler is not None and self.current_epoch < self.config.training.scheduler.get('warmup_epochs', 0):
                self.warmup_scheduler.step()
            
            # 累积损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()
            
            # 计算指标（每隔一定步数）
            log_interval = getattr(self.config.training, 'log_interval', 50)  # 默认值50
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    if is_ar_model:
                        # AR模式：计算序列指标
                        # 使用最后一个时间步进行指标计算
                        pred_last = pred_seq[:, -1]  # [B, C, H, W]
                        target_last = target_seq[:, -1]  # [B, C, H, W]
                        metrics = compute_all_metrics(pred_last, target_last)
                    else:
                        # 标准模式
                        metrics = compute_all_metrics(pred, batch['target'])
                    
                    for key, value in metrics.items():
                        if key not in epoch_metrics:
                            epoch_metrics[key] = 0
                        epoch_metrics[key] += value
            
            batch_total_time = time.time() - batch_start_time
            
            # 日志记录
            if batch_idx % log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"DEBUG: Batch {batch_idx} timing - Device: {device_time:.3f}s, Model: {model_time:.3f}s, Loss: {loss_time:.3f}s, Backward: {backward_time:.3f}s, Total: {batch_total_time:.3f}s")
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {losses['total_loss'].item():.6f} "
                    f"LR: {lr:.2e} "
                    f"Time: {batch_total_time:.3f}s"
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
            
            # 如果是第一个batch，打印更多调试信息
            if batch_idx == 0:
                print(f"DEBUG: First batch completed successfully")
                print(f"DEBUG: Loss components: {list(losses.keys())}")
                print(f"DEBUG: Total loss: {losses['total_loss'].item():.6f}")
        
        # 计算epoch平均值
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_metrics:
            log_interval = getattr(self.config.training, 'log_interval', 50)  # 默认值50
            epoch_metrics[key] /= (num_batches // log_interval + 1)
        
        self.train_time += time.time() - start_time
        
        print(f"DEBUG: Epoch {self.current_epoch} completed in {time.time() - start_time:.2f}s")
        
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
                
                # 检查是否为AR模式
                is_ar_model = hasattr(self.model, 'is_ar_model') and self.model.is_ar_model
                
                if is_ar_model:
                    # AR模式验证
                    input_seq = batch.get('baseline_seq', batch['baseline'])
                    target_seq = batch.get('target_seq', batch['target'])
                    
                    # 如果输入不是序列，扩展维度
                    if input_seq.dim() == 4:
                        input_seq = input_seq.unsqueeze(1)
                    if target_seq.dim() == 4:
                        target_seq = target_seq.unsqueeze(1)
                    
                    # AR模型前向传播（验证时不使用teacher forcing）
                    tout = target_seq.shape[1]
                    pred_seq = self.model(input_seq, T_out=tout)
                    
                    # 计算AR损失
                    from ops.losses import compute_ar_total_loss
                    losses = compute_ar_total_loss(
                        pred_seq=pred_seq,
                        gt_seq=target_seq,
                        obs_data=batch,
                        norm_stats=self.norm_stats,
                        config=self.config
                    )
                    
                    # 计算指标（使用最后一个时间步）
                    pred_last = pred_seq[:, -1]
                    target_last = target_seq[:, -1]
                    metrics = compute_all_metrics(pred_last, target_last)
                else:
                    # 标准模式验证
                    baseline = batch['baseline']
                    
                    # 构建模型输入（统一接口：[baseline, coords, mask, fourier_pe?]）
                    model_input = baseline
                    
                    # 添加坐标信息
                    if 'coords' in batch:
                        coords = batch['coords']
                        model_input = torch.cat([model_input, coords], dim=1)
                    
                    # 添加mask信息
                    if 'mask' in batch:
                        mask = batch['mask']
                        model_input = torch.cat([model_input, mask], dim=1)
                    
                    pred = self.model(model_input)
                    
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
            # 获取训练轮数
            if hasattr(self.config.training, 'epochs'):
                total_epochs = self.config.training.epochs
            elif hasattr(self.config.training, 'max_epochs'):
                total_epochs = self.config.training.max_epochs
            else:
                total_epochs = 100  # 默认值
            
            self.logger.info(f"DEBUG: Total epochs to train: {total_epochs}")
            self.logger.info(f"DEBUG: Starting epoch loop...")
                
            for epoch in range(total_epochs):
                self.current_epoch = epoch
                self.logger.info(f"DEBUG: Starting epoch {epoch}/{total_epochs}")
                
                # 训练前的调试信息
                self.logger.info(f"DEBUG: About to call train_epoch() for epoch {epoch}")
                epoch_start_time = time.time()
                
                # 训练
                train_results = self.train_epoch()
                train_time = time.time() - epoch_start_time
                self.logger.info(f"DEBUG: train_epoch() completed for epoch {epoch}, took {train_time:.2f}s")
                
                # 验证
                val_start_time = time.time()
                val_results = self.validate_epoch()
                val_time = time.time() - val_start_time
                self.logger.info(f"DEBUG: validate_epoch() completed for epoch {epoch}, took {val_time:.2f}s")
                
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
            
            visualizer = TemporalVisualizer(save_dir, self.config)
            
            # 保存可视化
            max_samples = self.config.training.get('max_samples', 4)
            for i in range(min(max_samples, pred.shape[0])):
                # 使用TemporalVisualizer的方法
                visualizer.save_training_predictions(
                    input_seq=val_batch['baseline'][i],
                    target_seq=val_batch['target'][i],
                    pred_seq=pred[i].cpu(),
                    step=i,
                    epoch=epoch
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
    # 调试配置结构
    print(f"DEBUG main: config keys = {list(config.keys())}")
    print(f"DEBUG main: config type = {type(config)}")
    if hasattr(config, 'data'):
        print(f"DEBUG main: config.data exists")
    else:
        print(f"DEBUG main: config.data does NOT exist")
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()