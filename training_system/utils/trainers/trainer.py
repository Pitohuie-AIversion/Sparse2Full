#!/usr/bin/env python3
"""
PDEBench训练器 - 集成训练、验证、监控功能
遵循黄金法则，支持课程学习和多模型训练
"""

import os
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from training_system.utils.datasets.pde_bench import PDEBenchDataset
from models.unet import UNet
from src.models.temporal_transformer import TemporalTransformerEncoderWrapper as TemporalTransformer
# 使用本仓库的CombinedLoss实现，避免ops.losses别名解析失败
from training_system.utils.losses import CombinedLoss
# 统一使用ops.metrics的实现
from ops.metrics import compute_all_metrics
from ops.degradation import apply_degradation_operator
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.performance import PerformanceProfiler as PerformanceMonitor
from utils.visualization import ARVisualizer as TrainingVisualizer
# from utils.paper_package import PaperPackageGenerator  # 暂时注释，后续需要时添加

logger = logging.getLogger(__name__)


class PDEBenchTrainer:
    """PDEBench主训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        # 统一获取实验名称
        exp = getattr(config, 'experiment', None)
        experiment_name = getattr(exp, 'name', None) or getattr(config, 'experiment_name', 'default_experiment')
        self.output_dir = Path('./runs') / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._setup_model()
        self._setup_data()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_monitoring()
        self._setup_visualization()
        
        # 混合精度训练（避免结构化配置属性错误）
        training_cfg = self.config.get('training', {})
        self.use_amp = training_cfg.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.early_stopping_counter = 0
        
        # 论文包生成器（可选）
        try:
            from utils.paper_package import PaperPackageGenerator
            self.paper_package_generator = PaperPackageGenerator(config, self.output_dir)
        except ImportError:
            self.paper_package_generator = None
            logger.info("论文包生成器未找到，跳过初始化")
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
    
    def _get_device(self) -> torch.device:
        """获取训练设备"""
        # 兼容config.experiment.device与顶层device
        exp = getattr(self.config, 'experiment', None)
        device_str = getattr(exp, 'device', None) or getattr(self.config, 'device', 'cpu')
        device = torch.device(device_str)
        logger.info(f"使用设备: {device}")
        return device
    
    def _setup_model(self):
        """设置模型"""
        # 兼容嵌套'spatial'分组
        cfg_root = getattr(self.config, 'spatial', None)
        model_cfg = getattr(self.config, 'model', None)
        if cfg_root is not None and hasattr(cfg_root, 'model'):
            model_cfg = cfg_root.model
        model_name = getattr(model_cfg, 'name', 'UNet') if model_cfg is not None else 'UNet'
        model_params = getattr(model_cfg, 'params', {}) if model_cfg is not None else {}
        
        logger.info(f"初始化模型: {model_name}")
        
        if model_name.lower() in ["unet"]:
            self.model = UNet(
                in_channels=model_params.get('in_channels', 1),
                out_channels=model_params.get('out_channels', 1),
                img_size=model_params.get('img_size', 256),
                features=model_params.get('kwargs', {}).get('features', [64, 128, 256, 512]),
                bilinear=True,
                dropout=model_params.get('kwargs', {}).get('drop_rate', 0.0)
            )
        else:
            # 兜底：使用UNet以避免SwinUNet依赖问题
            logger.warning(f"模型{model_name}不可用，回退到UNet。")
            self.model = UNet(
                in_channels=model_params.get('in_channels', 1),
                out_channels=model_params.get('out_channels', 1),
                img_size=model_params.get('img_size', 256),
                features=model_params.get('kwargs', {}).get('features', [64, 128, 256, 512]),
                bilinear=True,
                dropout=model_params.get('kwargs', {}).get('drop_rate', 0.0)
            )
        
        self.model.to(self.device)
        self.model_name = model_name
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    def _setup_data(self):
        """设置数据加载器"""
        logger.info("设置数据加载器...")
        # 获取数据配置，兼容'spatial'分组
        cfg_root = getattr(self.config, 'spatial', None)
        data_config = getattr(self.config, 'data', {})
        if cfg_root is not None and hasattr(cfg_root, 'data'):
            data_config = cfg_root.data
        
        # 检查数据类型并选择合适的数据集
        dataset_name = data_config.get('dataset_name', 'PDEBench')
        
        if dataset_name == "RealDiffusionReaction2D":
            logger.info("使用真实扩散-反应数据集")
            try:
                from utils.real_dr_dataset import RealDiffusionReactionDataModule
                
                # 创建数据模块
                self.data_module = RealDiffusionReactionDataModule(self.config)
                self.data_module.setup()
                
                # 获取数据加载器
                self.train_loader = self.data_module.train_dataloader()
                self.val_loader = self.data_module.val_dataloader()
                self.test_loader = self.data_module.test_dataloader()
                
                # 获取数据集引用
                self.train_dataset = self.data_module.train_dataset
                self.val_dataset = self.data_module.val_dataset
                
                logger.info(f"训练样本: {len(self.train_dataset)}, 验证样本: {len(self.val_dataset)}")
                return
            except ImportError:
                logger.warning("真实扩散-反应数据集模块未找到，使用默认PDEBench数据集")
        
        # 默认使用PDEBench数据集
        logger.info("使用PDEBench数据集")
        
        # 获取数据路径和配置：兼容别名 data_path/data_dir/path
        data_path = data_config.get('data_path') or data_config.get('data_dir') or data_config.get('path') or './data'
        keys = data_config.get('keys', ['u'])
        image_size = data_config.get('image_size', 256)
        
        # 观测配置
        observation_config = data_config.get('observation', {})
        
        # 预处理配置
        preprocessing = data_config.get('preprocessing', {})
        normalize = preprocessing.get('normalize', True)
        cache_data = preprocessing.get('cache_data', False)
        
        # 训练数据集
        self.train_dataset = PDEBenchDataset(
            data_path=data_path,
            mode='train',
            img_size=image_size,
            normalize=normalize,
            observation_mode=observation_config.get('mode', 'sr'),
            sr_scale=observation_config.get('scale_factor', 4),
            crop_size=observation_config.get('crop_size', None),
            data_key=data_config.get('data_key', 'data')
        )
        
        # 验证数据集
        self.val_dataset = PDEBenchDataset(
            data_path=data_path,
            mode='val',
            img_size=image_size,
            normalize=normalize,
            observation_mode=observation_config.get('mode', 'sr'),
            sr_scale=observation_config.get('scale_factor', 4),
            crop_size=observation_config.get('crop_size', None),
            data_key=data_config.get('data_key', 'data')
        )
        
        # 数据加载器配置
        dataloader_config = data_config.get('dataloader', {})
        batch_size = dataloader_config.get('batch_size', 16)
        num_workers = int(dataloader_config.get('num_workers', 4))
        pin_memory = bool(dataloader_config.get('pin_memory', True))
        persistent_workers_cfg = bool(dataloader_config.get('persistent_workers', True))
        # 仅在 num_workers>0 时启用 persistent_workers，避免PyTorch报错
        persistent_workers = (num_workers > 0) and persistent_workers_cfg
        val_batch_size = dataloader_config.get('val_batch_size', batch_size)
        prefetch_factor = dataloader_config.get('prefetch_factor', None)

        # 统一构建 DataLoader kwargs 并在可并行时设置预取与spawn上下文
        dl_kwargs = {
            'num_workers': max(0, num_workers),
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
        }
        if (num_workers > 0) and (prefetch_factor is not None):
            try:
                dl_kwargs['prefetch_factor'] = int(prefetch_factor)
            except Exception:
                pass
            # 在多进程加载时使用 spawn，提升稳定性
            try:
                # 兼容 PyTorch：可传入字符串或上下文对象
                dl_kwargs['multiprocessing_context'] = 'spawn'
            except Exception:
                pass

        # 日志输出构建的 DataLoader 关键参数，便于性能诊断
        logger.info(
            f"构建DataLoader: batch_size={batch_size}/{val_batch_size}, num_workers={num_workers}, "
            f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
            f"prefetch_factor={prefetch_factor if (num_workers>0 and prefetch_factor is not None) else 'N/A'}, "
            f"multiprocessing_context={'spawn' if (num_workers>0 and ('multiprocessing_context' in dl_kwargs)) else 'default'}"
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **dl_kwargs,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            **dl_kwargs,
        )
        
        logger.info(f"训练样本: {len(self.train_dataset)}, 验证样本: {len(self.val_dataset)}")
    
    def _setup_loss(self):
        """设置损失函数"""
        loss_config = self.config.get('loss', {})
        
        logger.info("设置损失函数...")
        # 将配置中的l2映射为mse，保持实现一致
        def _map_loss_type(t: str) -> str:
            return 'mse' if str(t).lower() in ['l2', 'mse'] else 'l1'

        self.criterion = CombinedLoss(
            rec_weight=loss_config.get('rec_weight', 1.0),
            spec_weight=loss_config.get('spec_weight', 0.5),
            dc_weight=loss_config.get('dc_weight', 1.0),
            rec_loss_type=_map_loss_type(loss_config.get('rec_loss_type', 'mse')),
            spec_loss_type=_map_loss_type(loss_config.get('spec_loss_type', 'mse')),
            dc_loss_type=_map_loss_type(loss_config.get('dc_loss_type', 'mse')),
            low_freq_modes=loss_config.get('low_freq_modes', 16),
            observation_config=self.config.get('data', {}).get('observation', {}),
            normalization_stats=getattr(self.train_dataset, 'get_normalization_stats', lambda: None)()
        )
        
        # 可选损失
        self.use_gradient_loss = loss_config.get('use_gradient_loss', False)
        self.gradient_weight = loss_config.get('gradient_weight', 0.0)
        
        self.use_pde_residual_loss = loss_config.get('use_pde_residual_loss', False)
        self.pde_residual_weight = loss_config.get('pde_residual_weight', 0.0)
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 使用OmegaConf安全访问，避免结构化配置下的属性错误
        training_config = self.config.get('training', {})
        optimizer_config = training_config.get('optimizer', {})

        logger.info("设置优化器...")

        # 优化器
        optimizer_name = optimizer_config.get('name', 'AdamW')
        if optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999))
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 学习率调度器
        scheduler_config = training_config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        epochs = training_config.get('epochs', 100)

        if scheduler_name == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_name == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
            logger.warning(f"未识别的调度器: {scheduler_name}")

        logger.info(f"优化器: {optimizer_name}, 调度器: {scheduler_name if self.scheduler else 'None'}")
    
    def _setup_monitoring(self):
        """设置监控"""
        logger.info("设置监控...")
        
        # 性能监控
        monitoring_config = self.config.get('monitoring', {})
        if monitoring_config.get('enabled', True):
            # PerformanceProfiler只接受device参数
            self.performance_monitor = PerformanceMonitor(device=str(self.device))
        else:
            self.performance_monitor = None
        
        # TensorBoard
        if monitoring_config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        else:
            self.writer = None
    
    def _setup_visualization(self):
        """设置可视化"""
        logger.info("设置可视化...")
        
        visualization_config = self.config.get('visualization', {})
        if visualization_config.get('enabled', True):
            # ARVisualizer仅接受保存目录参数名为save_dir
            self.visualizer = TrainingVisualizer(
                save_dir=str(self.output_dir / "visualizations")
            )
        else:
            self.visualizer = None
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 前向传播
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 模型前向（混合精度）
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    # 仅传递观测张量进行DC损失计算
                    loss_dict = self.criterion(outputs, targets, batch.get('observation'))
                    total_loss = loss_dict['total_loss']
            else:
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets, batch.get('observation'))
                total_loss = loss_dict['total_loss']
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                
                # 梯度裁剪
                if hasattr(self.config.training, 'grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # 梯度裁剪
                if hasattr(self.config.training, 'grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # 更新损失统计
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 全局步数
            self.global_step += 1
            
            # 日志记录
            if self.writer and batch_idx % self.config.training.log_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"train/{key}", value.item(), self.global_step)
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 模型前向（混合精度）
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # 计算基础指标（Rel-L2/MAE/PSNR/SSIM/频域/边界）
                metrics_all = compute_all_metrics(
                    outputs, targets,
                    observed=batch.get('observation'),
                    mask=None,
                    data_range=None
                )

                # 可选：计算数据一致性误差 ||H(ŷ)−y||，在原值域下
                try:
                    h_params = batch.get('h_params')
                    observation = batch.get('observation')
                    if h_params is not None and observation is not None:
                        pred_obs = apply_degradation_operator(outputs, h_params)
                        dc_err = torch.mean((pred_obs - observation) ** 2).item()
                        metrics_all['dc_error'] = dc_err
                except Exception:
                    pass

                # 根据配置筛选需要的指标
                requested = getattr(self.config, 'validation', {}).get('metrics', None)
                if requested:
                    metrics = {k: v for k, v in metrics_all.items() if k in requested}
                else:
                    metrics = metrics_all
                
                # 累积指标
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
        
        # 计算平均指标
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        return avg_metrics
    
    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        logger.info("开始训练循环...")
        
        training_history = {
            'train_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'resource_usage': [],
            'epoch_times': []
        }
        
        # 性能监控：记录一次模型资源使用作为起始快照
        if self.performance_monitor:
            try:
                input_shape = (1, 1, self.config.get('data', {}).get('image_size', 256), self.config.get('data', {}).get('image_size', 256))
                report = self.performance_monitor.profile_model(self.model, input_shape)
                training_history['resource_usage'].append(report)
                logger.info(f"初始资源快照: params={report['parameters']['total']}, gflops={report['flops'].get('total_gflops', 0):.2f}")
            except Exception as e:
                logger.warning(f"性能监控初始化失败: {e}")
        
        total_epochs = self.config.get('training', {}).get('epochs', 100)
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"Epoch {epoch + 1}/{total_epochs}")
            
            # 训练
            train_losses = self.train_epoch()
            training_history['train_losses'].append(train_losses)
            
            # 验证
            check_every = self.config.get('validation', {}).get('check_val_every_n_epoch', 1)
            if epoch % check_every == 0:
                val_metrics = self.validate()
                training_history['val_metrics'].append(val_metrics)
                
                # 记录验证指标
                if self.writer:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"val/{key}", value, epoch)
                
                # 检查最佳模型
                primary_metric = val_metrics.get('rel_l2', val_metrics.get('mse', float('inf')))
                if primary_metric < self.best_val_metric:
                    self.best_val_metric = primary_metric
                    self.early_stopping_counter = 0
                    
                    # 保存最佳模型
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.global_step, primary_metric,
                        self.output_dir / "best_model.pth"
                    )
                    logger.info(f"保存最佳模型 (val_metric: {primary_metric:.6f})")
                else:
                    self.early_stopping_counter += 1
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['learning_rates'].append(current_lr)
                
                if self.writer:
                    self.writer.add_scalar("train/learning_rate", current_lr, epoch)
            
            # 资源监控
            if self.performance_monitor and epoch % 10 == 0:
                resource_stats = self.performance_monitor.get_stats()
                training_history['resource_usage'].append(resource_stats)
                
                if self.writer:
                    for key, value in resource_stats.items():
                        self.writer.add_scalar(f"resource/{key}", value, epoch)
            
            # 保存检查点
            if epoch % self.config.training.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, primary_metric if 'primary_metric' in locals() else 0,
                    self.output_dir / f"checkpoint_epoch_{epoch}.pth"
                )
            
            # 可视化
            if self.visualizer and epoch % getattr(self.config.visualization, 'plot_interval', 50) == 0:
                self.visualizer.plot_training_progress(
                    training_history, epoch, self.output_dir
                )
            
            # 早停检查
            if hasattr(self.config.training, 'early_stopping'):
                es_config = self.config.training.early_stopping
                if es_config.enabled and self.early_stopping_counter >= es_config.patience:
                    logger.info(f"早停触发 (patience: {es_config.patience})")
                    break
            
            epoch_time = time.time() - epoch_start_time
            training_history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch + 1} 完成，用时: {epoch_time:.2f}s")
        
        # 停止性能监控
        if self.performance_monitor:
            self.performance_monitor.stop()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.current_epoch, self.global_step, self.best_val_metric,
            self.output_dir / "final_model.pth"
        )
        
        # 保存训练历史
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2, default=str)
        
        logger.info("训练完成！")
        
        # 生成论文包
        checkpoints = [self.output_dir / "best_model.pth", self.output_dir / "final_model.pth"]
        package_dir = self.paper_package_generator.generate_complete_package(
            trainer=self,
            validation_results=training_history['val_metrics'][-1] if training_history['val_metrics'] else {},
            checkpoints=checkpoints,
            seed_results=None
        )
        logger.info(f"论文包生成完成: {package_dir}")
        
        return {
            'best_val_metric': self.best_val_metric,
            'total_epochs': self.current_epoch + 1,
            'training_history': training_history,
            'paper_package_dir': str(package_dir)
        }
    
    def generate_paper_package(self):
        """生成论文包"""
        logger.info("生成论文包...")
        
        paper_package_dir = self.output_dir / "paper_package"
        paper_package_dir.mkdir(exist_ok=True)
        
        # 复制配置文件
        import shutil
        shutil.copy2(self.output_dir / "config_merged.yaml", paper_package_dir / "config.yaml")
        
        # 复制模型检查点
        checkpoints_dir = paper_package_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        for ckpt_file in ["best_model.pth", "final_model.pth"]:
            src = self.output_dir / ckpt_file
            if src.exists():
                shutil.copy2(src, checkpoints_dir / ckpt_file)
        
        # 生成指标文件
        metrics_file = paper_package_dir / "metrics.json"
        metrics_data = {
            'model_name': self.model_name,
            'best_val_metric': self.best_val_metric,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"论文包生成完成: {paper_package_dir}")
        
        return paper_package_dir