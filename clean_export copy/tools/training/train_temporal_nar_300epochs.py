#!/usr/bin/env python3
"""
时序NAR模型300轮训练脚本
基于temporal_nar_100epochs.py，优化为长期训练，增强监控和可视化功能
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
import h5py

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.temporal_pdebench import TemporalPDEBenchDataModule
from models.base import create_model
from models.wrappers.ar_nar_wrapper import ARNARWrapper
from ops.losses import ARLoss, SpectralLoss, DCLoss
from utils.metrics import compute_metrics
from utils.visualization import TemporalVisualizer
from utils.logger import setup_logger


class EnhancedTemporalNARTrainer:
    """增强版时序NAR模型训练器 - 300轮训练版本"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        self.use_amp = config.experiment.use_amp
        
        # 设置随机种子
        self._set_seed(config.experiment.seed)
        
        # 创建输出目录
        self.output_dir = Path(config.experiment.output_dir) / config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visualization_dir = self.output_dir / "visualizations"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        
        for dir_path in [self.checkpoint_dir, self.visualization_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(str(self.output_dir / "train.log"))
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # 初始化组件
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self._init_losses()
        self._init_metrics()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'ar_losses': [],
            'nar_losses': [],
            'spectral_losses': [],
            'dc_losses': [],
            'metrics': []
        }
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 可视化器
        viz_config = self.config.get('visualization', {})
        self.visualizer = TemporalVisualizer(save_dir=self.visualization_dir, config=viz_config)
        
        # 保存配置快照
        self._save_config_snapshot()
        
        self.logger.info(f"EnhancedTemporalNARTrainer initialized. Output dir: {self.output_dir}")
        self.logger.info(f"Training for {config.train.max_epochs} epochs with enhanced monitoring")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _save_config_snapshot(self):
        """保存配置快照"""
        config_path = self.output_dir / "config_snapshot.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        # 保存环境信息
        env_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'timestamp': datetime.now().isoformat()
        }
        
        env_path = self.output_dir / "environment.json"
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
    
    def _init_data(self):
        """初始化数据模块"""
        self.logger.info("初始化数据模块...")
        
        # 使用时序数据模块
        self.data_module = TemporalPDEBenchDataModule(self.config.data)
        
        # 获取数据加载器（TemporalPDEBenchDataModule没有setup方法，直接调用dataloader方法）
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        self.logger.info(f"数据加载完成. Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
    
    def _init_model(self):
        """初始化模型"""
        self.logger.info("初始化AR+NAR双头模型...")
        
        # 创建基础模型
        base_model = create_model("swin_unet", **self.config.model.base_kwargs)
        
        # 包装为AR+NAR模型
        self.model = ARNARWrapper(
            model_config=self.config.model,
            loss_config=self.config.loss,
            training_config=self.config.train
        )
        
        self.model = self.model.to(self.device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成: AR+NAR双头时序模型")
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        # 记录到TensorBoard
        self.writer.add_text("Model/Architecture", "AR+NAR双头时序模型")
        self.writer.add_scalar("Model/TotalParams", total_params)
        self.writer.add_scalar("Model/TrainableParams", trainable_params)
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.optimizer.lr,
            weight_decay=self.config.train.optimizer.weight_decay,
            betas=self.config.train.optimizer.betas,
            eps=self.config.train.optimizer.eps
        )
        
        # 学习率调度器 - CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.train.scheduler.T_0,
            T_mult=self.config.train.scheduler.T_mult,
            eta_min=self.config.train.scheduler.eta_min
        )
        
        # Warmup调度器（可选）
        self.warmup_steps = self.config.train.scheduler.warmup_steps
        self.warmup_lr = self.config.train.scheduler.warmup_lr
        
        self.logger.info(f"优化器: AdamW, LR: {self.config.train.optimizer.lr}")
        self.logger.info(f"调度器: CosineAnnealingWarmRestarts, T_0: {self.config.train.scheduler.T_0}")
    
    def _init_losses(self):
        """初始化损失函数"""
        # 配置AR损失函数
        ar_loss_config = OmegaConf.create({
            'loss_type': self.config.loss.get('ar_loss_type', 'mse'),
            'step_weights': None,
            'accumulate_loss': True
        })
        self.ar_loss = ARLoss(ar_loss_config)
        
        # NAR损失
        self.nar_loss = nn.MSELoss() if self.config.loss.get('nar_loss_type', 'mse') == "mse" else nn.L1Loss()
        
        # 频谱损失
        if hasattr(self.config.loss, 'spectral_loss') and self.config.loss.spectral_loss.weight > 0:
            spectral_loss_config = OmegaConf.create({
                'k_max': self.config.loss.spectral_loss.get('k_max', 16)
            })
            self.spectral_loss = SpectralLoss(spectral_loss_config)
        else:
            self.spectral_loss = None
        
        # DC损失
        if hasattr(self.config.loss, 'dc_loss') and self.config.loss.dc_loss.weight > 0:
            self.dc_loss = DCLoss(self.config.loss.dc_loss)
        else:
            self.dc_loss = None
        
        # 梯度损失
        if hasattr(self.config.loss, 'gradient_loss') and self.config.loss.gradient_loss.enabled:
            self.gradient_loss_weight = self.config.loss.gradient_loss.weight
        else:
            self.gradient_loss_weight = 0.0
        
        self.logger.info("损失函数初始化完成")
    
    def _init_metrics(self):
        """初始化评估指标"""
        self.metric_names = ['mse', 'mae', 'rel_l2', 'psnr', 'ssim']
        self.logger.info(f"评估指标: {self.metric_names}")
    
    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算梯度损失"""
        # 计算x方向梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # 计算y方向梯度
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 梯度损失
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def _get_loss_weights(self, epoch: int) -> Tuple[float, float]:
        """获取当前epoch的损失权重"""
        # AR权重调度
        if self.config.loss.ar_weight_schedule == "cosine":
            ar_weight = self.config.loss.ar_weight * (0.5 + 0.5 * np.cos(np.pi * epoch / self.config.train.max_epochs))
        else:
            ar_weight = self.config.loss.ar_weight
        
        # NAR权重调度
        if self.config.loss.nar_weight_schedule == "linear":
            nar_weight = self.config.loss.nar_weight * (epoch / self.config.train.max_epochs)
        else:
            nar_weight = self.config.loss.nar_weight
        
        return ar_weight, nar_weight
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        # 损失统计
        epoch_losses = {
            'total': 0.0,
            'ar': 0.0,
            'nar': 0.0,
            'spectral': 0.0,
            'dc': 0.0,
            'gradient': 0.0
        }
        num_batches = 0
        
        # 获取当前权重
        ar_weight, nar_weight = self._get_loss_weights(self.current_epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.train.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 获取数据
                x_seq = batch['input_sequence'].to(self.device)  # [B, T_in, C, H, W]
                y_seq = batch['target_sequence'].to(self.device)  # [B, T_out, C, H, W]
                
                # 前向传播
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(x_seq, y_seq, mode='train')
                        
                        # 计算损失
                        total_loss = 0.0
                        
                        # AR损失
                        if 'ar_pred' in outputs:
                            ar_loss_val = self.ar_loss(outputs['ar_pred'], y_seq)
                            total_loss += ar_weight * ar_loss_val
                            epoch_losses['ar'] += ar_loss_val.item()
                        
                        # NAR损失
                        if 'nar_pred' in outputs:
                            nar_loss_val = self.nar_loss(outputs['nar_pred'], y_seq)
                            total_loss += nar_weight * nar_loss_val
                            epoch_losses['nar'] += nar_loss_val.item()
                        
                        # 频谱损失
                        if self.spectral_loss is not None:
                            pred_for_spectral = outputs.get('ar_pred', outputs.get('nar_pred'))
                            if pred_for_spectral is not None:
                                spectral_loss_val = self.spectral_loss(pred_for_spectral, y_seq)
                                total_loss += spectral_loss_val
                                epoch_losses['spectral'] += spectral_loss_val.item()
                        
                        # DC损失
                        if self.dc_loss is not None:
                            pred_for_dc = outputs.get('ar_pred', outputs.get('nar_pred'))
                            if pred_for_dc is not None:
                                dc_loss_val = self.dc_loss(pred_for_dc, y_seq)
                                total_loss += dc_loss_val
                                epoch_losses['dc'] += dc_loss_val.item()
                        
                        # 梯度损失
                        if self.gradient_loss_weight > 0:
                            pred_for_grad = outputs.get('ar_pred', outputs.get('nar_pred'))
                            if pred_for_grad is not None:
                                grad_loss_val = self._compute_gradient_loss(pred_for_grad, y_seq)
                                total_loss += self.gradient_loss_weight * grad_loss_val
                                epoch_losses['gradient'] += grad_loss_val.item()
                    
                    self.scaler.scale(total_loss).backward()
                    
                    # 梯度裁剪
                    if self.config.train.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_val)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(x_seq, y_seq, mode='train')
                    
                    # 计算损失（同上，但不使用autocast）
                    total_loss = 0.0
                    
                    if 'ar_pred' in outputs:
                        ar_loss_val = self.ar_loss(outputs['ar_pred'], y_seq)
                        total_loss += ar_weight * ar_loss_val
                        epoch_losses['ar'] += ar_loss_val.item()
                    
                    if 'nar_pred' in outputs:
                        nar_loss_val = self.nar_loss(outputs['nar_pred'], y_seq)
                        total_loss += nar_weight * nar_loss_val
                        epoch_losses['nar'] += nar_loss_val.item()
                    
                    if self.spectral_loss is not None:
                        pred_for_spectral = outputs.get('ar_pred', outputs.get('nar_pred'))
                        if pred_for_spectral is not None:
                            spectral_loss_val = self.spectral_loss(pred_for_spectral, y_seq)
                            total_loss += spectral_loss_val
                            epoch_losses['spectral'] += spectral_loss_val.item()
                    
                    if self.dc_loss is not None:
                        pred_for_dc = outputs.get('ar_pred', outputs.get('nar_pred'))
                        if pred_for_dc is not None:
                            dc_loss_val = self.dc_loss(pred_for_dc, y_seq)
                            total_loss += dc_loss_val
                            epoch_losses['dc'] += dc_loss_val.item()
                    
                    if self.gradient_loss_weight > 0:
                        pred_for_grad = outputs.get('ar_pred', outputs.get('nar_pred'))
                        if pred_for_grad is not None:
                            grad_loss_val = self._compute_gradient_loss(pred_for_grad, y_seq)
                            total_loss += self.gradient_loss_weight * grad_loss_val
                            epoch_losses['gradient'] += grad_loss_val.item()
                    
                    total_loss.backward()
                    
                    # 梯度裁剪
                    if self.config.train.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_val)
                    
                    self.optimizer.step()
                
                # 更新统计
                epoch_losses['total'] += total_loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Warmup学习率调度
                if self.global_step <= self.warmup_steps:
                    lr = self.warmup_lr + (self.config.train.optimizer.lr - self.warmup_lr) * self.global_step / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                
                # 记录到TensorBoard
                if self.global_step % self.config.experiment.log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("Train/TotalLoss", total_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/LearningRate", current_lr, self.global_step)
                    self.writer.add_scalar("Train/ARWeight", ar_weight, self.global_step)
                    self.writer.add_scalar("Train/NARWeight", nar_weight, self.global_step)
                    
                    if 'ar_pred' in outputs:
                        self.writer.add_scalar("Train/ARLoss", epoch_losses['ar'] / num_batches, self.global_step)
                    if 'nar_pred' in outputs:
                        self.writer.add_scalar("Train/NARLoss", epoch_losses['nar'] / num_batches, self.global_step)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'AR': f'{ar_weight:.3f}',
                    'NAR': f'{nar_weight:.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                self.logger.error(f"训练批次 {batch_idx} 出错: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue
        
        # 计算平均损失
        avg_losses = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in epoch_losses.items()}
        
        # 更新学习率调度器
        if self.global_step > self.warmup_steps:
            self.scheduler.step()
        
        return avg_losses
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'ar': 0.0,
            'nar': 0.0
        }
        all_metrics = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    x_seq = batch['input_sequence'].to(self.device)
                    y_seq = batch['target_sequence'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(x_seq, mode='inference')
                    
                    # 计算损失
                    total_loss = 0.0
                    
                    if 'ar_pred' in outputs:
                        ar_loss_val = self.ar_loss(outputs['ar_pred'], y_seq)
                        val_losses['ar'] += ar_loss_val.item()
                        total_loss += ar_loss_val
                    
                    if 'nar_pred' in outputs:
                        nar_loss_val = self.nar_loss(outputs['nar_pred'], y_seq)
                        val_losses['nar'] += nar_loss_val.item()
                        total_loss += nar_loss_val
                    
                    val_losses['total'] += total_loss.item()
                    
                    # 计算指标
                    pred_for_metrics = outputs.get('nar_pred', outputs.get('ar_pred'))
                    if pred_for_metrics is not None:
                        batch_metrics = compute_metrics(pred_for_metrics, y_seq, self.metric_names)
                        all_metrics.append(batch_metrics)
                    
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"验证批次出错: {str(e)}")
                    continue
        
        # 计算平均损失和指标
        avg_val_losses = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in val_losses.items()}
        
        if all_metrics:
            avg_metrics = {}
            for metric_name in self.metric_names:
                metric_values = [m[metric_name] for m in all_metrics if metric_name in m]
                avg_metrics[metric_name] = np.mean(metric_values) if metric_values else 0.0
        else:
            avg_metrics = {name: 0.0 for name in self.metric_names}
        
        return avg_val_losses, avg_metrics
    
    def save_checkpoint(self, is_best: bool = False, milestone: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳检查点: {best_path}")
        
        # 保存里程碑检查点
        if milestone:
            milestone_path = self.checkpoint_dir / f"epoch_{self.current_epoch:03d}.pth"
            torch.save(checkpoint, milestone_path)
            self.logger.info(f"保存里程碑检查点: {milestone_path}")
        
        # 定期保存
        if self.current_epoch % self.config.checkpoint.every_n_epochs == 0:
            periodic_path = self.checkpoint_dir / f"periodic_epoch_{self.current_epoch:03d}.pth"
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
            self.logger.info(f"恢复到 epoch {self.current_epoch}, step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return False
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def create_training_report(self):
        """创建训练报告"""
        # 生成训练曲线图
        self.visualizer.plot_training_curves(
            self.training_history,
            save_path=str(self.visualization_dir / "training_curves.png")
        )
        
        # 生成损失分解图
        if len(self.training_history['ar_losses']) > 0:
            self.visualizer.plot_loss_breakdown(
                self.training_history,
                save_path=str(self.visualization_dir / "loss_breakdown.png")
            )
        
        # 生成学习率曲线
        if len(self.training_history['learning_rates']) > 0:
            self.visualizer.plot_learning_rate(
                self.training_history['learning_rates'],
                save_path=str(self.visualization_dir / "learning_rate.png")
            )
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        self.logger.info(f"训练配置: {self.config.train.max_epochs} epochs, {len(self.train_loader)} batches/epoch")
        
        # 自动恢复训练（如果配置存在）
        if hasattr(self.config, 'resume') and self.config.resume.get('auto_resume', False):
            latest_checkpoint = self.checkpoint_dir / "latest.pth"
            if latest_checkpoint.exists():
                self.load_checkpoint(str(latest_checkpoint))
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.train.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            try:
                # 训练
                train_losses = self.train_epoch()
                
                # 验证
                if epoch % self.config.train.validation.check_val_every_n_epoch == 0:
                    val_losses, val_metrics = self.validate_epoch()
                    
                    # 记录历史
                    self.training_history['train_losses'].append(train_losses['total'])
                    self.training_history['val_losses'].append(val_losses['total'])
                    self.training_history['ar_losses'].append(train_losses['ar'])
                    self.training_history['nar_losses'].append(train_losses['nar'])
                    self.training_history['spectral_losses'].append(train_losses['spectral'])
                    self.training_history['dc_losses'].append(train_losses['dc'])
                    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    self.training_history['metrics'].append(val_metrics)
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar("Val/TotalLoss", val_losses['total'], epoch)
                    self.writer.add_scalar("Val/ARLoss", val_losses['ar'], epoch)
                    self.writer.add_scalar("Val/NARLoss", val_losses['nar'], epoch)
                    
                    for metric_name, metric_value in val_metrics.items():
                        self.writer.add_scalar(f"Val/{metric_name.upper()}", metric_value, epoch)
                    
                    # 检查是否为最佳模型
                    is_best = val_losses['total'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_losses['total']
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    
                    # 保存检查点
                    is_milestone = epoch in self.config.monitoring.checkpoint_callback.milestone.epochs
                    self.save_checkpoint(is_best=is_best, milestone=is_milestone)
                    
                    # 早停检查
                    if self.early_stopping_counter >= self.config.train.early_stopping.patience:
                        self.logger.info(f"早停触发，在epoch {epoch}")
                        break
                    
                    # 日志输出
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.train.max_epochs} | "
                        f"Train Loss: {train_losses['total']:.6f} | "
                        f"Val Loss: {val_losses['total']:.6f} | "
                        f"Val Rel-L2: {val_metrics.get('rel_l2', 0):.6f} | "
                        f"Time: {epoch_time:.2f}s | "
                        f"Best: {self.best_val_loss:.6f}"
                    )
                else:
                    # 只记录训练损失
                    self.training_history['train_losses'].append(train_losses['total'])
                    self.training_history['ar_losses'].append(train_losses['ar'])
                    self.training_history['nar_losses'].append(train_losses['nar'])
                    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.train.max_epochs} | "
                        f"Train Loss: {train_losses['total']:.6f} | "
                        f"Time: {epoch_time:.2f}s"
                    )
                
                # 保存训练历史
                if epoch % 10 == 0:
                    self.save_training_history()
                
                # 生成可视化
                if hasattr(self.config.monitoring.visualization, 'save_every_n_epochs'):
                    if epoch % self.config.monitoring.visualization.save_every_n_epochs == 0:
                        self.create_training_report()
                
            except Exception as e:
                self.logger.error(f"Epoch {epoch} 训练出错: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue
        
        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"训练完成! 总时间: {total_time/3600:.2f} 小时")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        # 保存最终结果
        self.save_training_history()
        self.create_training_report()
        
        # 关闭TensorBoard writer
        self.writer.close()


@hydra.main(version_base=None, config_path="configs/experiment", config_name="temporal_nar_300epochs")
def main(config: DictConfig) -> None:
    """主函数"""
    try:
        # 创建训练器
        trainer = EnhancedTemporalNARTrainer(config)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logging.error(f"训练失败: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()