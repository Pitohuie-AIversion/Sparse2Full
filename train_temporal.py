#!/usr/bin/env python3
"""
时序PDE训练脚本
支持AR模型的时序预测训练，包含实时可视化和完整的训练管理
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.temporal_pdebench import TemporalPDEBenchDataModule
from models.base import create_model
from ops.losses import ARLoss, SpectralLoss, DCLoss
from utils.metrics import compute_metrics
from utils.visualization import TemporalVisualizer
from utils.logger import setup_logger


class TemporalTrainer:
    """时序PDE训练器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        self.use_amp = config.experiment.use_amp
        
        # 设置随机种子
        self._set_seed(config.experiment.seed)
        
        # 创建输出目录
        self.output_dir = Path(config.experiment.output_dir) / config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(str(self.output_dir / "train.log"))
        
        # 初始化组件
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self._init_losses()
        self._init_metrics()
        self._init_visualizer()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 课程学习
        self.curriculum_stage = 0
        self._init_curriculum()
        
        self.logger.info(f"Trainer initialized. Output dir: {self.output_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_data(self):
        """初始化数据模块"""
        self.logger.info("Initializing data module...")
        self.data_module = TemporalPDEBenchDataModule(self.config)
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
        
        self.logger.info(f"Data loaded. Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
    
    def _init_model(self):
        """初始化模型"""
        self.logger.info("Initializing model...")
        
        # 创建AR模型
        model_config = self.config.model.copy()
        model_config.ar_config = self.config.temporal.ar
        
        # 检查是否为AR模型
        if model_config.name == "ARWrapper" or model_config.name == "ar_wrapper":
            # 对于AR模型，需要先创建基础模型，然后用ARWrapper包装
            from models.ar.wrapper import ARWrapper
            
            # 从配置中获取基础模型信息
            base_model_config = model_config.get('base_model', {})
            if isinstance(base_model_config, str):
                # 如果base_model是字符串，直接使用
                base_model_name = base_model_config
                base_model_config = {}
            else:
                # 如果base_model是字典，从中获取name
                base_model_name = base_model_config.get('name', 'SwinUNet')
            
            # 创建基础模型的参数
            base_model_kwargs = {
                'in_channels': model_config.get('in_channels', 3),
                'out_channels': model_config.get('out_channels', 1),
                'img_size': model_config.get('img_size', 256)
            }
            
            # 添加基础模型的其他参数
            for key, value in base_model_config.items():
                if key != 'name':
                    base_model_kwargs[key] = value
            
            # 使用models.__init__.py中的create_model函数创建基础模型
            from models import create_model as create_model_init
            base_model = create_model_init(base_model_name, **base_model_kwargs)
            
            # 创建AR包装器的参数
            ar_kwargs = {}
            if 'detach_rollout' in model_config:
                ar_kwargs['detach_rollout'] = model_config['detach_rollout']
            if 'scheduled_sampling' in model_config:
                ar_kwargs['scheduled_sampling'] = model_config['scheduled_sampling']
            if 'sampling_schedule' in model_config:
                ar_kwargs['sampling_schedule'] = model_config['sampling_schedule']
            
            # 创建AR包装器
            self.model = ARWrapper(single_frame_model=base_model, **ar_kwargs)
        else:
            # 对于非AR模型，使用原来的逻辑
            # 手动构建参数字典，避免重复传递name
            model_kwargs = {
                'in_channels': model_config.in_channels,
                'out_channels': model_config.out_channels,
                'img_size': model_config.img_size,
            }
            
            # 添加其他配置参数（除了name）
            for key, value in model_config.items():
                if key not in ['name', 'in_channels', 'out_channels', 'img_size']:
                    model_kwargs[key] = value
            
            # 使用models.__init__.py中的create_model函数
            from models import create_model as create_model_init
            self.model = create_model_init(
                model_config.name,
                **model_kwargs
            )
        
        self.model = self.model.to(self.device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created: {model_config.name}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        optimizer_config = self.config.train.optimizer
        
        if optimizer_config.name.lower() in ["adamw", "AdamW"]:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas
            )
        elif optimizer_config.name.lower() in ["adam", "Adam"]:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
        
        # 学习率调度器
        scheduler_config = self.config.train.scheduler
        if scheduler_config.name.lower() in ["cosine_annealing", "cosineannealing", "cosineannealinglr"]:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
        elif scheduler_config.name.lower() in ["step", "steplr"]:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {optimizer_config.name}, LR: {optimizer_config.lr}")
    
    def _init_losses(self):
        """初始化损失函数"""
        loss_config = self.config.loss
        
        # AR时序损失
        if hasattr(loss_config, 'ar_loss'):
            self.ar_loss = ARLoss(config=loss_config.ar_loss)
        else:
            # 使用默认配置
            default_ar_config = {'weight': 1.0, 'reduction': 'mean'}
            self.ar_loss = ARLoss(config=default_ar_config)
        
        # 频域损失
        if hasattr(loss_config, 'spectral_loss') or hasattr(loss_config, 'spectral'):
            spectral_config = getattr(loss_config, 'spectral_loss', getattr(loss_config, 'spectral', {}))
            self.spectral_loss = SpectralLoss(config=spectral_config)
        else:
            # 使用默认配置
            default_spectral_config = {'weight': 0.1}
            self.spectral_loss = SpectralLoss(config=default_spectral_config)
        
        # DC一致性损失
        if hasattr(loss_config, 'dc_loss') or hasattr(loss_config, 'degradation_consistency'):
            dc_config = getattr(loss_config, 'dc_loss', getattr(loss_config, 'degradation_consistency', {}))
            self.dc_loss = DCLoss(config=dc_config)
        else:
            # 使用默认配置
            default_dc_config = {'weight': 0.5}
            self.dc_loss = DCLoss(config=default_dc_config)
        
        self.logger.info("Loss functions initialized")
    
    def _init_metrics(self):
        """初始化评估指标"""
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rel_l2': [],
            'val_rel_l2': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
    
    def _init_visualizer(self):
        """初始化可视化器"""
        # 检查是否有可视化配置
        if hasattr(self.config, 'visualization') and self.config.visualization.get('enabled', False):
            self.visualizer = TemporalVisualizer(
                save_dir=self.output_dir / self.config.visualization.get('save_dir', 'visualizations'),
                config=self.config.visualization
            )
        else:
            # 使用默认配置创建可视化器
            viz_config = {
                'enabled': True,
                'save_dir': 'visualizations',
                'training': {
                    'plot_curves': True,
                    'save_predictions': True,
                    'plot_interval': 100
                }
            }
            self.visualizer = TemporalVisualizer(
                save_dir=self.output_dir / 'visualizations',
                config=viz_config
            )
    
    def _init_curriculum(self):
        """初始化课程学习"""
        if self.config.curriculum.enabled:
            self.curriculum_stages = self.config.curriculum.stages
            self.logger.info(f"Curriculum learning enabled with {len(self.curriculum_stages)} stages")
        else:
            self.curriculum_stages = None
    
    def _update_curriculum(self):
        """更新课程学习阶段"""
        if not self.curriculum_stages:
            return
        
        # 检查是否需要切换到下一阶段
        current_stage = self.curriculum_stages[self.curriculum_stage]
        if self.current_epoch >= current_stage.epochs:
            if self.curriculum_stage < len(self.curriculum_stages) - 1:
                self.curriculum_stage += 1
                new_stage = self.curriculum_stages[self.curriculum_stage]
                
                # 更新模型配置
                if hasattr(self.model, 'update_ar_config'):
                    self.model.update_ar_config({
                        'T_out': new_stage.T_out,
                        'teacher_forcing_ratio': new_stage.teacher_forcing_ratio
                    })
                
                self.logger.info(f"Switched to curriculum stage {self.curriculum_stage + 1}: "
                               f"T_out={new_stage.T_out}, TF_ratio={new_stage.teacher_forcing_ratio}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'rel_l2': [], 'mae': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            input_seq = batch['input_sequence'].to(self.device)  # [B, T_in, C, H, W]
            target_seq = batch['target_sequence'].to(self.device)  # [B, T_out, C, H, W]
            
            # 如果有观测序列，使用观测序列作为输入
            if 'observation_sequence' in batch:
                input_seq = batch['observation_sequence'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    # 处理时序数据：将时间维度展平或选择最后一帧
                    if len(input_seq.shape) == 5:  # [B, T, C, H, W]
                        B, T, C, H, W = input_seq.shape
                        # 使用最后一帧作为输入
                        model_input = input_seq[:, -1]  # [B, C, H, W]
                    else:
                        model_input = input_seq
                    
                    outputs = self.model(model_input)
                    
                    # 如果输出是字典，提取预测结果
                    if isinstance(outputs, dict):
                        predictions = outputs
                    else:
                        predictions = {'predictions': outputs}
                    
                    loss = self._compute_loss(predictions, target_seq, batch)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.train.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 处理时序数据：将时间维度展平或选择最后一帧
                if len(input_seq.shape) == 5:  # [B, T, C, H, W]
                    B, T, C, H, W = input_seq.shape
                    # 使用最后一帧作为输入
                    model_input = input_seq[:, -1]  # [B, C, H, W]
                else:
                    model_input = input_seq
                
                outputs = self.model(model_input)
                
                # 如果输出是字典，提取预测结果
                if isinstance(outputs, dict):
                    predictions = outputs
                else:
                    predictions = {'predictions': outputs}
                
                loss = self._compute_loss(predictions, target_seq, batch)
                
                loss.backward()
                
                if self.config.train.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                # 确保predictions和target_seq的尺寸匹配
                pred_for_metrics = predictions['predictions']
                target_for_metrics = target_seq
                
                # 如果predictions是4D而target_seq是5D，需要调整
                if len(pred_for_metrics.shape) == 4 and len(target_for_metrics.shape) == 5:
                    # 上采样predictions到目标分辨率
                    B, C, H, W = pred_for_metrics.shape
                    B_t, T_out, C_t, H_t, W_t = target_for_metrics.shape
                    
                    if H != H_t or W != W_t:
                        pred_for_metrics = F.interpolate(pred_for_metrics, size=(H_t, W_t), mode='bilinear', align_corners=False)
                    
                    # 扩展时间维度
                    pred_for_metrics = pred_for_metrics.unsqueeze(1).repeat(1, T_out, 1, 1, 1)
                
                elif len(pred_for_metrics.shape) == 5 and len(target_for_metrics.shape) == 5:
                    # 都是5D，检查空间分辨率
                    B, T, C, H, W = pred_for_metrics.shape
                    B_t, T_t, C_t, H_t, W_t = target_for_metrics.shape
                    if H != H_t or W != W_t:
                        # 对每个时间步进行上采样
                        predictions_list = []
                        for t in range(T):
                            pred_t = F.interpolate(pred_for_metrics[:, t], size=(H_t, W_t), mode='bilinear', align_corners=False)
                            predictions_list.append(pred_t)
                        pred_for_metrics = torch.stack(predictions_list, dim=1)
                
                metrics = compute_metrics(pred_for_metrics, target_for_metrics)
                epoch_metrics['rel_l2'].append(metrics['rel_l2'].mean().item())
                epoch_metrics['mae'].append(metrics['mae'].mean().item())
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rel_l2': f"{metrics['rel_l2'].mean().item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 保存预测可视化
            if (self.visualizer and 
                hasattr(self.config, 'visualization') and
                self.config.visualization.get('training', {}).get('save_model_predictions', False) and
                self.global_step % self.config.visualization.get('training', {}).get('prediction_frequency', 100) == 0):
                
                # 处理可视化数据的维度
                print(f"可视化数据形状 - input: {input_seq.shape}, target: {target_seq.shape}")
                print(f"outputs type: {type(outputs)}")
                
                # 安全地获取第一个样本
                if len(input_seq.shape) == 5:  # [B, T, C, H, W]
                    vis_input = input_seq[0]  # [T, C, H, W]
                else:  # [B, C, H, W]
                    vis_input = input_seq[0:1]  # [1, C, H, W]
                
                if len(target_seq.shape) == 5:  # [B, T, C, H, W]
                    vis_target = target_seq[0]  # [T, C, H, W]
                else:  # [B, C, H, W]
                    vis_target = target_seq[0:1]  # [1, C, H, W]
                
                # 检查outputs的类型
                if isinstance(outputs, dict) and 'predictions' in outputs:
                    pred_tensor = outputs['predictions']
                    print(f"pred shape from dict: {pred_tensor.shape}")
                    if len(pred_tensor.shape) == 5:  # [B, T, C, H, W]
                        vis_pred = pred_tensor[0]  # [T, C, H, W]
                    else:  # [B, C, H, W]
                        vis_pred = pred_tensor[0:1]  # [1, C, H, W]
                else:
                    # outputs是tensor
                    print(f"pred shape from tensor: {outputs.shape}")
                    if len(outputs.shape) == 5:  # [B, T, C, H, W]
                        vis_pred = outputs[0]  # [T, C, H, W]
                    else:  # [B, C, H, W]
                        vis_pred = outputs[0:1]  # [1, C, H, W]
                
                print(f"可视化数据调整后 - input: {vis_input.shape}, target: {vis_target.shape}, pred: {vis_pred.shape}")
                
                self.visualizer.save_training_predictions(
                    vis_input, vis_target, vis_pred,
                    self.global_step, self.current_epoch
                )
            
            # 日志记录
            if self.global_step % self.config.experiment.log_every_n_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"rel_l2={metrics['rel_l2'].mean().item():.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # 计算epoch平均值
        avg_loss = np.mean(epoch_losses)
        avg_rel_l2 = np.mean(epoch_metrics['rel_l2'])
        avg_mae = np.mean(epoch_metrics['mae'])
        
        return {
            'loss': avg_loss,
            'rel_l2': avg_rel_l2,
            'mae': avg_mae
        }
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        val_losses = []
        val_metrics = {'rel_l2': [], 'mae': [], 'psnr': [], 'ssim': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                if 'observation_sequence' in batch:
                    input_seq = batch['observation_sequence'].to(self.device)
                
                # 前向传播
                # 处理时序数据：将时间维度展平或选择最后一帧
                if len(input_seq.shape) == 5:  # [B, T, C, H, W]
                    B, T, C, H, W = input_seq.shape
                    # 使用最后一帧作为输入
                    model_input = input_seq[:, -1]  # [B, C, H, W]
                else:
                    model_input = input_seq
                
                outputs = self.model(model_input)
                
                # 如果输出是字典，提取预测结果
                if isinstance(outputs, dict):
                    predictions = outputs
                else:
                    predictions = {'predictions': outputs}
                
                loss = self._compute_loss(predictions, target_seq, batch)
                
                # 计算指标 - 处理尺寸不匹配
                pred_for_metrics = predictions['predictions']
                target_for_metrics = target_seq
                
                # 调试信息
                print(f"验证阶段 - pred shape: {pred_for_metrics.shape}, target shape: {target_for_metrics.shape}")
                
                # 第一步：处理维度不匹配
                if len(pred_for_metrics.shape) == 4 and len(target_for_metrics.shape) == 5:
                    # pred: [B, C, H, W], target: [B, T, C, H, W]
                    print(f"情况1: pred 4D, target 5D")
                    target_for_metrics = target_for_metrics[:, -1]  # 取最后一个时间步
                elif len(pred_for_metrics.shape) == 5 and len(target_for_metrics.shape) == 4:
                    # pred: [B, T, C, H, W], target: [B, C, H, W]
                    print(f"情况2: pred 5D, target 4D")
                    pred_for_metrics = pred_for_metrics[:, -1]  # 取最后一个时间步
                
                # 第二步：处理空间尺寸不匹配
                if pred_for_metrics.shape[-2:] != target_for_metrics.shape[-2:]:
                    print(f"空间尺寸不匹配: pred {pred_for_metrics.shape[-2:]} vs target {target_for_metrics.shape[-2:]}")
                    target_size = target_for_metrics.shape[-2:]
                    
                    if len(pred_for_metrics.shape) == 5:
                        # 5D张量 [B, T, C, H, W]
                        B, T, C = pred_for_metrics.shape[:3]
                        pred_reshaped = pred_for_metrics.view(B*T, C, *pred_for_metrics.shape[-2:])
                        pred_resized = F.interpolate(pred_reshaped, size=target_size, mode='bilinear', align_corners=False)
                        pred_for_metrics = pred_resized.view(B, T, C, *target_size)
                    else:
                        # 4D张量 [B, C, H, W]
                        pred_for_metrics = F.interpolate(pred_for_metrics, size=target_size, mode='bilinear', align_corners=False)
                    
                    print(f"空间尺寸调整后 - pred shape: {pred_for_metrics.shape}")
                
                print(f"最终 - pred shape: {pred_for_metrics.shape}, target shape: {target_for_metrics.shape}")
                
                metrics = compute_metrics(pred_for_metrics, target_for_metrics)
                
                val_losses.append(loss.item())
                for key in val_metrics:
                    if key in metrics:
                        # 确保转换为CPU上的标量值
                        metric_value = metrics[key]
                        if isinstance(metric_value, torch.Tensor):
                            if metric_value.numel() > 1:
                                metric_value = metric_value.mean().cpu().item()
                            else:
                                metric_value = metric_value.cpu().item()
                        val_metrics[key].append(metric_value)
        
        # 计算平均值
        avg_metrics = {
            'loss': np.mean(val_losses),
            **{key: np.mean(values) for key, values in val_metrics.items() if values}
        }
        
        return avg_metrics
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     target_seq: torch.Tensor, batch: Dict) -> torch.Tensor:
        """计算损失"""
        predictions = outputs['predictions']  # [B, C, H, W] 或 [B, T_out, C, H, W]
        
        # 确保predictions和target_seq有相同的维度
        if len(predictions.shape) == 4 and len(target_seq.shape) == 5:
            # 如果predictions是4D [B, C, H, W]，target_seq是5D [B, T_out, C, H, W]
            # 需要将predictions上采样到目标分辨率，然后扩展时间维度
            B, C, H, W = predictions.shape
            B_t, T_out, C_t, H_t, W_t = target_seq.shape
            
            # 上采样predictions到目标分辨率
            if H != H_t or W != W_t:
                predictions = F.interpolate(predictions, size=(H_t, W_t), mode='bilinear', align_corners=False)
            
            # 扩展时间维度：复制预测结果到所有时间步
            predictions = predictions.unsqueeze(1).repeat(1, T_out, 1, 1, 1)  # [B, T_out, C, H, W]
            
        elif len(predictions.shape) == 4 and len(target_seq.shape) == 4:
            # 都是4D，检查空间分辨率
            B, C, H, W = predictions.shape
            B_t, C_t, H_t, W_t = target_seq.shape
            if H != H_t or W != W_t:
                predictions = F.interpolate(predictions, size=(H_t, W_t), mode='bilinear', align_corners=False)
                
        elif len(predictions.shape) == 5 and len(target_seq.shape) == 5:
            # 都是5D，检查空间分辨率
            B, T, C, H, W = predictions.shape
            B_t, T_t, C_t, H_t, W_t = target_seq.shape
            if H != H_t or W != W_t:
                # 对每个时间步进行上采样
                predictions_list = []
                for t in range(T):
                    pred_t = F.interpolate(predictions[:, t], size=(H_t, W_t), mode='bilinear', align_corners=False)
                    predictions_list.append(pred_t)
                predictions = torch.stack(predictions_list, dim=1)
        else:
            # 其他情况，尝试添加时间维度
            if len(predictions.shape) == 4:
                predictions = predictions.unsqueeze(1)  # [B, 1, C, H, W]
            if len(target_seq.shape) == 4:
                target_seq = target_seq.unsqueeze(1)  # [B, 1, C, H, W]
        
        # AR时序损失
        ar_loss_result = self.ar_loss(predictions, target_seq)
        if isinstance(ar_loss_result, dict):
            ar_loss = ar_loss_result['total_loss']
        else:
            ar_loss = ar_loss_result
        
        # 频域损失
        spectral_loss_result = self.spectral_loss(predictions, target_seq)
        if isinstance(spectral_loss_result, dict):
            spectral_loss = spectral_loss_result['total_loss']
        else:
            spectral_loss = spectral_loss_result
        
        # DC一致性损失
        dc_loss = torch.tensor(0.0, device=self.device)
        if 'h_params' in batch:
            # 对于DC损失，我们需要使用4D张量
            if len(predictions.shape) == 5:
                # 如果是5D [B, T_out, C, H, W]，取最后一个时间步
                pred_for_dc = predictions[:, -1]  # [B, C, H, W]
            else:
                pred_for_dc = predictions  # 已经是4D
            
            if len(target_seq.shape) == 5:
                # 如果是5D [B, T_out, C, H, W]，取最后一个时间步
                target_for_dc = target_seq[:, -1]  # [B, C, H, W]
            else:
                target_for_dc = target_seq  # 已经是4D
                
            dc_loss = self.dc_loss(pred_for_dc, target_for_dc, batch['h_params'])
        
        # 总损失
        total_loss = ar_loss + spectral_loss + dc_loss
        
        return total_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_yaml(self.config),
            'metrics_history': self.metrics_history,
            'curriculum_stage': self.curriculum_stage,
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / "last.ckpt")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.ckpt")
            self.logger.info(f"Best checkpoint saved at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics_history = checkpoint['metrics_history']
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.train.max_epochs):
            self.current_epoch = epoch
            
            # 更新课程学习
            self._update_curriculum()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            if epoch % self.config.experiment.val_check_interval == 0:
                val_metrics = self.validate()
                
                # 更新指标历史
                self.metrics_history['train_loss'].append(train_metrics['loss'])
                self.metrics_history['val_loss'].append(val_metrics['loss'])
                self.metrics_history['train_rel_l2'].append(train_metrics['rel_l2'])
                self.metrics_history['val_rel_l2'].append(val_metrics['rel_l2'])
                self.metrics_history['train_mae'].append(train_metrics['mae'])
                self.metrics_history['val_mae'].append(val_metrics['mae'])
                self.metrics_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # 检查是否是最佳模型
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # 保存检查点
                self.save_checkpoint(is_best)
                
                # 可视化训练过程
                if (self.visualizer and 
                    hasattr(self.config, 'visualization') and
                    self.config.visualization.get('training', {}).get('plot_losses', True)):
                    self.visualizer.plot_training_curves(self.metrics_history, epoch)
                
                # 日志记录
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_rel_l2={val_metrics['rel_l2']:.4f}, "
                    f"best_val_loss={self.best_val_loss:.4f}"
                )
                
                # 早停检查
                if (hasattr(self.config.experiment, 'early_stopping') and
                    self.early_stopping_counter >= self.config.experiment.early_stopping.patience):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # 最终评估和可视化
        self._final_evaluation()
    
    def _final_evaluation(self):
        """最终评估和可视化"""
        self.logger.info("Performing final evaluation...")
        
        # 加载最佳模型
        if (self.output_dir / "best.ckpt").exists():
            self.load_checkpoint(str(self.output_dir / "best.ckpt"))
        
        # 测试集评估
        test_metrics = self.validate()  # 使用验证函数评估测试集
        
        # 保存最终结果
        results = {
            'test_metrics': test_metrics,
            'training_history': self.metrics_history,
            'config': OmegaConf.to_yaml(self.config),
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
        }
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 生成最终可视化
        if self.visualizer:
            self.visualizer.create_final_visualizations(
                self.model, self.test_loader, self.device
            )
        
        self.logger.info(f"Final test metrics: {test_metrics}")
        self.logger.info(f"Results saved to {self.output_dir}")


@hydra.main(version_base=None, config_path="configs/experiment", config_name="temporal_training")
def main(config: DictConfig) -> None:
    """主函数"""
    # 打印配置信息
    exp_name = config.experiment.name if hasattr(config.experiment, 'name') else config.experiment
    print(f"实验名称: {exp_name}")
    print(f"数据路径: {config.data_path}")
    print(f"模型: {config.model.name}")
    print(f"时序设置: T_in={config.temporal.T_in}, T_out={config.temporal.T_out}")
    print(f"设备: {config.experiment.device if hasattr(config.experiment, 'device') else 'cuda'}")
    print(f"输出目录: {config.experiment.output_dir if hasattr(config.experiment, 'output_dir') else 'runs'}")
    print("=" * 80)
    print(f"设备: {config.experiment.device}")
    print(f"输出目录: {config.experiment.output_dir}")
    print("=" * 80)
    
    try:
        # 创建训练器
        trainer = TemporalTrainer(config)
        
        # 开始训练
        trainer.train()
        
        print("✅ 训练完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()