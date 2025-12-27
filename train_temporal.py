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
from utils.metrics import compute_metrics, MetricsCalculator
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
        
        model_config = self.config.model.copy()
        model_config.ar_config = self.config.temporal.ar

        img_size = model_config.get("img_size", None)
        if isinstance(img_size, (list, tuple)):
            model_config.img_size = img_size[0]

        self.model = create_model(model_config)
        
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
        # 训练阶段用轻量指标，避免 compute_metrics(SSIM/FFT) 过慢
        self.metric_calc = MetricsCalculator(image_size=(256, 256))
    
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
    
    def _forward_model(self, input_seq: torch.Tensor) -> Any:
        """
        优先尝试 5D 输入（AR/时序模型），失败则回退 4D（单帧模型）。
        input_seq: [B,T,C,H,W] 或 [B,C,H,W]
        """
        if input_seq.ndim == 5:
            try:
                return self.model(input_seq)
            except Exception:
                return self.model(input_seq[:, -1])
        return self.model(input_seq)

    def _compute_light_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        训练阶段轻量指标：rel_l2 / mae
        - 自动把 5D -> 取最后一帧
        - 自动把 pred 插值到 target 空间尺寸
        - 返回 float（全 batch & 全通道平均）
        """
        if pred.ndim == 5:
            pred = pred[:, -1]
        if target.ndim == 5:
            target = target[:, -1]

        # 对齐空间分辨率（只插值 pred）
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        rel_l2_bc = self.metric_calc.compute_rel_l2(pred, target)  # [B,C]
        mae_bc = self.metric_calc.compute_mae(pred, target)        # [B,C]

        rel_l2 = float(rel_l2_bc.mean().item())
        mae = float(mae_bc.mean().item())
        return rel_l2, mae

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
                    outputs = self._forward_model(input_seq)
                    
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
                outputs = self._forward_model(input_seq)
                
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
                pred_for_metrics = predictions['predictions']
                target_for_metrics = target_seq
                
                rel_l2, mae = self._compute_light_metrics(pred_for_metrics, target_for_metrics)
                epoch_metrics["rel_l2"].append(rel_l2)
                epoch_metrics["mae"].append(mae)

                # 统一一个 metrics dict 供后续日志/进度条使用（float）
                metrics = {"rel_l2": rel_l2, "mae": mae}
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rel_l2": f"{metrics['rel_l2']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
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
                    f"rel_l2={metrics['rel_l2']:.4f}, "
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
    
    def evaluate(self, loader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        self.model.eval()
        losses = []
        metrics_acc = {"rel_l2": [], "mae": [], "psnr": [], "ssim": []}

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_seq = batch["input_sequence"].to(self.device)
                target_seq = batch["target_sequence"].to(self.device)
                if "observation_sequence" in batch:
                    input_seq = batch["observation_sequence"].to(self.device)

                outputs = self._forward_model(input_seq)
                predictions = outputs if isinstance(outputs, dict) else {"predictions": outputs}

                loss = self._compute_loss(predictions, target_seq, batch)
                losses.append(loss.item())

                pred_for_metrics = predictions["predictions"]
                if pred_for_metrics.ndim == 4 and target_seq.ndim == 5:
                    target_for_metrics = target_seq[:, -1]
                else:
                    target_for_metrics = target_seq

                m = compute_metrics(pred_for_metrics, target_for_metrics)
                for k in metrics_acc:
                    if k in m:
                        v = m[k]
                        if isinstance(v, torch.Tensor):
                            v = v.mean().detach().cpu().item() if v.numel() > 1 else v.detach().cpu().item()
                        metrics_acc[k].append(v)

        out = {"loss": float(np.mean(losses))}
        for k, vals in metrics_acc.items():
            if vals:
                out[k] = float(np.mean(vals))
        return out

    def validate(self) -> Dict[str, float]:
        """验证"""
        return self.evaluate(self.val_loader, desc="Validating")
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     target_seq: torch.Tensor, batch: Dict) -> torch.Tensor:
        """计算损失"""
        predictions = outputs['predictions']  # [B, C, H, W] 或 [B, T_out, C, H, W]
        
        # ---- 统一成 5D: [B,T,C,H,W]，并与 target 对齐 ----
        if predictions.ndim == 4:
            # 单帧模型：只监督 target 最后一帧，严禁 repeat 成 T_out
            predictions = predictions.unsqueeze(1)  # [B,1,C,H,W]
            if target_seq.ndim == 5:
                target_seq = target_seq[:, -1:].contiguous()  # [B,1,C,H,W]
            else:
                target_seq = target_seq.unsqueeze(1)
        
        elif predictions.ndim == 5:
            if target_seq.ndim == 4:
                target_seq = target_seq.unsqueeze(1)  # [B,1,C,H,W]
        
            # 时序长度不一致：截断到最小 T
            if predictions.shape[1] != target_seq.shape[1]:
                T = min(predictions.shape[1], target_seq.shape[1])
                predictions = predictions[:, :T]
                target_seq = target_seq[:, :T]
        else:
            raise ValueError(f"Unsupported predictions.ndim={predictions.ndim}")
        
        # ---- 对齐空间分辨率 ----
        if predictions.shape[-2:] != target_seq.shape[-2:]:
            target_size = target_seq.shape[-2:]
            B, T, C = predictions.shape[:3]
            pred_bt = predictions.reshape(B * T, C, *predictions.shape[-2:])
            pred_bt = F.interpolate(pred_bt, size=target_size, mode="bilinear", align_corners=False)
            predictions = pred_bt.reshape(B, T, C, *target_size)
        
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
        if "h_params" in batch:
            pred_for_dc = predictions[:, -1]
            target_for_dc = target_seq[:, -1]
            dc_loss = self.dc_loss(pred_for_dc, target_for_dc, batch["h_params"])
        
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
        test_metrics = self.evaluate(self.test_loader, desc="Testing")
        
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
