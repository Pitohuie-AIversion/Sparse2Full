#!/usr/bin/env python3
"""
改进训练脚本 - 集成所有优化措施解决loss停滞问题
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback
import random
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import psutil

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
training_dir = Path(__file__).resolve().parent
for path in (project_root, training_dir):
    p = str(path)
    if p in sys.path:
        try:
            sys.path.remove(p)
        except Exception:
            pass
    sys.path.insert(0, p)

try:
    from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
except Exception:
    from datasets.real_dr_dataset import RealDiffusionReactionDataModule
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper
from ops.losses import compute_total_loss, compute_ar_total_loss
from ops.enhanced_losses import compute_enhanced_total_loss
from ops.enhanced_augmentation import AdvancedDataAugmentation
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from ops.degradation import SuperResolutionOperator, CropOperator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class ImprovedTrainer:
    """改进训练器 - 集成所有优化措施"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.device.accelerator if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        self._set_random_seeds(config.seed)
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir) / config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name=config.experiment.name,
            log_file=self.output_dir / "training.log",
            level=logging.INFO
        )
        
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_functions()
        self._setup_augmentation()
        self._setup_monitoring()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        self.logger.info(f"Improved trainer initialized for {config.experiment.name}")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """设置数据加载器"""
        self.data_module = RealDiffusionReactionDataModule(self.config)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
        
        # 获取归一化统计信息
        self.norm_stats = self.data_module.get_normalization_stats()
        
        self.logger.info(f"Data setup complete - Train: {len(self.train_loader.dataset)}, "
                        f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
    
    def _setup_model(self):
        """设置模型"""
        model_config = self.config.model
        
        # 创建基础模型
        self.base_model = SwinUNet(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            img_size=model_config.img_size,
            patch_size=model_config.patch_size,
            window_size=model_config.window_size,
            depths=model_config.depths,
            num_heads=model_config.num_heads,
            embed_dim=model_config.embed_dim,
            mlp_ratio=model_config.mlp_ratio,
            drop_rate=model_config.drop_rate,
            attn_drop_rate=model_config.attn_drop_rate,
            drop_path_rate=model_config.drop_path_rate
        ).to(self.device)
        
        # 包装为AR模型（如果需要）
        if self.config.ar.enabled:
            self.model = ARWrapper(self.base_model, self.config.ar)
        else:
            self.model = self.base_model
        
        # 使用DataParallel（如果有多个GPU）
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        self.model.to(self.device)
        
        self.logger.info(f"Model setup complete - Parameters: {self._count_parameters():,}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        optimizer_config = self.config.training.optimizer
        
        # 使用分组学习率：主干网络较低LR，头部较高LR
        base_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'head' in name or 'fc' in name or 'classifier' in name:
                    head_params.append(param)
                else:
                    base_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': optimizer_config.lr * 0.5},  # 主干网络较低LR
            {'params': head_params, 'lr': optimizer_config.lr}        # 头部正常LR
        ], lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas, eps=optimizer_config.eps)
        
        self.logger.info(f"Optimizer setup complete - Base LR: {optimizer_config.lr * 0.5}, "
                        f"Head LR: {optimizer_config.lr}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_config = self.config.training.scheduler
        total_epochs = self.config.training.epochs
        
        # 余弦退火调度器
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - scheduler_config.warmup_epochs,
            eta_min=scheduler_config.eta_min
        )
        
        # 线性预热调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=scheduler_config.warmup_epochs
        )
        
        # 组合调度器
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[scheduler_config.warmup_epochs]
        )
    
    def _setup_loss_functions(self):
        """设置损失函数"""
        # 使用增强损失函数
        self.loss_fn = compute_enhanced_total_loss
        self.logger.info("Enhanced loss functions setup complete")
    
    def _setup_augmentation(self):
        """设置数据增强"""
        if self.config.data.augmentation.enabled:
            self.augmentation = AdvancedDataAugmentation(self.config.data.augmentation)
            self.logger.info("Advanced data augmentation setup complete")
        else:
            self.augmentation = None
    
    def _setup_monitoring(self):
        """设置监控"""
        self.monitor_dir = self.output_dir / "monitoring"
        self.monitor_dir.mkdir(exist_ok=True)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.gradient_norms = []
    
    def _build_obs_data(self, baseline: torch.Tensor) -> Dict[str, Any]:
        obs_cfg = getattr(self.config.data, "observation", None)
        if obs_cfg is None:
            return {"observation": None, "baseline": baseline, "h_params": None}
        try:
            mode_raw = obs_cfg.get("mode", "sr")
            mode = str(mode_raw[0] if isinstance(mode_raw, (list, tuple)) else mode_raw).lower()
        except Exception:
            mode = "sr"
        if mode == "sr":
            sr_sub = obs_cfg.get("sr", {}) if isinstance(obs_cfg.get("sr", {}), dict) else {}
            scale = obs_cfg.get("scale_factor", sr_sub.get("scale_factor", 2))
            sigma = obs_cfg.get("blur_sigma", sr_sub.get("blur_sigma", 1.0))
            kernel_size = obs_cfg.get("kernel_size", sr_sub.get("blur_kernel_size", 5))
            boundary = obs_cfg.get("boundary", sr_sub.get("boundary_mode", "mirror"))
            downsample = obs_cfg.get("downsample_interpolation", sr_sub.get("downsample_mode", "area"))
            h_params = {
                "task": "SR",
                "scale": int(scale),
                "sigma": float(sigma),
                "kernel_size": int(kernel_size),
                "boundary": str(boundary),
                "downsample_interpolation": str(downsample),
            }
        elif mode == "crop":
            crop_sub = obs_cfg.get("crop", {}) if isinstance(obs_cfg.get("crop", {}), dict) else {}
            crop_size = obs_cfg.get("crop_size", crop_sub.get("crop_size"))
            crop_box = obs_cfg.get("crop_box", crop_sub.get("crop_box"))
            boundary = obs_cfg.get("boundary", crop_sub.get("boundary_mode", "mirror"))
            h_params = {
                "task": "Crop",
                "crop_size": crop_size,
                "crop_box": crop_box,
                "boundary": str(boundary),
            }
        else:
            h_params = None
        return {"observation": None, "baseline": baseline, "h_params": h_params}
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 更新增强强度
        if self.augmentation:
            self.augmentation.update_epoch(epoch)
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'spectral_loss': 0.0,
            'dc_loss': 0.0,
            'gradient_loss': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            if self.config.ar.enabled:
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)
                x = input_seq
                target = target_seq
            else:
                x = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
            
            # 应用数据增强
            if self.augmentation and self.config.data.augmentation.enabled:
                x, target = self.augmentation(x, target, epoch=epoch, mode='train')
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.config.training.amp.enabled:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred = self.model(x)
                    obs_data = self._build_obs_data(x)
                    losses = self.loss_fn(
                        pred_z=pred,
                        target_z=target,
                        obs_data=obs_data,
                        norm_stats=self.norm_stats,
                        config=self.config,
                        epoch=epoch
                    )
            else:
                pred = self.model(x)
                obs_data = self._build_obs_data(x)
                losses = self.loss_fn(
                    pred_z=pred,
                    target_z=target,
                    obs_data=obs_data,
                    norm_stats=self.norm_stats,
                    config=self.config,
                    epoch=epoch
                )
            
            loss = losses['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.gradient_clip_val
            )
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 记录梯度范数
            total_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            self.gradient_norms.append(total_grad_norm)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'grad_norm': f'{total_grad_norm:.2f}'
            })
            
            # 每10个batch记录一次详细损失
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                               f"Total Loss: {loss.item():.4f}, "
                               f"Rec: {losses.get('reconstruction_loss', 0):.4f}, "
                               f"Spec: {losses.get('spectral_loss', 0):.4f}, "
                               f"DC: {losses.get('dc_loss', 0):.4f}, "
                               f"Grad: {losses.get('gradient_loss', 0):.4f}")
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        avg_losses['learning_rate'] = self.optimizer.param_groups[0]['lr']
        avg_losses['avg_gradient_norm'] = np.mean(self.gradient_norms[-num_batches:])
        
        return avg_losses
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = len(self.val_loader)
        
        val_metrics = {
            'val_loss': 0.0,
            'rel_l2': 0.0,
            'mae': 0.0,
            'rmse': 0.0,
            'psnr': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 获取数据
                if self.config.ar.enabled:
                    input_seq = batch['input'].to(self.device)
                    target_seq = batch['target'].to(self.device)
                    x = input_seq
                    target = target_seq
                else:
                    x = batch['input'].to(self.device)
                    target = batch['target'].to(self.device)
                
                # 前向传播
                if self.config.training.amp.enabled:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        pred = self.model(x)
                        
                        # 准备观测数据
                        obs_data = {
                            'observation': None,
                            'baseline': x,
                            'h_params': {'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'}
                        }
                        
                        # 计算验证损失
                        losses = self.loss_fn(
                            pred_z=pred,
                            target_z=target,
                            obs_data=obs_data,
                            norm_stats=self.norm_stats,
                            config=self.config,
                            epoch=epoch
                        )
                else:
                    pred = self.model(x)
                    
                    # 准备观测数据
                    obs_data = {
                        'observation': None,
                        'baseline': x,
                        'h_params': {'task': 'SR', 'scale': 2, 'sigma': 1.0, 'kernel_size': 5, 'boundary': 'mirror'}
                    }
                    
                    # 计算验证损失
                    losses = self.loss_fn(
                        pred_z=pred,
                        target_z=target,
                        obs_data=obs_data,
                        norm_stats=self.norm_stats,
                        config=self.config,
                        epoch=epoch
                    )
                
                val_loss = losses['total_loss']
                total_val_loss += val_loss.item()
                
                # 计算指标
                metrics = compute_metrics(pred, target, self.norm_stats)
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key] += metrics[key]
        
        # 计算平均指标
        avg_val_loss = total_val_loss / num_val_batches
        for key in val_metrics:
            if key != 'val_loss':
                val_metrics[key] /= num_val_batches
        val_metrics['val_loss'] = avg_val_loss
        
        self.logger.info(f"Validation - Epoch {epoch}: "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Rel L2: {val_metrics['rel_l2']:.4f}, "
                        f"MAE: {val_metrics['mae']:.4f}, "
                        f"RMSE: {val_metrics['rmse']:.4f}, "
                        f"PSNR: {val_metrics['psnr']:.2f}")
        
        return val_metrics
    
    def train(self):
        """主训练循环"""
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['total_loss'])
            self.learning_rates.append(train_metrics['learning_rate'])
            
            # 验证
            if epoch % self.config.validation.check_val_every_n_epoch == 0:
                val_metrics = self.validate(epoch)
                self.val_losses.append(val_metrics['val_loss'])
                
                # 早停检查
                if val_metrics['val_loss'] < self.best_val_loss - self.config.training.early_stopping.min_delta:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"New best model saved - Val Loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1
                
                # 检查早停
                if self.patience_counter >= self.config.training.early_stopping.patience:
                    self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs")
                    break
            
            # 更新学习率
            self.scheduler.step()
            
            # 定期保存检查点
            if epoch % self.config.checkpoint.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)
            
            # 记录训练进度
            self.logger.info(f"Epoch {epoch} completed - "
                           f"Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"LR: {train_metrics['learning_rate']:.2e}, "
                           f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(self.current_epoch, is_last=True)
        
        # 生成训练报告
        self.generate_training_report()
        
        self.logger.info("Training completed successfully!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'gradient_norms': self.gradient_norms
            }
        }
        
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pth"
        elif is_last:
            checkpoint_path = self.output_dir / "last_model.pth"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def generate_training_report(self):
        """生成训练报告"""
        report_path = self.output_dir / "training_report.json"
        
        report = {
            'experiment_name': self.config.experiment.name,
            'final_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'total_epochs_trained': len(self.train_losses),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'gradient_norms': self.gradient_norms
            },
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Training report saved: {report_path}")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.config.experiment.name}', fontsize=16)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.7)
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            ax.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax = axes[0, 1]
        ax.plot(epochs, self.learning_rates, 'g-', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # 梯度范数
        ax = axes[1, 0]
        if self.gradient_norms:
            ax.plot(range(1, len(self.gradient_norms) + 1), self.gradient_norms, 'purple', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Evolution')
            ax.grid(True, alpha=0.3)
        
        # 损失下降率
        ax = axes[1, 1]
        if len(self.train_losses) > 1:
            loss_diff = np.diff(self.train_losses)
            ax.plot(range(2, len(self.train_losses) + 1), loss_diff, 'orange', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Change')
            ax.set_title('Loss Decrease Rate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Improved Training Script')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建改进训练器
    trainer = ImprovedTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
