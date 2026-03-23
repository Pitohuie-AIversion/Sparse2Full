#!/usr/bin/env python3
"""
增强训练监控脚本 - 解决loss停滞问题
提供详细的训练指标、梯度分析和损失可视化
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
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from ops.degradation import SuperResolutionOperator, CropOperator


class EnhancedTrainingMonitor:
    """增强训练监控器 - 提供详细的训练分析和可视化"""
    
    def __init__(self, config: DictConfig, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        
        # 监控历史
        self.train_loss_history = []
        self.val_loss_history = []
        self.learning_rate_history = []
        self.gradient_norm_history = []
        self.weight_norm_history = []
        self.loss_components_history = []
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_triggered = False
        
        # 创建监控目录
        self.monitor_dir = Path(config.output_dir) / config.experiment.name / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name=f"enhanced_monitor_{config.experiment.name}",
            log_file=self.monitor_dir / "training_monitor.log",
            level=logging.INFO
        )
        
        self.logger.info(f"EnhancedTrainingMonitor initialized for {config.experiment.name}")
    
    def log_training_step(self, epoch: int, batch_idx: int, loss_dict: Dict[str, float], 
                         gradients: Optional[Dict[str, torch.Tensor]] = None):
        """记录训练步骤的详细指标"""
        step_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'step': epoch * 1000 + batch_idx,  # 近似步数
            'timestamp': datetime.now().isoformat(),
            'losses': loss_dict,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'gpu_memory_cached_mb': torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        # 记录梯度信息
        if gradients is not None:
            grad_norms = {}
            for name, grad in gradients.items():
                if grad is not None:
                    grad_norms[name] = torch.norm(grad).item()
            step_info['gradient_norms'] = grad_norms
            
            # 计算总梯度范数
            total_grad_norm = torch.norm(torch.cat([g.view(-1) for g in gradients.values() if g is not None]))
            step_info['total_gradient_norm'] = total_grad_norm.item()
            self.gradient_norm_history.append(total_grad_norm.item())
        
        # 记录权重信息
        weight_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_norms[name] = torch.norm(param).item()
        step_info['weight_norms'] = weight_norms
        
        # 更新历史记录
        if 'total_loss' in loss_dict:
            self.train_loss_history.append(loss_dict['total_loss'])
        self.learning_rate_history.append(step_info['learning_rate'])
        
        # 每100步记录一次详细信息
        if batch_idx % 100 == 0:
            self.logger.info(f"Epoch {epoch}, Batch {batch_idx}: {json.dumps(step_info, indent=2, default=str)}")
    
    def log_validation(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]):
        """记录验证结果"""
        self.val_loss_history.append(val_loss)
        
        val_info = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat(),
            'improvement': val_loss < self.best_val_loss
        }
        
        # 检查是否有改进
        if val_loss < self.best_val_loss - self.config.training.early_stopping.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            val_info['best_model'] = True
        else:
            self.patience_counter += 1
            val_info['patience_counter'] = self.patience_counter
        
        self.logger.info(f"Validation - Epoch {epoch}: {json.dumps(val_info, indent=2, default=str)}")
        
        # 检查早停
        if self.patience_counter >= self.config.training.early_stopping.patience:
            self.early_stop_triggered = True
            self.logger.warning(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if not self.train_loss_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Monitoring - {self.config.experiment.name}', fontsize=16)
        
        # 损失曲线
        ax = axes[0, 0]
        epochs = range(1, len(self.train_loss_history) + 1)
        ax.plot(epochs, self.train_loss_history, 'b-', label='Training Loss', alpha=0.7)
        if self.val_loss_history:
            val_epochs = range(1, len(self.val_loss_history) + 1)
            ax.plot(val_epochs, self.val_loss_history, 'r-', label='Validation Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax = axes[0, 1]
        ax.plot(epochs, self.learning_rate_history, 'g-', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # 梯度范数
        ax = axes[1, 0]
        if self.gradient_norm_history:
            ax.plot(range(1, len(self.gradient_norm_history) + 1), self.gradient_norm_history, 'purple', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Evolution')
            ax.grid(True, alpha=0.3)
        
        # 损失分量分析
        ax = axes[1, 1]
        if self.loss_components_history:
            components = list(self.loss_components_history[0].keys())
            for comp in components:
                values = [h[comp] for h in self.loss_components_history]
                ax.plot(epochs[:len(values)], values, label=comp, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Component')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.monitor_dir / "training_curves.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to {save_path}")
    
    def analyze_gradient_flow(self, save_path: Optional[str] = None):
        """分析梯度流动情况"""
        if not self.gradient_norm_history:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Gradient Flow Analysis - {self.config.experiment.name}', fontsize=16)
        
        # 梯度范数分布
        ax = axes[0]
        ax.hist(self.gradient_norm_history, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Frequency')
        ax.set_title('Gradient Norm Distribution')
        ax.grid(True, alpha=0.3)
        
        # 梯度变化趋势
        ax = axes[1]
        if len(self.gradient_norm_history) > 1:
            gradient_changes = np.diff(self.gradient_norm_history)
            ax.plot(range(1, len(gradient_changes) + 1), gradient_changes, 'red', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm Change')
            ax.set_title('Gradient Norm Changes')
            ax.grid(True, alpha=0.3)
        
        # 梯度爆炸/消失检测
        ax = axes[2]
        grad_norms = np.array(self.gradient_norm_history)
        explosion_threshold = np.percentile(grad_norms, 95)
        vanishing_threshold = np.percentile(grad_norms, 5)
        
        normal_grads = (grad_norms >= vanishing_threshold) & (grad_norms <= explosion_threshold)
        ax.scatter(np.where(normal_grads)[0], grad_norms[normal_grads], 
                  c='green', alpha=0.6, label='Normal', s=10)
        ax.scatter(np.where(grad_norms > explosion_threshold)[0], 
                  grad_norms[grad_norms > explosion_threshold], 
                  c='red', alpha=0.8, label='Exploding', s=10)
        ax.scatter(np.where(grad_norms < vanishing_threshold)[0], 
                  grad_norms[grad_norms < vanishing_threshold], 
                  c='orange', alpha=0.8, label='Vanishing', s=10)
        
        ax.axhline(y=explosion_threshold, color='red', linestyle='--', alpha=0.7, label=f'95th percentile')
        ax.axhline(y=vanishing_threshold, color='orange', linestyle='--', alpha=0.7, label=f'5th percentile')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Explosion/Vanishing Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.monitor_dir / "gradient_analysis.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gradient analysis saved to {save_path}")
        
        # 返回分析结果
        analysis = {
            'gradient_norm_mean': float(np.mean(grad_norms)),
            'gradient_norm_std': float(np.std(grad_norms)),
            'exploding_gradients_pct': float(np.mean(grad_norms > explosion_threshold) * 100),
            'vanishing_gradients_pct': float(np.mean(grad_norms < vanishing_threshold) * 100),
            'explosion_threshold': float(explosion_threshold),
            'vanishing_threshold': float(vanishing_threshold)
        }
        
        return analysis
    
    def save_monitoring_report(self):
        """保存完整的监控报告"""
        report = {
            'experiment_name': self.config.experiment.name,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'training_history': {
                'train_loss_history': self.train_loss_history,
                'val_loss_history': self.val_loss_history,
                'learning_rate_history': self.learning_rate_history,
                'gradient_norm_history': self.gradient_norm_history,
                'loss_components_history': self.loss_components_history
            },
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_loss_history[-1] if self.train_loss_history else None,
                'early_stopped': self.early_stop_triggered,
                'total_epochs': len(self.train_loss_history)
            }
        }
        
        # 梯度分析
        if self.gradient_norm_history:
            grad_analysis = self.analyze_gradient_flow()
            report['gradient_analysis'] = grad_analysis
        
        # 保存报告
        report_path = self.monitor_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Training report saved to {report_path}")
        
        # 生成图表
        self.plot_training_curves()
        
        return report


def create_enhanced_trainer(config_path: str):
    """创建增强训练器"""
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # 创建数据模块
    data_module = RealDiffusionReactionDataModule(config)
    
    # 创建模型
    model_config = config.model
    base_model = SwinUNet(
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
    )
    
    # 包装为AR模型（如果需要）
    if config.ar.enabled:
        model = ARWrapper(base_model, config.ar)
    else:
        model = base_model
    
    # 创建优化器
    optimizer_config = config.training.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps
    )
    
    # 创建监控器
    monitor = EnhancedTrainingMonitor(config, model, optimizer)
    
    return model, data_module, optimizer, monitor, config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Enhanced Training Monitor')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    
    args = parser.parse_args()
    
    # 创建增强训练器
    model, data_module, optimizer, monitor, config = create_enhanced_trainer(args.config)
    
    print(f"Enhanced training monitor created for experiment: {config.experiment.name}")
    print(f"Monitoring directory: {monitor.monitor_dir}")
    print(f"Training will start with enhanced monitoring...")


if __name__ == "__main__":
    main()