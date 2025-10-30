#!/usr/bin/env python3
"""
时序NAR模型100轮训练脚本
基于现有train_temporal.py，支持AR+NAR双头训练
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
import h5py

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.temporal_pdebench import TemporalPDEBenchDataModule
from models.base import create_model
from ops.losses import ARLoss, SpectralLoss, DCLoss
from utils.metrics import compute_metrics
from utils.visualization import TemporalVisualizer
from utils.logger import setup_logger


class TemporalNARTrainer:
    """时序NAR模型训练器 - 100轮训练版本"""
    
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
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        self.logger.info(f"TemporalNARTrainer initialized. Output dir: {self.output_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_data(self):
        """初始化数据模块"""
        self.logger.info("初始化数据模块...")
        
        # 使用PDEBench数据模块而不是时序版本
        from datasets import PDEBenchDataModule
        self.data_module = PDEBenchDataModule(self.config.data)
        
        # 设置数据集
        self.data_module.setup()
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        self.logger.info(f"数据加载完成. Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
    
    def _init_model(self):
        """初始化模型"""
        self.logger.info("初始化模型...")
        
        # 使用现有的模型创建方式
        from models import create_model as create_model_init
        
        # 获取数据通道数 - diffusion-reaction数据每个case有2个通道(u,v)
        num_channels = 2  # 固定为2个通道
        
        # 构建模型参数
        model_kwargs = {
            'in_channels': num_channels,  # 根据数据键数量确定通道数
            'out_channels': num_channels,
            'img_size': 128,
        }
        
        # 创建模型
        self.model = create_model_init("swin_unet", **model_kwargs)
        self.model = self.model.to(self.device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成: SwinUNet")
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        # 使用默认的优化器配置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 余弦退火调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # 100轮训练
            eta_min=1e-6
        )
        
        self.logger.info(f"优化器: AdamW, LR: 1e-3")
    
    def _init_losses(self):
        """初始化损失函数"""
        # 使用MSE作为主要损失
        self.criterion = nn.MSELoss()
        self.logger.info("损失函数初始化完成")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/100")
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            if isinstance(batch, dict):
                # 从batch中获取数据
                x = batch['observation'].to(self.device)  # 观测数据作为输入
                y = batch['target'].to(self.device)       # 目标数据
            else:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{total_loss/num_batches:.6f}'
            })
            
            # 每100步记录一次
            if batch_idx % 100 == 0:
                self.logger.info(f"Epoch {self.current_epoch+1}, Step {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 获取数据
                if isinstance(batch, dict):
                    x = batch['observation'].to(self.device)
                    y = batch['target'].to(self.device)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                
                # 前向传播
                if self.use_amp:
                    with autocast():
                        pred = self.model(x)
                        loss = self.criterion(pred, y)
                else:
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_yaml(self.config)
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pth')
            self.logger.info(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始100轮训练...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(100):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/100 - Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 保存检查点
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # 早停检查
            if self.early_stopping_counter >= 20:
                self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses)
        
        self.logger.info(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
    
    def plot_training_curves(self, train_losses, val_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(train_losses) + 1)
        lrs = [1e-3 * (1e-6/1e-3) ** (epoch/100) for epoch in epochs]  # 余弦退火近似
        plt.plot(epochs, lrs, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("训练曲线已保存")


@hydra.main(version_base=None, config_path="configs/experiment", config_name="temporal_nar_100epochs")
def main(cfg: DictConfig) -> None:
    """主函数"""
    print("配置文件内容:")
    print(OmegaConf.to_yaml(cfg))
    
    # 创建训练器
    trainer = TemporalNARTrainer(cfg)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()