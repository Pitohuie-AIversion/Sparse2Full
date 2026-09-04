#!/usr/bin/env python3
"""
直接训练时序NAR模型脚本
避免复杂的配置文件结构问题，直接使用代码配置
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.temporal_pdebench import TemporalPDEBenchDataModule
from models.swin_unet import SwinUNet
from ops.losses import compute_total_loss
from ops.metrics import compute_metrics

class TemporalNARTrainer:
    """时序NAR模型训练器"""
    
    def __init__(self):
        self.setup_config()
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_monitoring()
        
    def setup_config(self):
        """设置配置"""
        self.config = {
            # 实验配置
            'experiment_name': 'TemporalNAR-DR2D-128-300epochs-direct',
            'seed': 2025,
            'epochs': 300,
            
            # 数据配置
            'data': {
                'data_path': r'data/DR2D/2D_diff-react_NA_NA.h5',
                'dataset_name': '2D_diff-react_NA_NA',
                'batch_size': 4,
                'image_size': 128,
                'task': 'Crop',
                'crop_ratio': 0.2,
                'num_workers': 4,
                'pin_memory': True,
                'use_official_format': False,
                'keys': [f'{i:04d}' for i in range(21)],  # 0000-0020
                'temporal': {
                    'T_in': 4,
                    'T_out': 20,
                    'dt': 0.1,
                    'ar': {
                        'teacher_forcing_ratio': 0.8,
                        'scheduled_sampling': True,
                        'sampling_decay': 0.99
                    }
                }
            },
            
            # 模型配置
            'model': {
                'in_channels': 2,
                'out_channels': 2,
                'img_size': 128,
                'patch_size': 4,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'norm_layer': 'LayerNorm',
                'patch_norm': True,
                'use_checkpoint': False,
                'temporal': {
                    'enabled': False
                },
                'nar_head': {
                    'enabled': True,
                    'max_timesteps': 32,
                    'time_embed_dim': 128,
                    'use_time_encoder': True
                }
            },
            
            # 训练配置
            'training': {
                'lr': 0.001,
                'weight_decay': 0.0001,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'scheduler': {
                    'T_0': 50,
                    'T_mult': 2,
                    'eta_min': 1e-6,
                    'warmup_epochs': 10
                },
                'amp': True,
                'gradient_clipping': 1.0,
                'ar': {
                    'enabled': True,
                    'loss_weight': 1.0,
                    'warmup_epochs': 20
                },
                'nar': {
                    'enabled': True,
                    'loss_weight': 1.0,
                    'warmup_epochs': 50
                }
            },
            
            # 损失配置
            'loss': {
                'reconstruction_weight': 1.0,
                'spectral_weight': 0.5,
                'data_consistency_weight': 1.0,
                'gradient_weight': 0.1
            },
            
            # 输出配置
            'output': {
                'base_dir': 'runs',
                'exp_dir': 'temporal_nar_direct_300epochs',
                'save_every_n_epochs': 10,
                'visualize_every_n_epochs': 20
            }
        }
        
        # 设置随机种子
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def setup_logging(self):
        """设置日志"""
        self.exp_dir = Path(self.config['output']['base_dir']) / self.config['output']['exp_dir']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.exp_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 保存配置
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def setup_device(self):
        """设置设备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
    def setup_data(self):
        """设置数据"""
        self.logger.info("Setting up data module...")
        
        # 创建数据模块
        self.data_module = TemporalPDEBenchDataModule(self.config['data'])
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        self.logger.info(f"Data loaded: train={len(self.train_loader)}, val={len(self.val_loader)}")
        
    def setup_model(self):
        """设置模型"""
        self.logger.info("Setting up model...")
        
        # 创建模型
        self.model = SwinUNet(**self.config['model'])
        self.model = self.model.to(self.device)
        
        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['training']['betas'],
            eps=self.config['training']['eps']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['scheduler']['T_0'],
            T_mult=self.config['training']['scheduler']['T_mult'],
            eta_min=self.config['training']['scheduler']['eta_min']
        )
        
        # AMP scaler
        if self.config['training']['amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def setup_loss(self):
        """设置损失函数"""
        # 使用ops.losses中的compute_total_loss函数
        self.loss_weights = {
            'reconstruction': self.config['loss']['reconstruction_weight'],
            'spectral': self.config['loss']['spectral_weight'],
            'data_consistency': self.config['loss']['data_consistency_weight'],
            'gradient': self.config['loss']['gradient_weight']
        }
        
    def setup_monitoring(self):
        """设置监控"""
        # TensorBoard
        self.writer = SummaryWriter(self.exp_dir / 'logs')
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_rel_l2': [],
            'val_mae': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr': []
        }
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            x = batch['input'].to(self.device)  # [B, T_in, C, H, W]
            y = batch['target'].to(self.device)  # [B, T_out, C, H, W]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    # 计算损失
                    loss_dict = compute_total_loss(
                        pred_z=pred,
                        target_z=y,
                        obs_data={'input': x},
                        norm_stats=getattr(self.data_module, 'norm_stats', None),
                        config=self.config
                    )
                    loss = loss_dict['total']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                # 计算损失
                loss_dict = compute_total_loss(
                    pred_z=pred,
                    target_z=y,
                    obs_data={'input': x},
                    norm_stats=getattr(self.data_module, 'norm_stats', None),
                    config=self.config
                )
                loss = loss_dict['total']
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 数据移动到设备
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                
                # 前向传播
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x)
                        # 计算损失
                        loss_dict = compute_total_loss(
                            pred_z=pred,
                            target_z=y,
                            obs_data={'input': x},
                            norm_stats=getattr(self.data_module, 'norm_stats', None),
                            config=self.config
                        )
                        loss = loss_dict['total']
                else:
                    pred = self.model(x)
                    # 计算损失
                    loss_dict = compute_total_loss(
                        pred_z=pred,
                        target_z=y,
                        obs_data={'input': x},
                        norm_stats=getattr(self.data_module, 'norm_stats', None),
                        config=self.config
                    )
                    loss = loss_dict['total']
                
                total_loss += loss.item()
                
                # 计算指标
                metrics = compute_metrics(pred, y)
                all_metrics.append(metrics)
        
        # 聚合指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.exp_dir / 'latest_checkpoint.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_checkpoint.pth')
            
        # 定期保存检查点
        if (epoch + 1) % self.config['output']['save_every_n_epochs'] == 0:
            torch.save(checkpoint, self.exp_dir / f'checkpoint_epoch_{epoch+1:03d}.pth')
    
    def visualize_predictions(self, epoch):
        """可视化预测结果"""
        if (epoch + 1) % self.config['output']['visualize_every_n_epochs'] != 0:
            return
            
        self.model.eval()
        
        # 获取一个验证批次
        val_batch = next(iter(self.val_loader))
        x = val_batch['input'].to(self.device)
        y = val_batch['target'].to(self.device)
        
        with torch.no_grad():
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
            else:
                pred = self.model(x)
        
        # 创建可视化
        viz_dir = self.exp_dir / 'visualizations' / f'epoch_{epoch+1:03d}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择第一个样本进行可视化
        sample_idx = 0
        x_sample = x[sample_idx].cpu().numpy()  # [T_in, C, H, W]
        y_sample = y[sample_idx].cpu().numpy()  # [T_out, C, H, W]
        pred_sample = pred[sample_idx].cpu().numpy()  # [T_out, C, H, W]
        
        # 创建简单的可视化
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, min(5, y_sample.shape[0]), figsize=(15, 9))
        if y_sample.shape[0] == 1:
            axes = axes.reshape(3, 1)
        
        for t in range(min(5, y_sample.shape[0])):
            # 真值
            axes[0, t].imshow(y_sample[t, 0], cmap='viridis')
            axes[0, t].set_title(f'GT t={t}')
            axes[0, t].axis('off')
            
            # 预测
            axes[1, t].imshow(pred_sample[t, 0], cmap='viridis')
            axes[1, t].set_title(f'Pred t={t}')
            axes[1, t].axis('off')
            
            # 误差
            error = np.abs(y_sample[t, 0] - pred_sample[t, 0])
            axes[2, t].imshow(error, cmap='hot')
            axes[2, t].set_title(f'Error t={t}')
            axes[2, t].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'prediction_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {viz_dir}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_rel_l2'].append(val_metrics.get('rel_l2', 0))
            self.train_history['val_mae'].append(val_metrics.get('mae', 0))
            self.train_history['val_psnr'].append(val_metrics.get('psnr', 0))
            self.train_history['val_ssim'].append(val_metrics.get('ssim', 0))
            self.train_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 可视化
            self.visualize_predictions(epoch)
            
            # 日志输出
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Rel-L2: {val_metrics.get('rel_l2', 0):.4f}, "
                f"Val MAE: {val_metrics.get('mae', 0):.4f}, "
                f"Best Epoch: {self.best_epoch}"
            )
            
            # 保存训练历史
            with open(self.exp_dir / 'train_history.json', 'w') as f:
                json.dump(self.train_history, f, indent=2)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        # 关闭TensorBoard writer
        self.writer.close()

def main():
    """主函数"""
    trainer = TemporalNARTrainer()
    trainer.train()

if __name__ == '__main__':
    main()