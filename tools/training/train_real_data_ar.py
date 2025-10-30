#!/usr/bin/env python3
"""
真实扩散-反应数据AR训练脚本
专门用于训练真实数据集的20步AR预测模型
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
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import h5py

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from datasets.real_dr_dataset import RealDiffusionReactionDataModule
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper
from ops.losses import compute_total_loss
from utils.metrics import compute_metrics
from utils.logger import setup_logger


class RealDataARTrainer:
    """真实数据AR训练器"""
    
    def __init__(self, config_path: str = None):
        """初始化训练器"""
        self.setup_config(config_path)
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_monitoring()
        
    def setup_config(self, config_path: str = None):
        """设置配置"""
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            # 默认配置
            self.config = DictConfig({
                'experiment': {
                    'name': 'Real-DR2D-AR-T20-128-SwinUNet-s2025',
                    'seed': 2025,
                    'output_dir': 'runs',
                    'device': 'cuda',
                    'precision': '16-mixed',
                    'log_every_n_steps': 10
                },
                'data': {
                    'data_path': 'E:/2D/diffusion-reaction/2D_diff-react_NA_NA.h5',
                    'T_in': 1,
                    'T_out': 20,
                    'img_size': 128,
                    'channels': 2,
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'time_step_start': 0,
                    'time_step_end': 980,
                    'time_step_stride': 1,
                    'normalize': True,
                    'augmentation': {
                        'enabled': True,
                        'flip_prob': 0.5,
                        'rotate_prob': 0.3,
                        'noise_std': 0.01
                    }
                },
                'model': {
                    'name': 'SwinUNet',
                    'in_channels': 2,
                    'out_channels': 2,
                    'img_size': 128,
                    'patch_size': 4,
                    'window_size': 8,
                    'depths': [2, 2, 6, 2],
                    'num_heads': [3, 6, 12, 24],
                    'embed_dim': 96,
                    'mlp_ratio': 4.0,
                    'drop_rate': 0.1,
                    'attn_drop_rate': 0.1,
                    'drop_path_rate': 0.2
                },
                'training': {
                    'epochs': 200,
                    'batch_size': 8,
                    'accumulate_grad_batches': 4,
                    'optimizer': {
                        'name': 'AdamW',
                        'lr': 5e-4,
                        'weight_decay': 1e-4,
                        'betas': [0.9, 0.999]
                    },
                    'scheduler': {
                        'name': 'CosineAnnealingLR',
                        'T_max': 200,
                        'eta_min': 1e-6,
                        'warmup_epochs': 10
                    },
                    'gradient_clip_val': 1.0,
                    'curriculum': {
                        'enabled': True,
                        'stages': [
                            {'epochs': 40, 'T_out': 5, 'description': '阶段1: 预测5步'},
                            {'epochs': 40, 'T_out': 10, 'description': '阶段2: 预测10步'},
                            {'epochs': 40, 'T_out': 15, 'description': '阶段3: 预测15步'},
                            {'epochs': 80, 'T_out': 20, 'description': '阶段4: 预测20步（最终目标）'}
                        ]
                    }
                },
                'loss': {
                    'reconstruction': {'name': 'MSELoss', 'weight': 1.0},
                    'spectral': {'name': 'SpectralLoss', 'weight': 0.3, 'freq_weight': 'low_freq', 'freq_ratio': 0.1},
                    'temporal_consistency': {'name': 'TemporalConsistencyLoss', 'weight': 0.2},
                    'gradient': {'name': 'GradientLoss', 'weight': 0.1}
                },
                'validation': {
                    'check_val_every_n_epoch': 5,
                    'val_check_interval': 0.5,
                    'metrics': ['mse', 'mae', 'rel_l2', 'psnr', 'ssim', 'temporal_mse', 'long_term_stability']
                },
                'hardware': {
                    'num_workers': 0,
                    'pin_memory': False,
                    'persistent_workers': False
                }
            })
        
        # 设置随机种子
        torch.manual_seed(self.config.experiment.seed)
        np.random.seed(self.config.experiment.seed)
        
    def setup_logging(self):
        """设置日志"""
        self.output_dir = Path(self.config.experiment.output_dir) / self.config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name="RealDataARTrainer",
            log_file=self.output_dir / "training.log",
            level=logging.INFO
        )
        
        self.logger.info(f"输出目录: {self.output_dir}")
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
    def setup_device(self):
        """设置设备"""
        # 检查配置中是否有device设置，如果没有则使用默认值
        if hasattr(self.config, 'device') and hasattr(self.config.device, 'accelerator'):
            device_name = self.config.device.accelerator
            if device_name == 'gpu':
                device_name = 'cuda'
        else:
            device_name = 'cuda'
            
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def setup_data(self):
        """设置数据"""
        self.logger.info("设置数据模块...")
        
        try:
            # 使用真实扩散反应数据模块
            self.data_module = RealDiffusionReactionDataModule(
                data_path=self.config.data.data_path,
                T_in=self.config.data.T_in,
                T_out=self.config.data.T_out,
                batch_size=self.config.data.dataloader.batch_size,
                num_workers=self.config.data.dataloader.num_workers,
                pin_memory=self.config.data.dataloader.pin_memory,
                persistent_workers=self.config.data.dataloader.persistent_workers,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                normalize=self.config.data.preprocessing.normalize,
                augmentation=self.config.data.preprocessing.augmentation,
                time_step_start=0,
                time_step_end=100
            )
            self.data_module.setup()
            
            # 获取数据加载器
            self.train_loader = self.data_module.train_dataloader()
            self.val_loader = self.data_module.val_dataloader()
            self.test_loader = self.data_module.test_dataloader()
            
            self.logger.info(f"训练集批次数: {len(self.train_loader)}")
            self.logger.info(f"验证集批次数: {len(self.val_loader)}")
            self.logger.info(f"测试集批次数: {len(self.test_loader)}")
            
            # 测试数据加载
            sample_batch = next(iter(self.train_loader))
            self.logger.info(f"✅ 输入序列形状: {sample_batch['input_sequence'].shape}")
            self.logger.info(f"✅ 目标序列形状: {sample_batch['target_sequence'].shape}")
            
        except Exception as e:
            self.logger.error(f"❌ 数据设置失败: {e}")
            raise
    
    def setup_model(self):
        """设置模型"""
        self.logger.info("🏗️ 设置模型...")
        
        try:
            # 创建基础模型
            base_model = SwinUNet(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                img_size=self.config.model.img_size,
                patch_size=self.config.model.patch_size,
                window_size=self.config.model.window_size,
                depths=self.config.model.depths,
                num_heads=self.config.model.num_heads,
                embed_dim=self.config.model.embed_dim,
                mlp_ratio=self.config.model.mlp_ratio,
                drop_rate=self.config.model.drop_rate,
                attn_drop_rate=self.config.model.attn_drop_rate,
                drop_path_rate=self.config.model.drop_path_rate
            )
            
            # 包装为AR模型
            self.model = ARWrapper(
                single_frame_model=base_model,
                detach_rollout=True,
                scheduled_sampling=False
            )
            
            self.model = self.model.to(self.device)
            
            # 计算参数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"✅ 模型参数量: {total_params:,} (可训练: {trainable_params:,})")
            
        except Exception as e:
            self.logger.error(f"❌ 模型设置失败: {e}")
            raise
    
    def setup_optimizer(self):
        """设置优化器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.optimizer.lr,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=self.config.training.optimizer.betas
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.scheduler.T_max,
            eta_min=self.config.training.scheduler.eta_min
        )
        
        # 梯度缩放器（混合精度）
        self.scaler = GradScaler()
        
        self.logger.info(f"✅ 优化器: {self.config.training.optimizer.name}")
        self.logger.info(f"✅ 学习率: {self.config.training.optimizer.lr}")
        
    def setup_monitoring(self):
        """设置监控"""
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': [],
            'curriculum_stages': []
        }
        
        # 课程学习状态
        self.current_stage = 0
        self.stage_epoch = 0
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            self.logger.info(f"✅ 成功加载检查点: {checkpoint_path}")
            self.logger.info(f"恢复到 epoch {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载检查点失败: {str(e)}")
            return False
        
    def get_current_T_out(self, epoch: int) -> int:
        """获取当前阶段的T_out"""
        if not self.config.training.curriculum.enabled:
            return self.config.data.T_out
        
        stages = self.config.training.curriculum.stages
        cumulative_epochs = 0
        
        for i, stage in enumerate(stages):
            cumulative_epochs += stage['epochs']
            if epoch < cumulative_epochs:
                if i != self.current_stage:
                    self.current_stage = i
                    self.stage_epoch = 0
                    self.logger.info(f"🎯 进入{stage['description']}")
                return stage['T_out']
        
        # 如果超出所有阶段，使用最后一个阶段的T_out
        return stages[-1]['T_out']
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 获取当前T_out
        current_T_out = self.get_current_T_out(epoch)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_seq = batch['input_sequence'].to(self.device)  # [B, T_in, C, H, W]
            target_seq = batch['target_sequence'].to(self.device)  # [B, T_out, C, H, W]
            
            # 根据课程学习调整目标序列长度
            if target_seq.shape[1] > current_T_out:
                target_seq = target_seq[:, :current_T_out]
            
            # 前向传播
            with autocast():
                pred_seq = self.model(input_seq, T_out=current_T_out, teacher=target_seq)
                
                # 计算损失
                loss = F.mse_loss(pred_seq, target_seq)
                
                # 添加其他损失项
                if self.config.loss.get('spectral', {}).get('weight', 0) > 0:
                    # 简化的频域损失
                    pred_fft = torch.fft.fft2(pred_seq.reshape(-1, *pred_seq.shape[-2:]))
                    target_fft = torch.fft.fft2(target_seq.reshape(-1, *target_seq.shape[-2:]))
                    spectral_loss = F.mse_loss(pred_fft.real, target_fft.real) + F.mse_loss(pred_fft.imag, target_fft.imag)
                    loss += self.config.loss.spectral.weight * spectral_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'T_out': current_T_out,
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录到TensorBoard
            if batch_idx % self.config.experiment.log_every_n_steps == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
                self.writer.add_scalar('Train/T_out', current_T_out, global_step)
        
        avg_loss = total_loss / num_batches
        self.stage_epoch += 1
        
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        current_T_out = self.get_current_T_out(epoch)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                with autocast():
                    pred_seq = self.model(input_seq, T_out=current_T_out)
                    loss = F.mse_loss(pred_seq, target_seq)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def test_epoch(self) -> Dict[str, float]:
        """测试集评估"""
        self.logger.info("🧪 开始测试集评估...")
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 模型预测（测试时不使用teacher forcing）
                pred_seq = self.model(input_seq, target_seq=None)
                
                # 计算损失
                loss = F.mse_loss(pred_seq, target_seq)
                total_loss += loss.item()
                
                # 计算详细指标
                pred_np = pred_seq.cpu().numpy()
                target_np = target_seq.cpu().numpy()
                
                batch_metrics = compute_metrics(pred_np, target_np)
                all_metrics.append(batch_metrics)
        
        # 聚合指标
        avg_loss = total_loss / num_batches
        
        # 计算平均指标
        final_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                final_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        final_metrics['test_loss'] = avg_loss
        
        self.logger.info(f"✅ 测试完成 - 损失: {avg_loss:.6f}")
        for key, value in final_metrics.items():
            if key != 'test_loss':
                self.logger.info(f"  {key}: {value:.6f}")
        
        return final_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_yaml(self.config),
            'training_history': self.training_history
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'last.ckpt')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.ckpt')
            self.logger.info(f"💾 保存最佳模型 (验证损失: {self.best_val_loss:.6f})")
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始训练...")
        
        start_time = time.time()
        start_epoch = self.current_epoch
        
        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                epoch_start_time = time.time()
                
                # 训练
                train_loss = self.train_epoch(epoch)
                
                # 验证
                if (epoch + 1) % self.config.validation.check_val_every_n_epoch == 0:
                    val_loss = self.validate_epoch(epoch)
                    
                    # 记录历史
                    self.training_history['train_losses'].append(train_loss)
                    self.training_history['val_losses'].append(val_loss)
                    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    self.training_history['epochs'].append(epoch)
                    
                    # 检查是否为最佳模型
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # 保存检查点
                    self.save_checkpoint(epoch, is_best)
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar('Val/Loss', val_loss, epoch)
                    
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Best: {self.best_val_loss:.6f} | "
                        f"Time: {epoch_time:.1f}s"
                    )
                else:
                    # 只记录训练损失
                    self.training_history['train_losses'].append(train_loss)
                    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    self.training_history['epochs'].append(epoch)
                    
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Time: {epoch_time:.1f}s"
                    )
                
                # 更新学习率
                self.scheduler.step()
                
                # 保存训练历史
                with open(self.output_dir / 'training_history.json', 'w') as f:
                    json.dump(self.training_history, f, indent=2)
        
        except KeyboardInterrupt:
            self.logger.info("⚠️ 训练被用户中断")
        except Exception as e:
            self.logger.error(f"❌ 训练过程中出现错误: {e}")
            traceback.print_exc()
        finally:
            total_time = time.time() - start_time
            self.logger.info(f"🏁 训练完成，总用时: {total_time/3600:.2f} 小时")
            
            # 加载最佳模型进行最终测试
            best_ckpt_path = self.output_dir / 'best.ckpt'
            if best_ckpt_path.exists():
                self.logger.info("📊 使用最佳模型进行最终测试评估...")
                self.load_checkpoint(str(best_ckpt_path))
                final_test_metrics = self.test_epoch()
                
                # 保存测试结果
                test_results = {
                    'final_test_metrics': final_test_metrics,
                    'test_time': time.time(),
                    'model_path': str(best_ckpt_path)
                }
                
                with open(self.output_dir / 'test_results.json', 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                self.logger.info("✅ 最终测试结果已保存到 test_results.json")
            
            self.writer.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="真实扩散-反应数据AR训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = RealDataARTrainer(args.config)
    
    # 如果指定了恢复检查点，加载它
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()