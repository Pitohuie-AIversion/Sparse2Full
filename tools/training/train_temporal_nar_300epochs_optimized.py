#!/usr/bin/env python3
"""
优化版时序NAR模型300轮训练脚本
针对GPU利用率和训练效率进行全面优化
"""

import os
import sys
import time
import json
import logging
import warnings
import psutil
import threading
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


class GPUMonitor:
    """GPU性能监控器"""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.gpu_stats = []
        self.running = False
        self.thread = None
        
    def start_monitoring(self):
        """开始监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                gpu_utilization = self._get_gpu_utilization()
                
                self.gpu_stats.append({
                    'timestamp': time.time(),
                    'memory_allocated': gpu_memory,
                    'memory_cached': gpu_memory_cached,
                    'utilization': gpu_utilization
                })
            
            time.sleep(1)  # 每秒监控一次
    
    def _get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0
    
    def log_stats(self, writer, step):
        """记录统计信息到TensorBoard"""
        if self.gpu_stats:
            recent_stats = self.gpu_stats[-10:]  # 最近10秒的数据
            avg_memory = np.mean([s['memory_allocated'] for s in recent_stats])
            avg_utilization = np.mean([s['utilization'] for s in recent_stats])
            
            writer.add_scalar('System/GPU_Memory_GB', avg_memory, step)
            writer.add_scalar('System/GPU_Utilization_%', avg_utilization, step)


class OptimizedTemporalNARTrainer:
    """优化版时序NAR模型训练器 - 300轮高效训练版本"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.experiment.device)
        self.use_amp = config.experiment.use_amp
        
        # 设置随机种子
        self._set_seed(config.experiment.seed)
        
        # 性能优化设置
        self._setup_performance_optimizations()
        
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
        
        # GPU监控器
        self.gpu_monitor = GPUMonitor(log_interval=config.monitoring.get('gpu_monitoring', {}).get('log_interval', 100))
        
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
            'metrics': [],
            'gpu_stats': [],
            'performance_stats': []
        }
        
        # AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # 可视化器
        # 配置可视化器
        vis_config = {
            'enabled': True,
            'save_dir': 'visualizations',
            'training': {
                'plot_curves': True,
                'save_predictions': True,
                'plot_interval': 100
            }
        }
        self.visualizer = TemporalVisualizer(save_dir=self.visualization_dir, config=vis_config)
        
        # 性能计时器
        self.batch_timer = 0
        self.data_timer = 0
        
        # 保存配置快照
        self._save_config_snapshot()
        
        self.logger.info(f"OptimizedTemporalNARTrainer initialized. Output dir: {self.output_dir}")
        self.logger.info(f"Training for {config.train.max_epochs} epochs with enhanced GPU optimization")
        
        # 启动GPU监控
        if config.monitoring.get('gpu_monitoring', {}).get('enabled', True):
            self.gpu_monitor.start_monitoring()
    
    def _setup_performance_optimizations(self):
        """设置性能优化"""
        # CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = self.config.train.get('benchmark', True)
            torch.backends.cudnn.deterministic = self.config.train.get('deterministic', False)
            
            # 设置CUDA内存管理
            if hasattr(self.config, 'system') and 'cuda' in self.config.system:
                cuda_config = self.config.system.cuda
                if 'max_split_size_mb' in cuda_config:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{cuda_config.max_split_size_mb}'
        
        # 设置多进程策略
        if hasattr(self.config, 'system') and 'multiprocessing' in self.config.system:
            mp_config = self.config.system.multiprocessing
            if 'sharing_strategy' in mp_config:
                torch.multiprocessing.set_sharing_strategy(mp_config.sharing_strategy)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # 注意：为了性能，可能需要关闭确定性
        if not self.config.train.get('deterministic', False):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
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
            'gpu_count': torch.cuda.device_count(),
            'timestamp': datetime.now().isoformat()
        }
        
        env_path = self.output_dir / "environment.json"
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
    
    def _init_data(self):
        """初始化数据模块"""
        self.logger.info("初始化优化数据模块...")
        
        # 使用时序数据模块
        self.data_module = TemporalPDEBenchDataModule(self.config.data)
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # 记录数据加载器配置
        batch_size = self.config.data.batch_size
        num_workers = self.config.data.num_workers
        
        self.logger.info(f"数据加载完成. Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
        self.logger.info(f"优化配置: batch_size={batch_size}, num_workers={num_workers}")
        
        # 记录到TensorBoard
        self.writer.add_text("Data/Config", f"batch_size={batch_size}, num_workers={num_workers}")
    
    def _init_model(self):
        """初始化模型"""
        self.logger.info("初始化优化AR+NAR双头模型...")
        
        # 包装为AR+NAR模型
        self.model = ARNARWrapper(
            model_config=self.config.model,
            loss_config=self.config.loss,
            training_config=self.config.train
        )
        
        self.model = self.model.to(self.device)
        
        # 模型编译优化（PyTorch 2.0+）
        if hasattr(torch, 'compile') and self.config.train.get('compile_model', False):
            self.model = torch.compile(self.model)
            self.logger.info("模型已编译优化")
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成: 优化AR+NAR双头时序模型")
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
        # 记录到TensorBoard
        self.writer.add_text("Model/Architecture", "优化AR+NAR双头时序模型")
        self.writer.add_scalar("Model/TotalParams", total_params)
        self.writer.add_scalar("Model/TrainableParams", trainable_params)
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        # 优化器配置
        optimizer_config = self.config.train.optimizer
        
        # 检查是否支持融合优化器
        fused = optimizer_config.get('fused', False) and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        
        optimizer_kwargs = {
            'lr': optimizer_config.lr,
            'weight_decay': optimizer_config.weight_decay,
            'betas': optimizer_config.betas,
            'eps': optimizer_config.eps
        }
        
        if fused:
            optimizer_kwargs['fused'] = True
            self.logger.info("使用融合AdamW优化器")
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        
        # 学习率调度器
        scheduler_config = self.config.train.scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config.T_0,
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.eta_min
        )
        
        self.logger.info(f"优化器初始化完成: lr={optimizer_config.lr}, fused={fused}")
    
    def _init_losses(self):
        """初始化损失函数"""
        # 配置AR损失函数
        ar_loss_config = OmegaConf.create({
            'loss_type': 'mse',
            'step_weights': None,
            'accumulate_loss': True
        })
        
        self.ar_loss = ARLoss(ar_loss_config)
        
        # 配置频谱损失函数
        spectral_loss_config = OmegaConf.create({
            'k_max': 16
        })
        self.spectral_loss = SpectralLoss(spectral_loss_config)
        
        # 配置DC损失函数
        dc_loss_config = OmegaConf.create({})
        self.dc_loss = DCLoss(dc_loss_config)
        
        # 损失权重
        self.loss_weights = {
            'ar': self.config.loss.ar_weight,
            'nar': self.config.loss.nar_weight,
            'spectral': self.config.loss.spectral_loss.weight,
            'dc': self.config.loss.dc_loss.weight
        }
    
    def _init_metrics(self):
        """初始化评估指标"""
        self.metrics = {}
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_ar_losses = []
        epoch_nar_losses = []
        epoch_spectral_losses = []
        epoch_dc_losses = []
        
        # 性能统计
        batch_times = []
        data_times = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.train.max_epochs}")
        
        data_start_time = time.time()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 检查batch是否为None（由safe_collate_fn返回）
            if batch is None:
                self.logger.warning(f"Skipping None batch at index {batch_idx}")
                continue
            
            # 数据加载时间
            data_time = time.time() - data_start_time
            data_times.append(data_time)
            
            batch_start_time = time.time()
            
            # 数据移动到GPU
            x = batch['input'].to(self.device, non_blocking=True)
            y = batch['target'].to(self.device, non_blocking=True)
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                # AR+NAR预测
                ar_pred, nar_pred = self.model(x)
                
                # 计算损失
                ar_loss = self.ar_loss(ar_pred, y) * self.loss_weights['ar']
                nar_loss = F.mse_loss(nar_pred, y) * self.loss_weights['nar']
                spectral_loss = self.spectral_loss(nar_pred, y) * self.loss_weights['spectral']
                dc_loss = self.dc_loss(nar_pred, y) * self.loss_weights['dc']
                
                total_loss = ar_loss + nar_loss + spectral_loss + dc_loss
            
            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_val)
                self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录损失
            epoch_losses.append(total_loss.item())
            epoch_ar_losses.append(ar_loss.item())
            epoch_nar_losses.append(nar_loss.item())
            epoch_spectral_losses.append(spectral_loss.item())
            epoch_dc_losses.append(dc_loss.item())
            
            # 批处理时间
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'AR': f'{ar_loss.item():.4f}',
                'NAR': f'{nar_loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'BatchTime': f'{batch_time:.3f}s'
            })
            
            # 记录到TensorBoard
            if self.global_step % self.config.experiment.log_every_n_steps == 0:
                self.writer.add_scalar('Train/TotalLoss', total_loss.item(), self.global_step)
                self.writer.add_scalar('Train/ARLoss', ar_loss.item(), self.global_step)
                self.writer.add_scalar('Train/NARLoss', nar_loss.item(), self.global_step)
                self.writer.add_scalar('Train/SpectralLoss', spectral_loss.item(), self.global_step)
                self.writer.add_scalar('Train/DCLoss', dc_loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.global_step)
                
                # 性能指标
                if batch_times:
                    self.writer.add_scalar('Performance/BatchTime', np.mean(batch_times[-10:]), self.global_step)
                    self.writer.add_scalar('Performance/DataTime', np.mean(data_times[-10:]), self.global_step)
                
                # GPU监控
                self.gpu_monitor.log_stats(self.writer, self.global_step)
            
            self.global_step += 1
            
            # CUDA缓存清理
            if hasattr(self.config, 'system') and 'cuda' in self.config.system:
                empty_cache_steps = self.config.system.cuda.get('empty_cache_steps', 100)
                if self.global_step % empty_cache_steps == 0:
                    torch.cuda.empty_cache()
            
            data_start_time = time.time()
        
        # 记录epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_ar_loss = np.mean(epoch_ar_losses)
        avg_nar_loss = np.mean(epoch_nar_losses)
        avg_spectral_loss = np.mean(epoch_spectral_losses)
        avg_dc_loss = np.mean(epoch_dc_losses)
        
        # 性能统计
        avg_batch_time = np.mean(batch_times) if batch_times else 0.0
        avg_data_time = np.mean(data_times) if data_times else 0.0
        total_batch_time = sum(batch_times) if batch_times else 1e-6  # 避免除零错误
        throughput = len(self.train_loader) * self.config.data.batch_size / total_batch_time
        
        self.logger.info(f"Epoch {self.current_epoch+1} - Train Loss: {avg_loss:.6f}, "
                        f"AR: {avg_ar_loss:.4f}, NAR: {avg_nar_loss:.4f}, "
                        f"Spectral: {avg_spectral_loss:.4f}, DC: {avg_dc_loss:.4f}")
        self.logger.info(f"Performance - BatchTime: {avg_batch_time:.3f}s, "
                        f"DataTime: {avg_data_time:.3f}s, Throughput: {throughput:.1f} samples/s")
        
        return {
            'total_loss': avg_loss,
            'ar_loss': avg_ar_loss,
            'nar_loss': avg_nar_loss,
            'spectral_loss': avg_spectral_loss,
            'dc_loss': avg_dc_loss,
            'performance': {
                'batch_time': avg_batch_time,
                'data_time': avg_data_time,
                'throughput': throughput
            }
        }
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 检查batch是否为None（由safe_collate_fn返回）
                if batch is None:
                    self.logger.warning("Skipping None batch in validation")
                    continue
                    
                x = batch['input'].to(self.device, non_blocking=True)
                y = batch['target'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    ar_pred, nar_pred = self.model(x)
                    
                    ar_loss = self.ar_loss(ar_pred, y) * self.loss_weights['ar']
                    nar_loss = F.mse_loss(nar_pred, y) * self.loss_weights['nar']
                    spectral_loss = self.spectral_loss(nar_pred, y) * self.loss_weights['spectral']
                    dc_loss = self.dc_loss(nar_pred, y) * self.loss_weights['dc']
                    
                    total_loss = ar_loss + nar_loss + spectral_loss + dc_loss
                
                val_losses.append(total_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        self.logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': OmegaConf.to_container(self.config)
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            self.logger.info(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
        
        # 定期保存
        if self.current_epoch % self.config.experiment.checkpoint.every_n_epochs == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始优化训练...")
        
        try:
            for epoch in range(self.config.train.max_epochs):
                self.current_epoch = epoch
                
                # 训练
                train_stats = self.train_epoch()
                
                # 验证
                if epoch % self.config.train.validation.check_val_every_n_epoch == 0:
                    val_loss = self.validate()
                    
                    # 记录历史
                    self.training_history['train_losses'].append(train_stats['total_loss'])
                    self.training_history['val_losses'].append(val_loss)
                    self.training_history['ar_losses'].append(train_stats['ar_loss'])
                    self.training_history['nar_losses'].append(train_stats['nar_loss'])
                    self.training_history['spectral_losses'].append(train_stats['spectral_loss'])
                    self.training_history['dc_losses'].append(train_stats['dc_loss'])
                    self.training_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
                    self.training_history['performance_stats'].append(train_stats['performance'])
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar('Val/Loss', val_loss, epoch)
                    self.writer.add_scalar('Performance/Throughput', train_stats['performance']['throughput'], epoch)
                    
                    # 检查是否为最佳模型
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    
                    # 保存检查点
                    self.save_checkpoint(is_best)
                    
                    # 早停检查
                    if self.early_stopping_counter >= self.config.train.early_stopping.patience:
                        self.logger.info(f"早停触发，在第{epoch+1}轮停止训练")
                        break
                
                # 保存训练历史
                history_path = self.output_dir / "training_history.json"
                with open(history_path, 'w') as f:
                    # 添加最佳验证损失
                    history_to_save = self.training_history.copy()
                    history_to_save['best_val_loss'] = self.best_val_loss
                    json.dump(history_to_save, f, indent=2)
                
                # 可视化
                if hasattr(self.config.monitoring, 'visualization') and \
                   self.config.monitoring.visualization.enabled and \
                   epoch % self.config.monitoring.visualization.save_every_n_epochs == 0:
                    self._create_visualizations(epoch)
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # 停止GPU监控
            self.gpu_monitor.stop_monitoring()
            
            # 关闭TensorBoard writer
            self.writer.close()
            
            self.logger.info("训练完成")
    
    def _create_visualizations(self, epoch):
        """创建可视化"""
        try:
            # 创建训练曲线
            self._plot_training_curves()
            
            # 创建性能图表
            self._plot_performance_stats()
            
        except Exception as e:
            self.logger.warning(f"可视化创建失败: {e}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not self.training_history['train_losses']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        
        axes[0, 0].plot(epochs, self.training_history['train_losses'], 'b-', label='Train Loss')
        if self.training_history['val_losses']:
            axes[0, 0].plot(epochs, self.training_history['val_losses'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 分项损失
        axes[0, 1].plot(epochs, self.training_history['ar_losses'], label='AR Loss')
        axes[0, 1].plot(epochs, self.training_history['nar_losses'], label='NAR Loss')
        axes[0, 1].plot(epochs, self.training_history['spectral_losses'], label='Spectral Loss')
        axes[0, 1].plot(epochs, self.training_history['dc_losses'], label='DC Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率
        axes[1, 0].plot(epochs, self.training_history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # 吞吐量
        if self.training_history['performance_stats']:
            throughputs = [stats['throughput'] for stats in self.training_history['performance_stats']]
            axes[1, 1].plot(epochs, throughputs)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Throughput (samples/s)')
            axes[1, 1].set_title('Training Throughput')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / f"training_curves_epoch_{self.current_epoch:03d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_stats(self):
        """绘制性能统计图表"""
        if not self.training_history['performance_stats']:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.training_history['performance_stats']) + 1)
        batch_times = [stats['batch_time'] for stats in self.training_history['performance_stats']]
        data_times = [stats['data_time'] for stats in self.training_history['performance_stats']]
        throughputs = [stats['throughput'] for stats in self.training_history['performance_stats']]
        
        # 批处理时间
        axes[0].plot(epochs, batch_times, 'b-', label='Batch Time')
        axes[0].plot(epochs, data_times, 'r-', label='Data Loading Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Processing Times')
        axes[0].legend()
        axes[0].grid(True)
        
        # 吞吐量
        axes[1].plot(epochs, throughputs, 'g-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Samples/s')
        axes[1].set_title('Training Throughput')
        axes[1].grid(True)
        
        # GPU统计（如果有的话）
        if self.gpu_monitor.gpu_stats:
            recent_stats = self.gpu_monitor.gpu_stats[-100:]  # 最近的统计
            gpu_memory = [s['memory_allocated'] for s in recent_stats]
            gpu_util = [s['utilization'] for s in recent_stats]
            
            ax2 = axes[2]
            ax3 = ax2.twinx()
            
            line1 = ax2.plot(gpu_memory, 'b-', label='GPU Memory (GB)')
            line2 = ax3.plot(gpu_util, 'r-', label='GPU Utilization (%)')
            
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Memory (GB)', color='b')
            ax3.set_ylabel('Utilization (%)', color='r')
            ax2.set_title('GPU Statistics')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
            
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / f"performance_stats_epoch_{self.current_epoch:03d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()


@hydra.main(version_base=None, config_path="configs/experiment", config_name="temporal_nar_300epochs_optimized")
def main(cfg: DictConfig) -> None:
    """主函数"""
    print("=" * 60)
    print("🚀 优化版时序NAR模型300轮训练")
    print("=" * 60)
    
    # 创建训练器
    trainer = OptimizedTemporalNARTrainer(cfg)
    
    # 开始训练
    trainer.train()
    
    print("✅ 训练完成!")


if __name__ == "__main__":
    main()