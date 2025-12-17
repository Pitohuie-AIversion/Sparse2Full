#!/usr/bin/env python3
"""
基于配置文件的真实数据AR训练脚本

使用YAML配置文件来训练AR模型，支持RealDiffusionReaction数据集
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import torch.cuda.amp as amp
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

# 导入本地模块
from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
from losses.ar_losses import compute_ar_total_loss
from models.swin_unet import SwinUNet
from utils.ar_metrics import ARMetrics
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.logging_utils import setup_logger
from utils.visualization import ARVisualizer
from utils.resource_monitor import ResourceMonitor


class ConfigBasedARTrainer:
    """基于配置文件的AR训练器"""
    
    def __init__(self, config: Dict[str, Any], config_path: Path, test_mode: bool = False):
        self.config = config
        self.config_path = config_path
        self.test_mode = test_mode
        self.logger = setup_logger(name="ConfigBasedARTrainer")
        
        # 实验设置
        self.experiment_name = config["experiment"]["name"]
        self.seed = config["experiment"]["seed"]
        self.device = self._setup_device()
        self._set_seed(self.seed)
        
        # 训练参数
        self.max_epochs = config["training"]["max_epochs"]
        
        # 修复学习率配置问题 - 从optimizer配置获取
        if "optimizer" in config["training"] and "lr" in config["training"]["optimizer"]:
            self.learning_rate = config["training"]["optimizer"]["lr"]
        else:
            self.learning_rate = 1e-3  # 默认值
            self.logger.warning("No learning_rate found in config, using default value: 1e-3")
        
        # 修复权重衰减配置问题 - 从optimizer配置获取
        if "optimizer" in config["training"] and "weight_decay" in config["training"]["optimizer"]:
            self.weight_decay = config["training"]["optimizer"]["weight_decay"]
        else:
            self.weight_decay = 1e-4  # 默认值
            self.logger.warning("No weight_decay found in config, using default value: 1e-4")
        
        # 修复其他训练参数配置问题
        if "early_stopping" in config["training"] and "patience" in config["training"]["early_stopping"]:
            self.patience = config["training"]["early_stopping"]["patience"]
        else:
            self.patience = 10  # 默认值
            self.logger.warning("No patience found in config, using default value: 10")
        
        # 梯度裁剪
        if "gradient_clip_val" in config["training"]:
            self.grad_clip_norm = config["training"]["gradient_clip_val"]
        else:
            self.grad_clip_norm = 1.0  # 默认值
            self.logger.warning("No gradient_clip_val found in config, using default value: 1.0")
        
        # 梯度累积
        if "gradient_accumulation_steps" in config["training"]:
            self.accumulate_grad_batches = config["training"]["gradient_accumulation_steps"]
        else:
            self.accumulate_grad_batches = 1  # 默认值
            self.logger.warning("No gradient_accumulation_steps found in config, using default value: 1")
        
        # 数据设置 - 修复batch_size配置问题
        # 优先从data.dataloader获取batch_size，如果没有则从training获取
        if "dataloader" in config["data"] and "batch_size" in config["data"]["dataloader"]:
            self.batch_size = config["data"]["dataloader"]["batch_size"]
        elif "batch_size" in config["training"]:
            self.batch_size = config["training"]["batch_size"]
        else:
            # 默认值
            self.batch_size = 32
            self.logger.warning("No batch_size found in config, using default value: 32")
        
        self.num_workers = config["data"].get("num_workers", 4)
        self.dataset_name = config["data"]["dataset_name"]
        self.data_path = config["data"]["data_path"]
        
        # 模型设置
        self.model_config = config["model"]
        
        # 修复time_steps和pred_steps配置问题
        if "ar_config" in config["model"]:
            ar_config = config["model"]["ar_config"]
            if "T_in" in config["data"]:
                self.time_steps = config["data"]["T_in"]
            else:
                self.time_steps = 1  # 默认值
                self.logger.warning("No T_in found in data config, using default time_steps: 1")
            
            if "T_out" in config["data"]:
                self.pred_steps = config["data"]["T_out"]
            else:
                self.pred_steps = 5  # 默认值
                self.logger.warning("No T_out found in data config, using default pred_steps: 5")
        else:
            # 兼容旧配置
            self.time_steps = config["model"].get("time_steps", 1)
            self.pred_steps = config["model"].get("pred_steps", 5)
        
        # 损失函数设置
        self.loss_config = config["loss"]
        
        # 初始化组件
        self.data_module = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = ARMetrics()
        self.visualizer = ARVisualizer()
        self.resource_monitor = ResourceMonitor()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 输出目录
        self.output_dir = Path(config["experiment"]["output_dir"]) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        
        # 混合精度训练
        if "amp" in config["training"] and "enabled" in config["training"]["amp"]:
            self.use_amp = HAS_AMP and config["training"]["amp"]["enabled"]
        else:
            self.use_amp = HAS_AMP  # 默认启用AMP如果可用
        if self.use_amp:
            self.scaler = amp.GradScaler()
        
        self.logger.info(f"ConfigBasedARTrainer initialized with experiment: {self.experiment_name}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Using AMP: {self.use_amp}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        return device
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"Set random seed: {seed}")
    
    def setup_data_module(self) -> None:
        """设置数据模块"""
        self.logger.info("Setting up data module...")
        
        try:
            # 创建数据模块 - 修复参数传递问题
            # RealDiffusionReactionDataModule需要DictConfig对象
            from omegaconf import DictConfig, OmegaConf
            
            # 在测试模式下使用单进程以避免DataLoader问题
            num_workers = 0 if self.test_mode else self.num_workers
            
            # 创建完整的配置结构
            config_dict = {
                'data': self.config['data'].copy(),
                'training': {'batch_size': self.batch_size},
                'hardware': {
                    'num_workers': num_workers,
                    'pin_memory': torch.cuda.is_available() and not self.test_mode
                },
                'testing': {'batch_size': 1},
                'seed': self.seed
            }
            
            # 确保必要的参数存在
            if 'dataloader' not in config_dict['data']:
                config_dict['data']['dataloader'] = {
                    'batch_size': self.batch_size,
                    'num_workers': num_workers,
                    'pin_memory': torch.cuda.is_available() and not self.test_mode,
                    'persistent_workers': False,  # 测试模式下禁用
                    'prefetch_factor': None,    # 测试模式下禁用
                    'drop_last': True,
                    'shuffle': True
                    # 注意：不设置pin_memory_device，让DataLoader使用默认值
                }
            
            config_obj = OmegaConf.create(config_dict)
            self.data_module = RealDiffusionReactionDataModule(config_obj)
            
            # 准备数据
            self.data_module.prepare_data()
            self.data_module.setup()
            
            self.logger.info("Data module setup completed")
            self.logger.info(f"Train samples: {len(self.data_module.train_dataset)}")
            self.logger.info(f"Val samples: {len(self.data_module.val_dataset)}")
            self.logger.info(f"Test samples: {len(self.data_module.test_dataset)}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup data module: {e}")
            raise
    
    def create_model(self) -> nn.Module:
        """创建模型"""
        self.logger.info("Creating model...")
        
        # 获取数据维度信息 - 使用更安全的方式
        try:
            # 尝试从数据集直接获取样本，避免DataLoader迭代问题
            if hasattr(self.data_module, 'train_dataset') and len(self.data_module.train_dataset) > 0:
                sample_data = self.data_module.train_dataset[0]
                if isinstance(sample_data, dict) and 'input_sequence' in sample_data:
                    input_data = sample_data['input_sequence']
                elif isinstance(sample_data, (list, tuple)):
                    input_data = sample_data[0]
                else:
                    input_data = sample_data
                
                if hasattr(input_data, 'shape'):
                    in_channels = input_data.shape[1]  # 假设shape是 [C, H, W] 或 [T, C, H, W]
                else:
                    in_channels = 2  # 根据数据集特点，默认2个通道
            else:
                in_channels = 2  # 默认通道数
        except Exception as e:
            self.logger.warning(f"Failed to get data shape from dataset: {e}, using default channels: 2")
            in_channels = 2  # 扩散-反应数据集通常有2个通道
        
        out_channels = self.model_config.get("out_channels", in_channels)
        img_size = self.model_config.get("img_size", 128)  # 根据数据集调整默认大小
        
        # 创建SwinUNet模型
        model = SwinUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            **self.model_config
        )
        
        self.logger.info(f"Model created: {model.__class__.__name__}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def setup_training(self) -> None:
        """设置训练组件"""
        self.logger.info("Setting up training components...")
        
        # 创建模型
        self.model = self.create_model()
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        self.logger.info("Training components setup completed")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取输入和目标
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_data, target_data = batch[0], batch[1]
            else:
                input_data = batch
                target_data = batch
            
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # 前向传播和损失计算
            if self.use_amp:
                with amp.autocast():
                    output = self.model(input_data)
                    loss = compute_ar_total_loss(
                        output, target_data, 
                        self.model, self.loss_config
                    )
            else:
                output = self.model(input_data)
                loss = compute_ar_total_loss(
                    output, target_data,
                    self.model, self.loss_config
                )
            
            # 梯度累积
            loss = loss / self.accumulate_grad_batches
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度更新
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.use_amp:
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.accumulate_grad_batches
            num_batches += 1
            self.global_step += 1
            
            # 记录到TensorBoard
            if self.global_step % 100 == 0:
                self.tb_writer.add_scalar('train/loss_step', loss.item(), self.global_step)
        
        avg_loss = epoch_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取输入和目标
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_data, target_data = batch[0], batch[1]
                else:
                    input_data = batch
                    target_data = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                # 前向传播和损失计算
                if self.use_amp:
                    with amp.autocast():
                        output = self.model(input_data)
                        loss = compute_ar_total_loss(
                            output, target_data,
                            self.model, self.loss_config
                        )
                else:
                    output = self.model(input_data)
                    loss = compute_ar_total_loss(
                        output, target_data,
                        self.model, self.loss_config
                    )
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / max(num_batches, 1)
        return {"val_loss": avg_val_loss}
    
    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        self.logger.info("Starting training...")
        
        # 设置数据和训练组件
        self.setup_data_module()
        self.setup_training()
        
        training_history = []
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证一个epoch
            val_metrics = self.validate_epoch(epoch)
            
            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            training_history.append(epoch_metrics)
            
            # 记录到TensorBoard
            for key, value in epoch_metrics.items():
                self.tb_writer.add_scalar(f'epoch/{key}', value, epoch)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.tb_writer.add_scalar('train/lr', current_lr, epoch)
            
            # 打印进度
            self.logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f} - "
                f"LR: {current_lr:.2e}"
            )
            
            # 早停检查
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                # 保存最佳模型
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, self.best_val_loss,
                    str(self.output_dir / "best_model.pth")
                )
                self.logger.info("Saved best model checkpoint")
            else:
                self.patience_counter += 1
                
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, val_metrics['val_loss'],
                    str(self.output_dir / f"checkpoint_epoch_{epoch+1}.pth")
                )
            
            # 更新学习率
            self.scheduler.step()
            
            # 早停
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.current_epoch, self.global_step, self.best_val_loss,
            str(self.output_dir / "final_model.pth")
        )
        
        self.logger.info("Training completed!")
        self.tb_writer.close()
        
        return {
            "experiment_name": self.experiment_name,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
            "training_history": training_history
        }
    
    def test(self) -> Dict[str, float]:
        """测试模型"""
        self.logger.info("Starting testing...")
        
        self.model.eval()
        test_metrics = {}
        
        test_loader = self.data_module.test_dataloader()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 获取输入和目标
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_data, target_data = batch[0], batch[1]
                else:
                    input_data = batch
                    target_data = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                # 前向传播
                output = self.model(input_data)
                
                # 计算指标
                batch_metrics = self.metrics.compute_metrics(output, target_data)
                
                # 累积指标
                for key, value in batch_metrics.items():
                    if key not in test_metrics:
                        test_metrics[key] = []
                    test_metrics[key].append(value)
        
        # 计算平均指标
        avg_test_metrics = {
            key: np.mean(values) for key, values in test_metrics.items()
        }
        
        self.logger.info("Test results:")
        for key, value in avg_test_metrics.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        return avg_test_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于配置文件的真实数据AR训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--test", action="store_true", help="仅运行测试")
    parser.add_argument("--max_epochs", type=int, help="覆盖配置文件中的max_epochs")
    parser.add_argument("--test_mode", action="store_true", help="测试模式（仅运行少量epoch）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs
    
    if args.test_mode:
        config["training"]["max_epochs"] = 2
        # 修复test_mode下的batch_size设置问题
        if "dataloader" in config["data"] and "batch_size" in config["data"]["dataloader"]:
            config["data"]["dataloader"]["batch_size"] = min(2, config["data"]["dataloader"]["batch_size"])
        elif "batch_size" in config["training"]:
            config["training"]["batch_size"] = min(2, config["training"]["batch_size"])
        else:
            # 如果都没有，设置默认值
            config["training"]["batch_size"] = 2
    
    # 创建训练器
    trainer = ConfigBasedARTrainer(config, Path(args.config))
    
    try:
        if args.test:
            # 仅运行测试
            if args.resume:
                # 加载检查点
                checkpoint = torch.load(args.resume, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.logger.info(f"Loaded checkpoint from {args.resume}")
            
            # 运行测试
            test_results = trainer.test()
            print("Test Results:")
            for key, value in test_results.items():
                print(f"  {key}: {value:.6f}")
        
        else:
            # 运行训练
            if args.resume:
                # 恢复训练
                checkpoint = torch.load(args.resume, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                trainer.current_epoch = checkpoint['epoch']
                trainer.global_step = checkpoint['global_step']
                trainer.best_val_loss = checkpoint['best_val_loss']
                trainer.logger.info(f"Resumed training from epoch {trainer.current_epoch + 1}")
            
            # 开始训练
            training_results = trainer.train()
            
            # 运行测试
            test_results = trainer.test()
            
            # 打印最终结果
            print("\n" + "="*50)
            print("TRAINING COMPLETED")
            print("="*50)
            print(f"Experiment: {training_results['experiment_name']}")
            print(f"Best Validation Loss: {training_results['best_val_loss']:.6f}")
            print(f"Total Epochs: {training_results['total_epochs']}")
            print("\nTest Results:")
            for key, value in test_results.items():
                print(f"  {key}: {value:.6f}")
            print("="*50)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()