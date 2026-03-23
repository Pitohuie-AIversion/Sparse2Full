#!/usr/bin/env python3
"""
基于配置文件的真实扩散-反应数据AR训练脚本
使用ar_training_config debug_backup.yaml配置进行真实数据训练
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 确保项目根目录在sys.path中，避免模块导入问题
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""导入项目模块，支持备用路径"""
try:
    # 优先使用 training_system 路径（本仓库实际位置）
    from training_system.utils.real_dr_dataset import RealDiffusionReactionDataModule
except ImportError:
    try:
        # 兼容旧路径
        from datasets.real_diffusion_reaction_dataset import RealDiffusionReactionDataModule
    except ImportError as e:
        print(f"导入数据模块失败: {e}")
        print("请确认存在 training_system/utils/real_dr_dataset.py 或 datasets/real_diffusion_reaction_dataset.py")
        exit(1)

try:
    from models.swin_unet import SwinUnet
    from ops.losses import ARLoss
    from utils.logger import setup_logger
    from utils.metrics import compute_ar_metrics  # 若缺失，训练仍可继续
    from utils.ar_visualizer import ARTrainingVisualizer
except ImportError as e:
    print(f"导入模型或工具模块失败: {e}")
    print("确保在项目根目录运行并且 models/、ops/、utils/ 路径可用")
    exit(1)


class ConfigBasedARTrainer:
    """基于配置文件的AR训练器"""
    
    def __init__(self, config_path: str, resume_path: Optional[str] = None):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
            resume_path: 恢复训练的检查点路径
        """
        self.config_path = config_path
        self.resume_path = resume_path
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置实验目录
        self.setup_experiment()
        
        # 设置日志
        self.logger = setup_logger(
            name=f"ar_training_{self.experiment_name}",
            log_file=self.log_dir / "training.log"
        )
        
        # 设置设备
        self.device = self._setup_device()
        
        # 初始化训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': [],
            'gpu_memory': []
        }
        
        # 设置随机种子
        self._set_seed()
        
        # 初始化组件
        self._setup_components()
        
        # 恢复训练状态
        if resume_path:
            self.load_checkpoint(resume_path)
    
    def _load_config(self, config_path: str) -> DictConfig:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        config = OmegaConf.load(config_path)
        
        # 验证必要配置项
        required_sections = ['experiment', 'data', 'model', 'training']
        for section in required_sections:
            if not hasattr(config, section):
                raise ValueError(f"配置缺少必要部分: {section}")
        
        return config
    
    def setup_experiment(self):
        """设置实验目录"""
        # 获取实验名称
        self.experiment_name = self.config.experiment.get('name', 'real_diffusion_reaction_ar')
        
        # 添加时间戳和种子信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.config.experiment.get('seed', 42)
        self.experiment_name = f"{self.experiment_name}-s{seed}-{timestamp}"
        
        # 设置输出目录
        output_root = Path(self.config.experiment.get('output_dir', 'runs'))
        self.output_dir = output_dir = output_root / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.log_dir = output_dir / 'logs'
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.visualization_dir = output_dir / 'visualizations'
        
        for dir_path in [self.log_dir, self.checkpoint_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置副本
        config_copy_path = self.output_dir / 'config.yaml'
        OmegaConf.save(self.config, config_copy_path)
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config.training.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        self.logger.info(f"使用设备: {device}")
        return device
    
    def _set_seed(self):
        """设置随机种子"""
        seed = self.config.experiment.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _setup_components(self):
        """初始化训练组件"""
        # 初始化数据模块
        self.logger.info("设置数据模块...")
        self.data_module = RealDiffusionReactionDataModule(self.config)
        self.data_module.setup()
        
        # 获取数据加载器
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
        
        # 获取归一化统计信息
        self.norm_stats = self.data_module.norm_stats
        
        # 初始化模型
        self.logger.info("设置模型...")
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器
        self.logger.info("设置优化器...")
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.logger.info("设置学习率调度器...")
        self.scheduler = self._create_scheduler()

        # 初始化混合精度训练
        self.use_amp = self.config.training.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # 初始化AR损失
        self.ar_loss_fn = ARLoss(self.config.get('loss', {}))
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # 初始化可视化器
        self.visualizer = ARTrainingVisualizer(
            output_dir=self.visualization_dir,
            config=self.config
        )
        
        # 初始化观测算子
        self._setup_observation_operator()
    
    def _create_model(self) -> nn.Module:
        """创建AR包装模型"""
        model_config = self.config.model
        
        # 基础单帧模型（SwinUNet）
        base_model = SwinUnet(
            img_size=model_config.img_size,
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            embed_dim=model_config.get('embed_dim', 96),
            depths=model_config.get('depths', [2, 2, 6, 2]),
            num_heads=model_config.get('num_heads', [3, 6, 12, 24]),
            window_size=model_config.get('window_size', 8),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            drop_rate=model_config.get('drop_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            drop_path_rate=model_config.get('drop_path_rate', 0.1),
            qkv_bias=model_config.get('qkv_bias', True),
            patch_norm=model_config.get('patch_norm', True)
        )
        
        # AR包装器
        from models.ar.wrapper import ARWrapper
        ar_cfg = model_config.get('ar_config', {})
        ar_wrapper = ARWrapper(
            single_frame_model=base_model,
            detach_rollout=ar_cfg.get('detach_rollout', True),
            scheduled_sampling=ar_cfg.get('scheduled_sampling', False),
            sampling_schedule=ar_cfg.get('sampling_schedule', None)
        )
        
        return ar_wrapper
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_config = self.config.training.optimizer
        
        optimizer_type = opt_config.get('type', 'adamw')
        lr = opt_config.get('lr', 1e-3)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config.training.get('scheduler', {})
        
        if not scheduler_config or not scheduler_config.get('enabled', True):
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.logger.warning(f"不支持的调度器类型: {scheduler_type}")
            return None
        
        return scheduler
    
    def _setup_observation_operator(self):
        """设置观测算子"""
        # 从配置中获取观测算子参数
        # 根级 observation 与 data.observation 保持一致
        obs_config = {}
        if hasattr(self.config, 'observation'):
            obs_config = dict(self.config.observation)
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'observation'):
            # 若二者都有，以根级为准，但记录二者一致性
            obs_config_data = dict(self.config.data.observation)
            # 简要一致性检查（相同键值）
            try:
                mismatch = {k: (obs_config.get(k), obs_config_data.get(k)) for k in obs_config_data.keys() if obs_config.get(k) != obs_config_data.get(k)}
                if mismatch:
                    self.logger.warning(f"观测算子配置不一致，以根级为准: {mismatch}")
            except Exception:
                pass
        
        
        if obs_config.get('enabled', False):
            # 这里可以初始化具体的观测算子
            # 例如：超分辨率、降采样等
            self.observation_op = None  # 占位符
            self.h_params = obs_config
        else:
            self.observation_op = None
            self.h_params = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        current_T_out = self.get_current_T_out(epoch)
        
        # 获取梯度累积步数
        accumulation_steps = self.config.training.get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据预处理
            input_seq = batch['input_sequence'].to(self.device)
            target_seq = batch['target_sequence'].to(self.device)
            
            # 根据当前T_out调整目标序列长度
            if target_seq.shape[1] > current_T_out:
                target_seq = target_seq[:, :current_T_out]
            
            # 前向传播
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred_seq = self.model(input_seq, current_T_out, target_seq)
                
                # 计算AR损失（简化版：仅AR序列重建损失）
                losses = self.ar_loss_fn(predictions=pred_seq, targets=target_seq)
                loss = losses['total_loss'] / accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                # 梯度裁剪
                grad_clip_val = self.config.training.get('gradient_clip_val', 1.0)
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)
                
                # 更新参数
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # 记录到TensorBoard
            if batch_idx % 100 == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/T_out', current_T_out, global_step)
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        current_T_out = self.get_current_T_out(epoch)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 数据预处理
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 根据当前T_out调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                # 前向传播
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred_seq = self.model(input_seq, current_T_out, target_seq)
                    
                    # 计算AR损失（简化版）
                    losses = self.ar_loss_fn(predictions=pred_seq, targets=target_seq)
                    loss = losses['total_loss']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def get_current_T_out(self, epoch: int) -> int:
        """根据课程学习配置返回当前T_out"""
        curriculum_config = self.config.training.get('curriculum', {})
        
        if not curriculum_config.get('enabled', False):
            return self.config.data.T_out
        
        stages = curriculum_config.get('stages', [])
        if not stages:
            return self.config.data.T_out
        
        # 累积epoch定位阶段
        total_epochs = 0
        for stage in stages:
            total_epochs += stage.get('epochs', 0)
            if epoch < total_epochs:
                return stage.get('T_out', self.config.data.T_out)
        
        # 超出课程阶段范围，返回最后一个阶段的T_out
        return stages[-1].get('T_out', self.config.data.T_out)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        # 保存最新检查点
        last_path = self.checkpoint_dir / 'last.ckpt'
        torch.save(checkpoint, last_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'best.ckpt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏅 已更新最佳检查点: {best_path}")
        
        # 周期性保存
        save_every_n = self.config.training.checkpoint.get('save_every_n_epochs', 0)
        if save_every_n > 0 and (epoch + 1) % save_every_n == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch+1:04d}.ckpt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
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
    
    def create_visualizations(self, epoch: int):
        """创建可视化"""
        try:
            # 获取一个验证批次用于可视化
            val_batch = next(iter(self.val_loader))
            
            # 创建预测
            self.model.eval()
            with torch.no_grad():
                input_seq = val_batch['input_sequence'].to(self.device)
                target_seq = val_batch['target_sequence'].to(self.device)
                current_T_out = self.get_current_T_out(epoch)
                
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred_seq = self.model(input_seq, current_T_out, target_seq)
            
            # 保存可视化
            sample_batch = {
                'input': input_seq.cpu(),
                'target': target_seq.cpu(),
                'prediction': pred_seq.cpu(),
                'epoch': epoch
            }
            
            self.visualizer.save_sample_batch(sample_batch, epoch)
            self.visualizer.save_training_curves(self.training_history)
            
            self.logger.info(f"🎨 已保存第{epoch}轮的可视化")
            
        except Exception as e:
            self.logger.warning(f"可视化创建失败: {e}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        
        num_epochs = self.config.training.epochs
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录指标
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rate'].append(current_lr)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Val/EpochLoss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # 检查是否为最佳模型
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 创建可视化
            if epoch % self.config.experiment.get('viz_interval', 10) == 0:
                self.create_visualizations(epoch)
            
            # 打印进度
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_metrics['loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f} - "
                f"LR: {current_lr:.2e} - "
                f"Best Val Loss: {self.best_val_loss:.6f}"
            )
        
        self.logger.info("训练完成！")
        
        # 最终可视化
        self.create_visualizations(num_epochs - 1)
        
        # 关闭TensorBoard
        self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于配置文件的真实扩散-反应数据AR训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子列表")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        exit(1)
    
    # 如果提供了多种子列表
    if args.seeds:
        try:
            seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except ValueError:
            print("❌ 种子格式错误，应为逗号分隔的整数")
            exit(1)
        
        if len(seed_list) < 1:
            print("❌ 至少需要提供一个种子")
            exit(1)
        
        # 运行多种子实验
        print(f"🔄 运行多种子实验: {seed_list}")
        
        for seed in seed_list:
            print(f"\n🎯 运行种子: {seed}")
            
            # 为每个种子创建临时配置
            config = OmegaConf.load(args.config)
            config.experiment.seed = seed
            
            # 更新实验名称
            base_name = config.experiment.get('name', 'real_dr_ar')
            config.experiment.name = f"{base_name}-s{seed}"
            
            # 保存临时配置
            tmp_config_path = f"tmp_config_s{seed}.yaml"
            OmegaConf.save(config, tmp_config_path)
            
            try:
                # 运行训练
                trainer = ConfigBasedARTrainer(tmp_config_path, args.resume)
                trainer.train()
                
                print(f"✅ 种子 {seed} 训练完成")
                
            except Exception as e:
                print(f"❌ 种子 {seed} 训练失败: {e}")
                continue
            
            finally:
                # 清理临时配置
                if os.path.exists(tmp_config_path):
                    os.remove(tmp_config_path)
    
    else:
        # 单次训练
        try:
            trainer = ConfigBasedARTrainer(args.config, args.resume)
            trainer.train()
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)


if __name__ == "__main__":
    main()