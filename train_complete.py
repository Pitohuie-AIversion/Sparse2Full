#!/usr/bin/env python3
"""完整的PDEBench稀疏观测重建训练脚本

支持分布式训练、AMP、课程学习、数据一致性验证等完整功能

使用方法:
    # 单卡训练
    python train_complete.py --config configs/train/default.yaml
    
    # 多卡训练
    torchrun --nproc_per_node=2 train_complete.py --config configs/train/default.yaml
    
    # 指定实验名称
    python train_complete.py --config configs/train/default.yaml --exp_name my_experiment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from train import Trainer
from utils.logging import setup_logger
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process


class CompleteTrainer(Trainer):
    """完整功能的训练器
    
    继承基础Trainer，添加分布式训练和完整的实验管理功能
    """
    
    def __init__(self, config: DictConfig, exp_name: Optional[str] = None):
        """
        Args:
            config: 训练配置
            exp_name: 实验名称（可选）
        """
        # 设置分布式训练
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_distributed = self.world_size > 1
        
        if self.is_distributed:
            setup_distributed()
            torch.cuda.set_device(self.local_rank)
            config.device = f'cuda:{self.local_rank}'
        
        # 生成实验名称
        if exp_name is None:
            exp_name = self._generate_experiment_name(config)
        
        # 设置实验目录
        self.exp_dir = Path(config.get('runs_dir', 'runs')) / exp_name
        if is_main_process():
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 同步所有进程
        if self.is_distributed:
            dist.barrier()
        
        # 设置日志
        if is_main_process():
            setup_logger(self.exp_dir / 'train.log')
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Starting experiment: {exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")
            
            # 保存配置快照
            config_snapshot = OmegaConf.to_yaml(config)
            with open(self.exp_dir / 'config_merged.yaml', 'w') as f:
                f.write(config_snapshot)
            
            # 保存环境信息
            self._save_environment_info()
        
        # 更新配置中的实验目录
        config.exp_dir = str(self.exp_dir)
        
        # 初始化基础训练器
        super().__init__(config)
    
    def _generate_experiment_name(self, config: DictConfig) -> str:
        """生成标准化的实验名称
        
        格式: <task>-<data>-<res>-<model>-<keyhyper>-<seed>-<date>
        
        Args:
            config: 训练配置
            
        Returns:
            exp_name: 实验名称
        """
        from datetime import datetime
        
        # 任务类型
        task = config.get('task', 'SR')
        if 'sr' in config.get('data', {}).get('name', '').lower():
            scale = config.get('data', {}).get('scale_factor', 4)
            task = f"SRx{scale}"
        elif 'crop' in config.get('data', {}).get('name', '').lower():
            crop_ratio = config.get('data', {}).get('crop_ratio', 0.2)
            task = f"Crop{int(crop_ratio*100)}"
        
        # 数据集
        data_name = config.get('data', {}).get('name', 'DR2D')
        
        # 分辨率
        img_size = config.get('data', {}).get('img_size', 256)
        resolution = f"{img_size}"
        
        # 模型
        model_name = config.get('model', {}).get('name', 'swin_unet')
        model_params = []
        
        if 'swin' in model_name.lower():
            window_size = config.get('model', {}).get('window_size', 8)
            depths = config.get('model', {}).get('depths', [2, 2, 6, 2])
            embed_dim = config.get('model', {}).get('embed_dim', 96)
            model_params.append(f"w{window_size}")
            model_params.append(f"d{''.join(map(str, depths))}")
            model_params.append(f"e{embed_dim}")
        
        model_str = model_name
        if model_params:
            model_str += "_" + "_".join(model_params)
        
        # 种子
        seed = config.get('seed', 42)
        
        # 日期
        date = datetime.now().strftime("%Y%m%d")
        
        # 组合实验名称
        exp_name = f"{task}-{data_name}-{resolution}-{model_str}-s{seed}-{date}"
        
        return exp_name
    
    def _save_environment_info(self) -> None:
        """保存环境信息"""
        import subprocess
        import platform
        
        env_info = {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'platform': platform.platform(),
            'hostname': platform.node(),
        }
        
        # Git信息
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_commit'] = git_commit
        except:
            env_info['git_commit'] = 'unknown'
        
        # 保存环境信息
        import json
        with open(self.exp_dir / 'environment.json', 'w') as f:
            json.dump(env_info, f, indent=2)
    
    def _init_data(self) -> None:
        """初始化数据加载器（支持分布式）"""
        super()._init_data()
        
        # 如果是分布式训练，需要重新包装DataLoader
        if self.is_distributed:
            # 训练集
            train_sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.train.batch_size,
                sampler=train_sampler,
                num_workers=self.config.train.get('num_workers', 4),
                pin_memory=True,
                drop_last=True
            )
            
            # 验证集
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.train.get('val_batch_size', self.config.train.batch_size),
                sampler=val_sampler,
                num_workers=self.config.train.get('num_workers', 4),
                pin_memory=True,
                drop_last=False
            )
            
            self.train_sampler = train_sampler
            self.val_sampler = val_sampler
    
    def _init_model(self) -> None:
        """初始化模型（支持分布式）"""
        super()._init_model()
        
        # 分布式训练包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            
            if is_main_process():
                self.logger.info(f"Model wrapped with DDP on {self.world_size} GPUs")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch（分布式版本）"""
        # 设置sampler的epoch（用于shuffle）
        if self.is_distributed and hasattr(self, 'train_sampler'):
            self.train_sampler.set_epoch(epoch)
        
        # 调用父类方法
        epoch_metrics = super().train_epoch(epoch)
        
        # 分布式聚合指标
        if self.is_distributed:
            epoch_metrics = self._reduce_metrics(epoch_metrics)
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch（分布式版本）"""
        # 调用父类方法
        val_metrics = super().validate_epoch(epoch)
        
        # 分布式聚合指标
        if self.is_distributed:
            val_metrics = self._reduce_metrics(val_metrics)
        
        return val_metrics
    
    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """分布式聚合指标"""
        reduced_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_metrics[key] = (tensor / self.world_size).item()
            else:
                reduced_metrics[key] = value
        
        return reduced_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False) -> None:
        """保存检查点（只在主进程）"""
        if not is_main_process():
            return
        
        # 获取模型状态（处理DDP包装）
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        # 保存最新检查点
        checkpoint_path = self.exp_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.exp_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        # 定期保存检查点
        if epoch % self.config.train.get('save_freq', 50) == 0:
            epoch_path = self.exp_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.is_distributed:
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='PDEBench Sparse Observation Reconstruction Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 调试模式
    if args.debug:
        config.train.epochs = 2
        config.train.val_freq = 1
        config.train.log_freq = 1
        warnings.filterwarnings('ignore')
    
    try:
        # 创建训练器
        trainer = CompleteTrainer(config, args.exp_name)
        
        # 恢复训练
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
            if is_main_process():
                trainer.logger.info(f"Resumed training from epoch {start_epoch}")
        
        # 开始训练
        if is_main_process():
            trainer.logger.info("Starting training...")
        
        trainer.train(start_epoch)
        
        if is_main_process():
            trainer.logger.info("Training completed successfully!")
    
    except Exception as e:
        if is_main_process():
            logging.error(f"Training failed: {e}")
        raise
    
    finally:
        # 清理资源
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == '__main__':
    main()