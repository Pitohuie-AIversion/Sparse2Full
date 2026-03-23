#!/usr/bin/env python3
"""
增强时序训练脚本
集成SwinTemporalWrapper、多模式训练、课程学习等新功能
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
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
project_root = Path(__file__).resolve().parents[2]
training_dir = Path(__file__).resolve().parent
for path in (training_dir, project_root):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# 导入模块
from datasets.real_dr_dataset import RealDiffusionReactionDataModule
from models.temporal.wrappers.swin_temporal_wrapper import SwinTemporalWrapper
from models.swin_unet import SwinUNet
from models.ar.wrapper import ARWrapper
from ops.losses import compute_total_loss
from utils.metrics import compute_metrics
from utils.logger import setup_logger

# 导入可视化模块
try:
    from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer
    from utils.ar_visualizer import ARTrainingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False


def convert_numpy_types(obj):
    """递归转换numpy类型为JSON可序列化的Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class TemporalEnhancedTrainer:
    """增强时序训练器"""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """初始化训练器"""
        self.setup_config(config_path, config_dict)
        self.setup_logging()
        self.setup_device()
        self.setup_memory_management()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.setup_curriculum_learning()
        self.setup_monitoring()
        
    def setup_config(self, config_path: str = None, config_dict: Dict = None):
        """设置配置"""
        if config_dict:
            # 使用传入的配置字典，转换为DictConfig以支持点符号访问
            self.config = OmegaConf.create(config_dict)
        elif config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            # 默认配置 - 使用新的时序配置
            default_config = {
                'experiment': {
                    'name': 'Temporal-Enhanced-SwinWrapper-s2025',
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
                    'name': 'SwinTemporalWrapper',
                    'prediction_mode': 'ar',  # ar, nar, hybrid
                    'in_channels': 2,
                    'out_channels': 2,
                    'img_size': 128,
                    'T_in': 1,
                    'T_out': 20,
                    
                    # SwinUNet配置
                    'swin_config': {
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
                    
                    # 时序编码器配置
                    'temporal_encoder': {
                        'type': 'conv1d',
                        'c_out': 64,
                        'k': 3,
                        'causal': True
                    },
                    
                    # NAR预测头配置
                    'nar_head': {
                        'type': 'simple',
                        'd_model': 256,
                        'max_timesteps': 32
                    },
                    
                    # 调度采样配置
                    'scheduled_sampling': {
                        'enabled': True,
                        'initial_prob': 1.0,
                        'final_prob': 0.0,
                        'decay_type': 'exponential',
                        'decay_steps': 1000
                    }
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 16,
                    'accumulate_grad_batches': 4,
                    'optimizer': {
                        'name': 'AdamW',
                        'lr': 1e-4,
                        'weight_decay': 1e-4,
                        'betas': [0.9, 0.999]
                    },
                    'scheduler': {
                        'name': 'CosineAnnealingLR',
                        'T_max': 100,
                        'eta_min': 1e-6,
                        'warmup_epochs': 5
                    },
                    'gradient_clip_val': 1.0,
                    'amp': {
                        'enabled': True,
                        'opt_level': 'O1'
                    }
                },
                'loss': {
                    'reconstruction': {'name': 'MSELoss', 'weight': 1.0},
                    'relative_l2': {'name': 'RelativeL2Loss', 'weight': 2.0, 'eps': 1e-8},
                    'spectral': {'name': 'SpectralLoss', 'weight': 0.5, 'freq_bands': [8, 16, 32]},
                    'temporal_consistency': {'name': 'TemporalConsistencyLoss', 'weight': 0.5},
                    'stability': {'name': 'StabilityLoss', 'weight': 0.3}
                },
                'curriculum': {
                    'enabled': True,
                    'strategy': 'progressive',
                    'stages': [
                        {'epochs': 20, 'T_out': 5, 'mode': 'ar', 'description': '阶段1: AR模式预测5步'},
                        {'epochs': 40, 'T_out': 10, 'mode': 'ar', 'description': '阶段2: AR模式预测10步'},
                        {'epochs': 60, 'T_out': 15, 'mode': 'hybrid', 'description': '阶段3: 混合模式预测15步'},
                        {'epochs': 100, 'T_out': 20, 'mode': 'hybrid', 'description': '阶段4: 混合模式预测20步'}
                    ]
                },
                'validation': {
                    'check_val_every_n_epoch': 5,
                    'val_check_interval': 1.0,
                    'metrics': ['mse', 'mae', 'rel_l2', 'temporal_consistency', 'spectral_loss']
                },
                'hardware': {
                    'num_workers': 4,
                    'pin_memory': True,
                    'persistent_workers': True
                }
            }
            self.config = OmegaConf.create(default_config)

        
        # 设置随机种子
        torch.manual_seed(self.config.experiment.seed)
        np.random.seed(self.config.experiment.seed)
        
    def setup_logging(self):
        """设置日志"""
        self.output_dir = Path(self.config.experiment.output_dir) / self.config.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name="TemporalEnhancedTrainer",
            log_file=self.output_dir / "training.log",
            level=logging.INFO
        )
        
        self.logger.info(f"输出目录: {self.output_dir}")
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # 保存配置
        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(self.config, config_path)
        self.logger.info(f"配置已保存到: {config_path}")
        
    def setup_device(self):
        """设置设备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"检测到 {gpu_count} 张GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            
            self.use_multi_gpu = gpu_count > 1
        else:
            self.use_multi_gpu = False
            
        self.logger.info(f"使用设备: {self.device}")
    
    def setup_memory_management(self):
        """设置内存管理"""
        self.memory_config = {
            'gradient_accumulation_steps': self.config.training.accumulate_grad_batches,
            'memory_cleanup_frequency': 10,
            'auto_batch_size_reduction': True,
            'memory_threshold': 0.9,
        }
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
        self.logger.info(f"内存管理配置: {self.memory_config}")
        
    def setup_data(self):
        """设置数据"""
        self.logger.info("设置数据模块...")
        
        try:
            batch_size = self.config.training.batch_size
            self.logger.info(f"使用批次大小: {batch_size}")
            
            # 使用真实扩散反应数据模块
            self.data_module = RealDiffusionReactionDataModule(
                data_path=self.config.data.data_path,
                T_in=self.config.data.T_in,
                T_out=self.config.data.T_out,
                batch_size=batch_size,
                num_workers=self.config.hardware.num_workers,
                pin_memory=self.config.hardware.pin_memory,
                persistent_workers=self.config.hardware.persistent_workers,
                train_ratio=self.config.data.train_ratio,
                val_ratio=self.config.data.val_ratio,
                test_ratio=self.config.data.test_ratio,
                normalize=self.config.data.normalize,
                augmentation=self.config.data.augmentation,
                time_step_start=self.config.data.time_step_start,
                time_step_end=self.config.data.time_step_end
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
            # 创建SwinTemporalWrapper模型
            self.model = SwinTemporalWrapper(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                img_size=self.config.model.img_size,
                T_in=self.config.model.T_in,
                T_out=self.config.model.T_out,
                prediction_mode=self.config.model.prediction_mode,
                swin_config=self.config.model.swin_config,
                temporal_encoder_config=self.config.model.temporal_encoder,
                nar_head_config=self.config.model.nar_head,
                scheduled_sampling_config=self.config.model.scheduled_sampling
            )
            
            self.model = self.model.to(self.device)
            
            # 多GPU支持
            if self.use_multi_gpu:
                self.logger.info(f"🔄 启用DataParallel，使用 {torch.cuda.device_count()} 张GPU")
                self.model = nn.DataParallel(self.model)
            
            # 计算参数量
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            model_info = model_for_params.get_model_info()
            
            self.logger.info(f"✅ 模型信息: {model_info}")
            
        except Exception as e:
            self.logger.error(f"❌ 模型设置失败: {e}")
            raise
    
    def setup_optimizer(self):
        """设置优化器和调度器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # 优化器
        if self.config.training.optimizer.name == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay,
                betas=self.config.training.optimizer.betas
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer.name}")
        
        # 学习率调度器
        if self.config.training.scheduler.name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.scheduler.T_max,
                eta_min=self.config.training.scheduler.eta_min
            )
        else:
            self.scheduler = None
        
        # AMP Scaler
        self.scaler = GradScaler() if self.config.training.amp.enabled else None
        
        self.logger.info(f"✅ 优化器: {self.config.training.optimizer.name}")
        self.logger.info(f"✅ 调度器: {self.config.training.scheduler.name}")
        
    def setup_loss_functions(self):
        """设置损失函数"""
        self.logger.info("📊 设置损失函数...")
        
        self.loss_functions = {}
        self.loss_weights = {}
        
        # 重建损失
        if 'reconstruction' in self.config.loss:
            self.loss_functions['reconstruction'] = nn.MSELoss()
            self.loss_weights['reconstruction'] = self.config.loss.reconstruction.weight
        
        # 相对L2损失
        if 'relative_l2' in self.config.loss:
            def rel_l2_loss(pred, target, eps=1e-8):
                return torch.mean(torch.norm(pred - target, dim=(-2, -1)) / 
                                (torch.norm(target, dim=(-2, -1)) + eps))
            self.loss_functions['relative_l2'] = rel_l2_loss
            self.loss_weights['relative_l2'] = self.config.loss.relative_l2.weight
        
        # 频域损失
        if 'spectral' in self.config.loss:
            def spectral_loss(pred, target, freq_bands=[8, 16, 32]):
                pred_fft = torch.fft.fft2(pred.reshape(-1, *pred.shape[-2:]))
                target_fft = torch.fft.fft2(target.reshape(-1, *target.shape[-2:]))
                
                loss = 0.0
                for freq_band in freq_bands:
                    # 低频损失
                    pred_low = pred_fft[:, :freq_band, :freq_band]
                    target_low = target_fft[:, :freq_band, :freq_band]
                    loss += F.mse_loss(pred_low.real, target_low.real) + F.mse_loss(pred_low.imag, target_low.imag)
                
                return loss / len(freq_bands)
            
            self.loss_functions['spectral'] = spectral_loss
            self.loss_weights['spectral'] = self.config.loss.spectral.weight
        
        # 时序一致性损失 - 改进版本
        if 'temporal_consistency' in self.config.loss:
            def temporal_consistency_loss(pred_seq, target_seq):
                """改进的时序一致性：对齐真实动力学变化而非简单平滑"""
                # 基础：导数一致性损失 - 对齐变化模式而非惩罚变化
                pred_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
                target_diff = target_seq[:, 1:] - target_seq[:, :-1]
                
                # 相对L2损失：对齐变化幅度和方向
                diff_error = pred_diff - target_diff
                num = torch.sqrt((diff_error**2).sum(dim=(-3, -2, -1)) + 1e-8)
                den = torch.sqrt((target_diff**2).sum(dim=(-3, -2, -1)) + 1e-8)
                derivative_loss = torch.mean(num / den)
                
                # 能量变化一致性 - 匹配物理能量演化
                pred_energy = (pred_seq**2).sum(dim=(-3, -2, -1))
                target_energy = (target_seq**2).sum(dim=(-3, -2, -1))
                
                pred_energy_diff = pred_energy[:, 1:] - pred_energy[:, :-1]
                target_energy_diff = target_energy[:, 1:] - target_energy[:, :-1]
                
                energy_diff_error = torch.abs(pred_energy_diff - target_energy_diff)
                energy_diff_norm = torch.abs(target_energy_diff) + 1e-8
                energy_consistency_loss = torch.mean(energy_diff_error / energy_diff_norm)
                
                # 二阶导数一致性（曲率）- 对齐加速度变化
                pred_second_diff = pred_diff[:, 1:] - pred_diff[:, :-1]
                target_second_diff = target_diff[:, 1:] - target_diff[:, :-1]
                
                second_derivative_error = pred_second_diff - target_second_diff
                num_2nd = torch.sqrt((second_derivative_error**2).sum(dim=(-3, -2, -1)) + 1e-8)
                den_2nd = torch.sqrt((target_second_diff**2).sum(dim=(-3, -2, -1)) + 1e-8)
                curvature_loss = torch.mean(num_2nd / den_2nd)
                
                # 组合损失 - 主要关注一阶导数，辅以能量和二阶导数
                total_temporal_loss = (
                    0.6 * derivative_loss +          # 主要：变化模式对齐
                    0.3 * energy_consistency_loss +   # 辅助：能量演化匹配
                    0.1 * curvature_loss                # 微调：曲率一致性
                )
                
                return total_temporal_loss
            
            self.loss_functions['temporal_consistency'] = temporal_consistency_loss
            self.loss_weights['temporal_consistency'] = self.config.loss.temporal_consistency.weight
            
            # 添加时序一致性权重的课程学习调度
            self.temporal_weight_scheduler = self.create_temporal_weight_scheduler()
            
    def create_temporal_weight_scheduler(self):
        """创建时序一致性权重的课程学习调度器"""
        def scheduler(epoch, base_weight=self.config.loss.temporal_consistency.weight):
            """根据训练阶段动态调整时序一致性权重"""
            total_epochs = self.config.training.epochs
            
            # 阶段1：前20%训练，低权重（避免早期过约束）
            if epoch < total_epochs * 0.2:
                return base_weight * 0.3
            
            # 阶段2：20%-60%训练，逐步增加权重
            elif epoch < total_epochs * 0.6:
                progress = (epoch - total_epochs * 0.2) / (total_epochs * 0.4)
                return base_weight * (0.3 + 0.7 * progress)
            
            # 阶段3：60%-100%训练，高权重（强化时序一致性）
            else:
                return base_weight * 1.2  # 稍微超过基础权重
        
        return scheduler
        
        # 稳定性损失 - 改进版本
        if 'stability' in self.config.loss:
            def stability_loss(pred_seq, threshold=5.0):
                """改进的稳定性损失：检测发散和异常增长"""
                energy = torch.norm(pred_seq, dim=(-2, -1))
                growth = energy[:, 1:] / (energy[:, :-1] + 1e-8)
                
                # 检测发散（能量异常增长）
                divergence_penalty = torch.relu(growth - threshold)
                
                # 检测震荡（能量快速变化）
                energy_variance = torch.std(growth, dim=1)
                oscillation_penalty = torch.relu(energy_variance - 1.0)
                
                # 检测长期漂移（平均增长偏离1.0太多）
                mean_growth = torch.mean(growth, dim=1)
                drift_penalty = torch.abs(mean_growth - 1.0)
                
                # 组合稳定性约束
                total_stability_loss = (
                    torch.mean(divergence_penalty) * 0.6 +      # 主要：发散检测
                    torch.mean(oscillation_penalty) * 0.3 +    # 次要：震荡检测
                    torch.mean(drift_penalty) * 0.1             # 微调：长期漂移
                )
                
                return total_stability_loss
            
            self.loss_functions['stability'] = stability_loss
            self.loss_weights['stability'] = self.config.loss.stability.weight
        
        # AR模式下的roll-out一致性损失（可选增强）
        if self.config.get('ar_rollout_consistency', False):
            def ar_rollout_consistency_loss(pred_seq, target_seq):
                """AR模式下的roll-out一致性：对齐逐步预测与真实变化"""
                # 只在前半段应用更强的约束（避免误差累积过大）
                T = pred_seq.shape[1]
                mid_point = T // 2
                
                # 前半段：强约束
                early_pred = pred_seq[:, :mid_point]
                early_target = target_seq[:, :mid_point]
                
                # 变化一致性
                pred_diff_early = early_pred[:, 1:] - early_pred[:, :-1]
                target_diff_early = early_target[:, 1:] - early_target[:, :-1]
                
                early_consistency = F.mse_loss(pred_diff_early, target_diff_early)
                
                # 后半段：宽松约束（允许合理误差累积）
                if mid_point < T:
                    late_pred = pred_seq[:, mid_point:]
                    late_target = target_seq[:, mid_point:]
                    
                    # 只约束能量范围，不强制精确对齐
                    late_energy_pred = torch.norm(late_pred, dim=(-3, -2, -1))
                    late_energy_target = torch.norm(late_target, dim=(-3, -2, -1))
                    
                    # 允许±20%的能量偏差
                    energy_ratio = late_energy_pred / (late_energy_target + 1e-8)
                    late_consistency = torch.mean(torch.clamp(torch.abs(energy_ratio - 1.0) - 0.2, min=0.0))
                else:
                    late_consistency = 0.0
                
                return early_consistency * 0.7 + late_consistency * 0.3
            
            self.loss_functions['ar_rollout_consistency'] = ar_rollout_consistency_loss
            self.loss_weights['ar_rollout_consistency'] = self.config.get('ar_rollout_consistency_weight', 0.1)
        
        self.logger.info(f"✅ 损失函数: {list(self.loss_functions.keys())}")
        self.logger.info(f"✅ 损失权重: {self.loss_weights}")
        
    def setup_curriculum_learning(self):
        """设置课程学习"""
        self.curriculum_enabled = self.config.curriculum.enabled
        
        if self.curriculum_enabled:
            self.curriculum_stages = self.config.curriculum.stages
            self.current_stage = 0
            self.stage_epoch = 0
            
            self.logger.info("📚 课程学习已启用")
            for i, stage in enumerate(self.curriculum_stages):
                self.logger.info(f"  阶段{i+1}: {stage['description']}")
        else:
            self.logger.info("📚 课程学习已禁用")
    
    def setup_monitoring(self):
        """设置监控"""
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rate': [],
            'curriculum_stage': []
        }
        
        # 可视化器
        if VISUALIZATION_AVAILABLE:
            self.visualizer = ARTrainingVisualizer(self.output_dir / "visualizations")
        else:
            self.visualizer = None
            
        self.logger.info("📈 监控系统已设置")
    
    def get_current_curriculum_config(self, epoch: int) -> Dict[str, Any]:
        """获取当前课程学习配置"""
        if not self.curriculum_enabled:
            return {
                'T_out': self.config.model.T_out,
                'mode': self.config.model.prediction_mode,
                'stage': 0
            }
        
        cumulative_epochs = 0
        for i, stage in enumerate(self.curriculum_stages):
            cumulative_epochs += stage['epochs']
            if epoch < cumulative_epochs:
                if i != self.current_stage:
                    self.current_stage = i
                    self.stage_epoch = 0
                    self.logger.info(f"🎯 进入{stage['description']}")
                
                return {
                    'T_out': stage['T_out'],
                    'mode': stage['mode'],
                    'stage': i,
                    'description': stage['description']
                }
        
        # 如果超出所有阶段，使用最后一个阶段
        last_stage = self.curriculum_stages[-1]
        return {
            'T_out': last_stage['T_out'],
            'mode': last_stage['mode'],
            'stage': len(self.curriculum_stages) - 1,
            'description': last_stage['description']
        }
    
    def compute_total_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失 - 改进版本支持更详细的损失分解和动态权重调整"""
        total_loss = 0.0
        loss_components = {}
        detailed_components = {}
        
        # 动态调整时序一致性权重（如果存在调度器）
        if hasattr(self, 'temporal_weight_scheduler') and 'temporal_consistency' in self.loss_weights:
            current_epoch = getattr(self, 'current_epoch', 0)
            dynamic_weight = self.temporal_weight_scheduler(current_epoch)
            self.loss_weights['temporal_consistency'] = dynamic_weight
            detailed_components['temporal_consistency_dynamic_weight'] = dynamic_weight
        
        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.loss_weights[loss_name]
            
            if loss_name == 'temporal_consistency':
                # 新的时序一致性损失需要target_seq
                loss_value = loss_fn(pred_seq, target_seq)
            elif loss_name == 'stability':
                loss_value = loss_fn(pred_seq)
            else:
                loss_value = loss_fn(pred_seq, target_seq)
            
            weighted_loss = weight * loss_value
            total_loss += weighted_loss
            loss_components[loss_name] = loss_value.item()
            
            # 记录加权损失
            detailed_components[f'{loss_name}_weighted'] = weighted_loss.item()
        
        # 合并详细组件到主要组件中
        loss_components.update(detailed_components)
        
        return total_loss, loss_components
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch - 改进版本支持动态权重调整"""
        self.model.train()
        self.current_epoch = epoch  # 记录当前epoch用于权重调度
        total_loss = 0.0
        total_loss_components = {name: 0.0 for name in self.loss_functions.keys()}
        num_batches = len(self.train_loader)
        
        # 获取当前课程学习配置
        curriculum_config = self.get_current_curriculum_config(epoch)
        current_T_out = curriculum_config['T_out']
        current_mode = curriculum_config['mode']
        
        # 更新模型预测模式
        model_to_update = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_update.set_prediction_mode(current_mode)
        
        # 梯度累积配置
        accumulation_steps = self.memory_config['gradient_accumulation_steps']
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [{current_mode}]")
        
        # 初始化梯度累积
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 移动数据到设备
                input_seq = batch['input_sequence'].to(self.device, non_blocking=True)
                target_seq = batch['target_sequence'].to(self.device, non_blocking=True)
                
                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                # 前向传播
                with autocast(enabled=self.scaler is not None):
                    pred_seq = self.model(input_seq, T_out=current_T_out)
                    
                    # 计算损失
                    loss, loss_components = self.compute_total_loss(pred_seq, target_seq)
                    
                    # 梯度累积：损失除以累积步数
                    loss = loss / accumulation_steps
                
                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 每accumulation_steps步或最后一个batch时更新参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                    
                    # 更新参数
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # 清零梯度
                    self.optimizer.zero_grad()
                
                # 累积损失
                total_loss += loss.item() * accumulation_steps
                for name, value in loss_components.items():
                    total_loss_components[name] += value
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'T_out': current_T_out,
                    'Mode': current_mode,
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # 记录到TensorBoard - 改进版本支持更详细的损失记录
                if batch_idx % self.config.experiment.log_every_n_steps == 0:
                    global_step = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                    self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
                    self.writer.add_scalar('Train/T_out', current_T_out, global_step)
                    self.writer.add_scalar('Train/Stage', curriculum_config['stage'], global_step)
                    
                    # 记录损失组件 - 支持嵌套组件
                    for name, value in loss_components.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'Train/Loss_{name}', value, global_step)
                        elif isinstance(value, dict):
                            for sub_name, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    self.writer.add_scalar(f'Train/Loss_{name}_{sub_name}', sub_value, global_step)
                    
                    # 记录动态权重（如果存在）
                    if hasattr(self, 'temporal_weight_scheduler'):
                        current_temporal_weight = self.loss_weights.get('temporal_consistency', 0)
                        self.writer.add_scalar('Train/TemporalWeight_Dynamic', current_temporal_weight, global_step)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"CUDA内存不足在batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_loss_components = {name: value / num_batches for name, value in total_loss_components.items()}
        
        self.stage_epoch += 1
        
        return avg_loss, avg_loss_components
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Optional[Dict]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_loss_components = {name: 0.0 for name in self.loss_functions.keys()}
        all_metrics = []
        num_batches = len(self.val_loader)
        
        # 获取当前课程学习配置
        curriculum_config = self.get_current_curriculum_config(epoch)
        current_T_out = curriculum_config['T_out']
        current_mode = curriculum_config['mode']
        
        sample_batch = None  # 保存一个样本用于可视化
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 根据课程学习调整目标序列长度
                if target_seq.shape[1] > current_T_out:
                    target_seq = target_seq[:, :current_T_out]
                
                with autocast(enabled=self.scaler is not None):
                    pred_seq = self.model(input_seq, T_out=current_T_out)
                    loss, loss_components = self.compute_total_loss(pred_seq, target_seq)
                
                total_loss += loss.item()
                for name, value in loss_components.items():
                    total_loss_components[name] += value
                
                # 计算详细指标
                pred_np = pred_seq.cpu().numpy()
                target_np = target_seq.cpu().numpy()
                
                batch_metrics = compute_metrics(pred_np, target_np)
                all_metrics.append(batch_metrics)
                
                # 保存第一个batch用于可视化
                if batch_idx == 0:
                    sample_batch = {
                        'input_sequence': batch['input_sequence'],
                        'target_sequence': batch['target_sequence'],
                        'pred_sequence': pred_seq.cpu()
                    }
        
        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        avg_loss_components = {name: value / num_batches for name, value in total_loss_components.items()}
        
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_loss, avg_loss_components, avg_metrics, sample_batch
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始训练...")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.training.epochs):
                epoch_start_time = time.time()
                
                # 训练
                train_loss, train_loss_components = self.train_epoch(epoch)
                
                # 验证
                if epoch % self.config.validation.check_val_every_n_epoch == 0:
                    val_loss, val_loss_components, val_metrics, sample_batch = self.validate_epoch(epoch)
                    
                    # 记录历史
                    self.training_history['train_loss'].append(train_loss)
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['val_metrics'].append(val_metrics)
                    self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                    self.training_history['curriculum_stage'].append(self.current_stage)
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                    self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                    
                    for name, value in val_metrics.items():
                        self.writer.add_scalar(f'Epoch/Val_{name}', value, epoch)
                    
                    # 保存最佳模型
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_metrics = val_metrics.copy()
                        
                        # 保存检查点
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                            'best_val_loss': self.best_val_loss,
                            'best_metrics': self.best_metrics,
                            'config': OmegaConf.to_container(self.config)
                        }
                        
                        torch.save(checkpoint, self.output_dir / "best_model.pth")
                        self.logger.info(f"💾 保存最佳模型 (val_loss: {val_loss:.6f})")
                    
                    # 可视化
                    if self.visualizer and sample_batch:
                        self.visualizer.plot_predictions(
                            sample_batch['input_sequence'][:4],
                            sample_batch['target_sequence'][:4],
                            sample_batch['pred_sequence'][:4],
                            epoch=epoch,
                            save_path=self.output_dir / "visualizations" / f"epoch_{epoch:03d}.png"
                        )
                    
                    # 日志输出
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.training.epochs} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Val Rel-L2: {val_metrics.get('rel_l2', 0):.6f} | "
                        f"Time: {epoch_time:.2f}s"
                    )
                
                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()
                
                # 定期保存检查点
                if epoch % 20 == 0:
                    checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    }, checkpoint_path)
        
        except KeyboardInterrupt:
            self.logger.info("⏹️ 训练被用户中断")
        except Exception as e:
            self.logger.error(f"❌ 训练过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            total_time = time.time() - start_time
            self.logger.info(f"🏁 训练完成，总用时: {total_time/3600:.2f}小时")
            
            # 保存训练历史
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(convert_numpy_types(self.training_history), f, indent=2)
            
            # 关闭TensorBoard
            self.writer.close()
    
    def test(self):
        """测试集评估"""
        self.logger.info("🧪 开始测试集评估...")
        
        # 加载最佳模型
        checkpoint_path = self.output_dir / "best_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("✅ 已加载最佳模型")
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                input_seq = batch['input_sequence'].to(self.device)
                target_seq = batch['target_sequence'].to(self.device)
                
                # 测试时使用完整的T_out
                test_T_out = target_seq.shape[1]
                pred_seq = self.model(input_seq, T_out=test_T_out)
                
                # 计算损失
                loss, _ = self.compute_total_loss(pred_seq, target_seq)
                total_loss += loss.item()
                
                # 计算详细指标
                pred_np = pred_seq.cpu().numpy()
                target_np = target_seq.cpu().numpy()
                
                batch_metrics = compute_metrics(pred_np, target_np)
                all_metrics.append(batch_metrics)
        
        # 聚合指标
        avg_loss = total_loss / num_batches
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 保存测试结果
        test_results = {
            'test_loss': avg_loss,
            'test_metrics': avg_metrics,
            'num_samples': len(self.test_loader.dataset),
            'config': OmegaConf.to_container(self.config)
        }
        
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(convert_numpy_types(test_results), f, indent=2)
        
        self.logger.info(f"📊 测试完成，平均损失: {avg_loss:.6f}")
        self.logger.info(f"📊 测试结果已保存到: {results_path}")
        
        return test_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强时序训练脚本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--test-only", action="store_true", help="仅运行测试")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = TemporalEnhancedTrainer(args.config)
    
    if args.test_only:
        # 仅测试
        trainer.test()
    else:
        # 训练和测试
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    main()