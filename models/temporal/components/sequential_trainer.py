"""
分阶段时空预测训练器
协调两阶段训练流程，管理阶段间数据传递
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from .sequential_spatiotemporal import (
    SequentialSpatiotemporalModel, 
    SpatialPredictionModule,
    TemporalPredictionModule
)
from .sequential_dc_consistency import SequentialConsistencyChecker


@dataclass
class TrainingMetrics:
    """训练指标"""
    spatial_loss: float
    temporal_loss: float
    dc_loss: float
    total_loss: float
    spatial_metrics: Dict[str, float]
    temporal_metrics: Dict[str, float]
    consistency_metrics: Dict[str, Dict[str, float]]


class SpatialTrainer:
    """空间预测阶段训练器"""
    
    def __init__(self, model: SpatialPredictionModule, config: Dict):
        self.model = model
        self.config = config
        self.spatial_loss_weight = config.get('spatial_loss_weight', 1.0)
        self.dc_loss_weight = config.get('dc_loss_weight', 1.0)
        
        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def train_step(self, batch: Dict[str, torch.Tensor], dc_consistency) -> Dict[str, float]:
        """
        空间训练步骤
        
        Args:
            batch: 训练批次数据
            dc_consistency: 数据一致性模块
            
        Returns:
            训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取数据
        input_data = batch['input']  # [B, T_in, C, H, W]
        target_data = batch['target']  # [B, T_out, C, H, W]
        observation = batch.get('observation')  # [B, T_out, C, H_obs, W_obs]
        
        # 前向传播
        spatial_output = self.model(input_data, target_data)
        
        # 计算重建损失
        spatial_pred = spatial_output.spatial_pred
        recon_loss = self.reconstruction_loss(spatial_pred, target_data)
        
        # 计算DC损失（如果有观测数据）
        dc_loss = 0.0
        if observation is not None:
            dc_loss = dc_consistency.compute_dc_loss(spatial_pred, observation)
        
        # 总损失
        total_loss = self.spatial_loss_weight * recon_loss + self.dc_loss_weight * dc_loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        grad_clip = self.config.get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        # 更新参数
        self.optimizer.step()
        
        # 收集指标
        metrics = {
            'spatial_loss': recon_loss.item(),
            'dc_loss': dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
            'total_loss': total_loss.item(),
            'spatial_metrics': spatial_output.spatial_metrics
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader, dc_consistency) -> Dict[str, float]:
        """验证空间模型"""
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 前向传播
                input_data = batch['input']
                target_data = batch['target']
                observation = batch.get('observation')
                
                spatial_output = self.model(input_data, target_data)
                
                # 计算损失
                spatial_pred = spatial_output.spatial_pred
                recon_loss = self.reconstruction_loss(spatial_pred, target_data)
                
                dc_loss = 0.0
                if observation is not None:
                    dc_loss = dc_consistency.compute_dc_loss(spatial_pred, observation)
                
                total_loss = self.spatial_loss_weight * recon_loss + self.dc_loss_weight * dc_loss
                
                # 累积指标
                batch_metrics = {
                    'spatial_loss': recon_loss.item(),
                    'dc_loss': dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
                    'total_loss': total_loss.item(),
                    'spatial_metrics': spatial_output.spatial_metrics
                }
                
                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    if isinstance(value, dict):
                        if key not in total_metrics or not isinstance(total_metrics[key], dict):
                            total_metrics[key] = {}
                        for k, v in value.items():
                            if k not in total_metrics[key]:
                                total_metrics[key][k] = 0.0
                            total_metrics[key][k] += v
                    else:
                        total_metrics[key] += value
                
                num_batches += 1
        
        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / num_batches
            else:
                total_metrics[key] = value / num_batches
        
        return total_metrics


class TemporalTrainer:
    """时间预测阶段训练器"""
    
    def __init__(self, model: TemporalPredictionModule, config: Dict):
        self.model = model
        self.config = config
        self.temporal_loss_weight = config.get('temporal_loss_weight', 1.0)
        self.consistency_loss_weight = config.get('consistency_loss_weight', 0.5)
        
        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def train_step(self, spatial_output, batch: Dict[str, torch.Tensor], dc_consistency) -> Dict[str, float]:
        """
        时间训练步骤
        
        Args:
            spatial_output: 空间预测输出
            batch: 训练批次数据
            dc_consistency: 数据一致性模块
            
        Returns:
            训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取数据
        target_data = batch['target']  # [B, T_out, C, H, W]
        observation = batch.get('observation')  # [B, T_out, C, H_obs, W_obs]
        
        # 前向传播
        temporal_output = self.model(spatial_output, target_data)
        
        # 计算重建损失
        final_pred = temporal_output.final_pred
        recon_loss = self.reconstruction_loss(final_pred, target_data)
        
        # 计算一致性损失（空间预测与时间预测之间）
        spatial_pred = spatial_output.spatial_pred
        consistency_loss = self.reconstruction_loss(final_pred, spatial_pred)
        
        # 计算DC损失（如果有观测数据）
        dc_loss = 0.0
        if observation is not None:
            dc_loss = dc_consistency.compute_dc_loss(final_pred, observation)
        
        # 总损失
        total_loss = (
            self.temporal_loss_weight * recon_loss + 
            self.consistency_loss_weight * consistency_loss + 
            self.temporal_loss_weight * dc_loss
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        grad_clip = self.config.get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        # 更新参数
        self.optimizer.step()
        
        # 收集指标
        metrics = {
            'temporal_loss': recon_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'dc_loss': dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
            'total_loss': total_loss.item(),
            'temporal_metrics': temporal_output.temporal_metrics
        }
        
        return metrics
    
    def validate(self, spatial_outputs, val_loader: DataLoader, dc_consistency) -> Dict[str, float]:
        """验证时间模型"""
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # 获取对应的空间输出
                spatial_output = spatial_outputs[i] if i < len(spatial_outputs) else None
                if spatial_output is None:
                    continue
                    
                target_data = batch['target']
                observation = batch.get('observation')
                
                # 前向传播
                temporal_output = self.model(spatial_output, target_data)
                
                # 计算损失
                final_pred = temporal_output.final_pred
                recon_loss = self.reconstruction_loss(final_pred, target_data)
                
                spatial_pred = spatial_output.spatial_pred
                consistency_loss = self.reconstruction_loss(final_pred, spatial_pred)
                
                dc_loss = 0.0
                if observation is not None:
                    dc_loss = dc_consistency.compute_dc_loss(final_pred, observation)
                
                total_loss = (
                    self.temporal_loss_weight * recon_loss + 
                    self.consistency_loss_weight * consistency_loss + 
                    self.temporal_loss_weight * dc_loss
                )
                
                # 累积指标
                batch_metrics = {
                    'temporal_loss': recon_loss.item(),
                    'consistency_loss': consistency_loss.item(),
                    'dc_loss': dc_loss.item() if isinstance(dc_loss, torch.Tensor) else dc_loss,
                    'total_loss': total_loss.item(),
                    'temporal_metrics': temporal_output.temporal_metrics
                }
                
                for key, value in batch_metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    if isinstance(value, dict):
                        if key not in total_metrics or not isinstance(total_metrics[key], dict):
                            total_metrics[key] = {}
                        for k, v in value.items():
                            if k not in total_metrics[key]:
                                total_metrics[key][k] = 0.0
                            total_metrics[key][k] += v
                    else:
                        total_metrics[key] += value
                
                num_batches += 1
        
        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)
        
        return total_metrics


class SequentialSpatiotemporalTrainer:
    """分阶段时空预测训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 提取子配置
        spatial_config = config.get('spatial', {})
        temporal_config = config.get('temporal', {})
        data_config = config.get('data', {})
        
        # 创建模型
        self.model = SequentialSpatiotemporalModel(
            spatial_config=spatial_config,
            temporal_config=temporal_config,
            data_config=data_config,
            device=str(self.device)
        )
        self.model.to(self.device)
        
        # 创建一致性检查器
        self.consistency_checker = SequentialConsistencyChecker(config)
        
        # 创建阶段训练器（禁用空间时跳过）
        sf_dim = int(spatial_config.get('spatial_feature_dim', spatial_config.get('feature_dim', 0)))
        bk_type = str(spatial_config.get('backbone_type', '')).lower()
        if (sf_dim == 0) or (bk_type == 'identity'):
            self.spatial_trainer = None
        else:
            self.spatial_trainer = SpatialTrainer(self.model.spatial_module, config)
        self.temporal_trainer = TemporalTrainer(self.model.temporal_module, config)
        
        # 训练配置
        self.num_epochs = config.get('num_epochs', 100)
        self.spatial_pretrain_epochs = config.get('spatial_pretrain_epochs', 10)
        self.temporal_pretrain_epochs = config.get('temporal_pretrain_epochs', 10)
        
        # 日志配置
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('SequentialTrainer')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'sequential_training.log')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        执行分阶段训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史
        """
        self.logger.info("Starting sequential spatiotemporal training")
        
        training_history = {
            'spatial_phase': [],
            'temporal_phase': [],
            'joint_phase': []
        }
        
        # 第一阶段：空间预测预训练
        self.logger.info(f"Phase 1: Spatial prediction pre-training ({self.spatial_pretrain_epochs} epochs)")
        for epoch in range(self.spatial_pretrain_epochs):
            epoch_metrics = self._train_spatial_epoch(train_loader, epoch)
            training_history['spatial_phase'].append(epoch_metrics)
            
            if val_loader is not None:
                val_metrics = self.spatial_trainer.validate(val_loader, self.consistency_checker.dc_consistency) if self.spatial_trainer is not None else {}
                self.logger.info(f"Spatial validation - Epoch {epoch}: {val_metrics}")
        
        # 第二阶段：时间预测预训练
        self.logger.info(f"Phase 2: Temporal prediction pre-training ({self.temporal_pretrain_epochs} epochs)")
        
        # 首先生成空间预测结果用于时间训练
        spatial_outputs = self._generate_spatial_outputs(train_loader)
        
        for epoch in range(self.temporal_pretrain_epochs):
            epoch_metrics = self._train_temporal_epoch(spatial_outputs, train_loader, epoch)
            training_history['temporal_phase'].append(epoch_metrics)
            
            if val_loader is not None:
                val_spatial_outputs = self._generate_spatial_outputs(val_loader)
                val_metrics = self.temporal_trainer.validate(val_spatial_outputs, val_loader, self.consistency_checker.dc_consistency)
                self.logger.info(f"Temporal validation - Epoch {epoch}: {val_metrics}")
        
        # 第三阶段：联合微调
        self.logger.info(f"Phase 3: Joint fine-tuning ({self.num_epochs - self.spatial_pretrain_epochs - self.temporal_pretrain_epochs} epochs)")
        joint_epochs = self.num_epochs - self.spatial_pretrain_epochs - self.temporal_pretrain_epochs
        
        for epoch in range(joint_epochs):
            epoch_metrics = self._train_joint_epoch(train_loader, epoch + self.spatial_pretrain_epochs + self.temporal_pretrain_epochs)
            training_history['joint_phase'].append(epoch_metrics)
            
            if val_loader is not None:
                val_metrics = self._validate_joint(val_loader)
                self.logger.info(f"Joint validation - Epoch {epoch + self.spatial_pretrain_epochs + self.temporal_pretrain_epochs}: {val_metrics}")
        
        self.logger.info("Sequential spatiotemporal training completed")
        
        return training_history
    
    def _train_spatial_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练空间预测的一个epoch"""
        total_metrics = {}
        num_batches = 0
        
        for batch in train_loader:
            # 将数据移到设备上
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 训练步骤
            batch_metrics = self.spatial_trainer.train_step(batch, self.consistency_checker.dc_consistency) if self.spatial_trainer is not None else {}
            
            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                if isinstance(value, dict):
                    if key not in total_metrics or not isinstance(total_metrics[key], dict):
                        total_metrics[key] = {}
                    for k, v in value.items():
                        if k not in total_metrics[key]:
                            total_metrics[key][k] = 0.0
                        total_metrics[key][k] += v
                else:
                    total_metrics[key] += value
            
            num_batches += 1
        
        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / num_batches
            else:
                total_metrics[key] = value / num_batches
        
        self.logger.info(f"Spatial training - Epoch {epoch}: {total_metrics}")
        
        return total_metrics
    
    def _train_temporal_epoch(self, spatial_outputs, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练时间预测的一个epoch"""
        total_metrics = {}
        num_batches = 0
        
        for i, batch in enumerate(train_loader):
            if i >= len(spatial_outputs):
                break
                
            # 将数据移到设备上
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            spatial_output = spatial_outputs[i]
            
            # 训练步骤
            batch_metrics = self.temporal_trainer.train_step(spatial_output, batch, self.consistency_checker.dc_consistency)
            
            # 累积指标
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                if isinstance(value, dict):
                    if key not in total_metrics or not isinstance(total_metrics[key], dict):
                        total_metrics[key] = {}
                    for k, v in value.items():
                        if k not in total_metrics[key]:
                            total_metrics[key][k] = 0.0
                        total_metrics[key][k] += v
                else:
                    total_metrics[key] += value
            
            num_batches += 1
        
        # 计算平均值
        for key, value in total_metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total_metrics[key][k] = v / max(num_batches, 1)
            else:
                total_metrics[key] = value / max(num_batches, 1)
        
        self.logger.info(f"Temporal training - Epoch {epoch}: {total_metrics}")
        
        return total_metrics
    
    def _train_joint_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """联合训练的一个epoch"""
        # TODO: 实现联合训练逻辑
        return {}
    
    def _validate_joint(self, val_loader: DataLoader) -> Dict[str, float]:
        """联合验证"""
        # TODO: 实现联合验证逻辑
        return {}
    
    def _generate_spatial_outputs(self, data_loader: DataLoader):
        """生成空间预测输出用于时间训练"""
        # 若禁用空间模块，则返回占位输出（使用输入序列的最后一帧作为空间预测）
        if getattr(self.model.spatial_module, 'feature_extractor', None) is None:
            spatial_outputs = []
            with torch.no_grad():
                for batch in data_loader:
                    input_data = batch['input'].to(self.device)
                    # 占位空间输出：直接使用上一帧作为空间预测，空间特征置零
                    B, T_in, C, H, W = input_data.shape
                    placeholder = {
                        'spatial_pred': input_data[:, -1:].clone(),
                        'spatial_features': torch.zeros(B, T_in, 1, H, W, device=self.device, dtype=input_data.dtype)
                    }
                    spatial_outputs.append(placeholder)
            return spatial_outputs

        self.model.spatial_module.eval()
        spatial_outputs = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_data = batch['input'].to(self.device)
                target_data = batch['target'].to(self.device) if 'target' in batch else None
                
                spatial_output = self.model.spatial_module(input_data, target_data)
                spatial_outputs.append(spatial_output)
        
        return spatial_outputs
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'spatial_optimizer_state_dict': self.spatial_trainer.optimizer.state_dict() if getattr(self, 'spatial_trainer', None) else None,
            'temporal_optimizer_state_dict': self.temporal_trainer.optimizer.state_dict() if getattr(self, 'temporal_trainer', None) else None,
            'config': self.config,
            'metrics': metrics
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.spatial_trainer is not None and checkpoint.get('spatial_optimizer_state_dict'):
            self.spatial_trainer.optimizer.load_state_dict(checkpoint['spatial_optimizer_state_dict'])
        self.temporal_trainer.optimizer.load_state_dict(checkpoint['temporal_optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        
        return checkpoint['epoch'], checkpoint['metrics']
