#!/usr/bin/env python3
"""
课程学习训练器 - 支持渐进式、分阶段和自适应课程学习
遵循黄金法则，确保训练过程的渐进性和一致性
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from training_system.utils.trainers.trainer import PDEBenchTrainer
from training_system.utils.datasets.pde_bench import PDEBenchDataset
from training_system.utils.losses import CombinedLoss

logger = logging.getLogger(__name__)


class CurriculumTrainer(PDEBenchTrainer):
    """课程学习训练器"""
    
    def __init__(self, config: DictConfig):
        self.curriculum_config = config.curriculum
        self.current_stage = 0
        self.stage_start_epoch = 0
        self.stage_metrics = []
        
        # 初始化基础训练器
        super().__init__(config)
        
        logger.info(f"课程学习训练器初始化完成，策略: {self.curriculum_config.strategy}")
    
    def _setup_data(self):
        """设置数据加载器 - 支持课程学习"""
        logger.info("设置课程学习数据加载器...")
        
        # 获取当前阶段的配置
        current_stage_config = self._get_current_stage_config()
        
        # 合并数据配置
        data_config = OmegaConf.merge(self.config.data, current_stage_config.get('data_config', {}))
        
        # 创建数据集
        self.train_dataset = PDEBenchDataset(
            data_path=data_config.data_path,
            mode='train',
            img_size=data_config.image_size,
            normalize=data_config.preprocessing.normalize,
            observation_mode=data_config.observation.get('mode', 'super_resolution'),
            sr_scale=data_config.observation.get('scale_factor', 4),
            crop_size=data_config.observation.get('crop_size', None)
        )
        
        self.val_dataset = PDEBenchDataset(
            data_path=data_config.data_path,
            mode='val',
            img_size=data_config.image_size,
            normalize=data_config.preprocessing.normalize,
            observation_mode=data_config.observation.get('mode', 'super_resolution'),
            sr_scale=data_config.observation.get('scale_factor', 4),
            crop_size=data_config.observation.get('crop_size', None)
        )
        
        # 数据加载器
        dataloader_config = data_config.dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=getattr(dataloader_config, 'persistent_workers', True)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=getattr(dataloader_config, 'val_batch_size', dataloader_config.batch_size),
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=getattr(dataloader_config, 'persistent_workers', True)
        )
        
        logger.info(f"课程学习 - 训练样本: {len(self.train_dataset)}, 验证样本: {len(self.val_dataset)}")
    
    def _setup_loss(self):
        """设置损失函数 - 支持课程学习"""
        logger.info("设置课程学习损失函数...")
        
        # 获取当前阶段的损失权重
        current_stage_config = self._get_current_stage_config()
        loss_weights = current_stage_config.get('loss_weights', {})
        
        # 合并损失配置
        loss_config = OmegaConf.merge(self.config.loss, loss_weights)
        
        self.criterion = CombinedLoss(
            rec_weight=loss_config.rec_weight,
            spec_weight=loss_config.spec_weight,
            dc_weight=loss_config.dc_weight,
            rec_loss_type=loss_config.rec_loss_type,
            spec_loss_type=loss_config.spec_loss_type,
            dc_loss_type=loss_config.dc_loss_type,
            low_freq_modes=loss_config.low_freq_modes,
            observation_config=self.config.data.observation,
            normalization_stats=self.train_dataset.get_normalization_stats()
        )
        
        logger.info(f"课程学习 - 损失权重: rec={loss_config.rec_weight}, spec={loss_config.spec_weight}, dc={loss_config.dc_weight}")
    
    def _get_current_stage_config(self) -> Dict[str, Any]:
        """获取当前阶段的配置"""
        if self.current_stage < len(self.curriculum_config.stages):
            return self.curriculum_config.stages[self.current_stage]
        else:
            # 返回最后一个阶段的配置
            return self.curriculum_config.stages[-1]
    
    def _should_advance_stage(self, current_metrics: Dict[str, float]) -> bool:
        """判断是否应该进入下一阶段"""
        if not self.curriculum_config.adaptive.enabled:
            return False
        
        # 检查最小训练轮数
        epochs_in_stage = self.current_epoch - self.stage_start_epoch
        min_epochs = self.curriculum_config.adaptive.min_epochs_per_stage
        if epochs_in_stage < min_epochs:
            return False
        
        # 检查指标改进
        metric_name = self.curriculum_config.adaptive.metric
        improvement_threshold = self.curriculum_config.adaptive.improvement_threshold
        patience = self.curriculum_config.adaptive.patience
        
        if metric_name not in current_metrics:
            logger.warning(f"指标 {metric_name} 不存在，无法判断阶段进展")
            return False
        
        current_value = current_metrics[metric_name]
        self.stage_metrics.append(current_value)
        
        # 检查最近patience轮是否有足够改进
        if len(self.stage_metrics) >= patience:
            recent_metrics = self.stage_metrics[-patience:]
            best_recent = min(recent_metrics) if metric_name in ['rel_l2', 'mae', 'mse'] else max(recent_metrics)
            
            # 计算改进幅度
            improvement = abs(recent_metrics[0] - best_recent) / abs(recent_metrics[0])
            
            if improvement < improvement_threshold:
                logger.info(f"阶段 {self.current_stage} 指标改进不足 ({improvement:.4f} < {improvement_threshold})，准备进入下一阶段")
                return True
        
        return False
    
    def _advance_stage(self):
        """进入下一阶段"""
        if self.current_stage >= len(self.curriculum_config.stages) - 1:
            logger.info("已达到最后阶段，课程学习完成")
            return False
        
        self.current_stage += 1
        self.stage_start_epoch = self.current_epoch
        self.stage_metrics = []
        
        logger.info(f"进入课程学习阶段 {self.current_stage}: {self.curriculum_config.stages[self.current_stage].name}")
        
        # 重新配置数据和损失
        self._setup_data()
        self._setup_loss()
        
        return True
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch - 支持课程学习"""
        # 更新实验名称以包含当前阶段
        original_name = self.config.experiment.name
        self.config.experiment.name = f"{original_name}_stage_{self.current_stage}"
        
        try:
            # 调用父类方法
            epoch_losses = super().train_epoch()
            
            # 恢复原始名称
            self.config.experiment.name = original_name
            
            return epoch_losses
            
        except Exception as e:
            # 恢复原始名称
            self.config.experiment.name = original_name
            raise e
    
    def validate(self) -> Dict[str, float]:
        """验证模型 - 支持课程学习"""
        # 更新实验名称以包含当前阶段
        original_name = self.config.experiment.name
        self.config.experiment.name = f"{original_name}_stage_{self.current_stage}"
        
        try:
            # 调用父类方法
            val_metrics = super().validate()
            
            # 恢复原始名称
            self.config.experiment.name = original_name
            
            return val_metrics
            
        except Exception as e:
            # 恢复原始名称
            self.config.experiment.name = original_name
            raise e
    
    def train(self) -> Dict[str, Any]:
        """主训练循环 - 支持课程学习"""
        logger.info("开始课程学习训练循环...")
        
        training_history = {
            'train_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'resource_usage': [],
            'epoch_times': [],
            'stage_transitions': [],
            'curriculum_info': {
                'total_stages': len(self.curriculum_config.stages),
                'strategy': self.curriculum_config.strategy
            }
        }
        
        # 性能监控启动
        if self.performance_monitor:
            self.performance_monitor.start()
        
        stage_epochs = [stage.epochs for stage in self.curriculum_config.stages]
        total_planned_epochs = sum(stage_epochs)
        
        for epoch in range(total_planned_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 获取当前阶段信息
            current_stage_config = self._get_current_stage_config()
            stage_name = current_stage_config.name
            
            logger.info(f"Epoch {epoch + 1}/{total_planned_epochs} - 阶段 {self.current_stage}: {stage_name}")
            
            # 训练
            train_losses = self.train_epoch()
            training_history['train_losses'].append(train_losses)
            
            # 验证
            if epoch % self.config.validation.check_val_every_n_epoch == 0:
                val_metrics = self.validate()
                training_history['val_metrics'].append(val_metrics)
                
                # 记录验证指标
                if self.writer:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"val/stage_{self.current_stage}/{key}", value, epoch)
                
                # 检查阶段进展
                if self._should_advance_stage(val_metrics):
                    training_history['stage_transitions'].append({
                        'epoch': epoch,
                        'from_stage': self.current_stage,
                        'to_stage': self.current_stage + 1,
                        'metrics': val_metrics
                    })
                    
                    if not self._advance_stage():
                        break
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['learning_rates'].append(current_lr)
                
                if self.writer:
                    self.writer.add_scalar(f"train/stage_{self.current_stage}/learning_rate", current_lr, epoch)
            
            # 资源监控
            if self.performance_monitor and epoch % 10 == 0:
                resource_stats = self.performance_monitor.get_stats()
                training_history['resource_usage'].append(resource_stats)
                
                if self.writer:
                    for key, value in resource_stats.items():
                        self.writer.add_scalar(f"resource/{key}", value, epoch)
            
            # 保存检查点
            if epoch % self.config.training.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, val_metrics.get('rel_l2', 0) if 'val_metrics' in locals() else 0,
                    self.output_dir / f"checkpoint_stage_{self.current_stage}_epoch_{epoch}.pth"
                )
            
            # 可视化
            if self.visualizer and epoch % getattr(self.config.visualization, 'plot_interval', 50) == 0:
                self.visualizer.plot_curriculum_progress(
                    training_history, epoch, self.current_stage, self.output_dir
                )
            
            epoch_time = time.time() - epoch_start_time
            training_history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch + 1} 完成，用时: {epoch_time:.2f}s")
        
        # 停止性能监控
        if self.performance_monitor:
            self.performance_monitor.stop()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        # 保存最终模型
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.current_epoch, self.global_step, self.best_val_metric,
            self.output_dir / "final_model.pth"
        )
        
        # 保存训练历史
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2, default=str)
        
        logger.info("课程学习训练完成！")
        
        return {
            'best_val_metric': self.best_val_metric,
            'total_epochs': self.current_epoch + 1,
            'final_stage': self.current_stage,
            'training_history': training_history
        }