"""课程学习调度器模块 (Curriculum Learning Scheduler)

负责多阶段任务参数切换与训练损失权重的动态调度。
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig


class CurriculumScheduler:
    """课程学习调度器
    
    支持随训练轮次动态调整任务退化参数（如 SR scale_factor、Crop crop_ratio）
    以及各损失分量权重（如 Data Consistency、Spectral Loss）。
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        training_cfg = getattr(config, 'training', getattr(config, 'train', {}))
        self.curriculum_config = training_cfg.get('curriculum_learning', {}) if hasattr(training_cfg, 'get') else {}
        
    @property
    def is_enabled(self) -> bool:
        """检查课程学习是否启用"""
        if not self.curriculum_config:
            return False
        return bool(getattr(self.curriculum_config, 'enabled', False))

    def get_current_task_params(self, epoch: int) -> Dict[str, Any]:
        """获取当前 epoch 的任务参数"""
        params: Dict[str, Any] = {}
        
        if not self.is_enabled:
            return params
        
        # SR 任务调度
        sr_sched = getattr(self.curriculum_config, 'sr_schedule', None)
        if sr_sched and getattr(sr_sched, 'enabled', False):
            for stage in getattr(sr_sched, 'stages', []):
                epochs = getattr(stage, 'epochs', None)
                if epochs and len(epochs) == 2 and epochs[0] <= epoch < epochs[1]:
                    params['scale_factor'] = stage.scale_factor
                    break
        
        # Crop 任务调度
        crop_sched = getattr(self.curriculum_config, 'crop_schedule', None)
        if crop_sched and getattr(crop_sched, 'enabled', False):
            for stage in getattr(crop_sched, 'stages', []):
                epochs = getattr(stage, 'epochs', None)
                if epochs and len(epochs) == 2 and epochs[0] <= epoch < epochs[1]:
                    params['crop_ratio'] = stage.crop_ratio
                    break
        
        return params
    
    def get_loss_weights(self, epoch: int, total_epochs: int, base_weights: Dict[str, float]) -> Dict[str, float]:
        """获取当前 epoch 的动态损失权重"""
        if not self.is_enabled:
            return base_weights
            
        weight_sched = getattr(self.curriculum_config, 'loss_weight_schedule', None)
        if not weight_sched or not getattr(weight_sched, 'enabled', False):
            return base_weights
        
        weights = base_weights.copy()
        progress = epoch / max(1, total_epochs)
        
        # DC 损失权重调度
        if hasattr(weight_sched, 'data_consistency') or (isinstance(weight_sched, dict) and 'data_consistency' in weight_sched):
            dc_config = weight_sched['data_consistency'] if isinstance(weight_sched, dict) else weight_sched.data_consistency
            schedule_type = getattr(dc_config, 'schedule_type', 'linear')
            if schedule_type == 'linear':
                start_w = getattr(dc_config, 'start_weight', 0.0)
                end_w = getattr(dc_config, 'end_weight', 1.0)
                weights['data_consistency'] = start_w + (end_w - start_w) * progress
        
        # 频谱损失权重调度
        if hasattr(weight_sched, 'spectral') or (isinstance(weight_sched, dict) and 'spectral' in weight_sched):
            spec_config = weight_sched['spectral'] if isinstance(weight_sched, dict) else weight_sched.spectral
            schedule_type = getattr(spec_config, 'schedule_type', 'peak')
            if schedule_type == 'peak':
                peak_ratio = getattr(spec_config, 'peak_epoch_ratio', 0.5)
                start_w = getattr(spec_config, 'start_weight', 0.0)
                peak_w = getattr(spec_config, 'peak_weight', 1.0)
                end_w = getattr(spec_config, 'end_weight', 0.1)
                
                if progress <= peak_ratio:
                    factor = progress / max(1e-6, peak_ratio)
                    weights['spectral'] = start_w + (peak_w - start_w) * factor
                else:
                    factor = (progress - peak_ratio) / max(1e-6, 1.0 - peak_ratio)
                    weights['spectral'] = peak_w + (end_w - peak_w) * factor
        
        return weights
