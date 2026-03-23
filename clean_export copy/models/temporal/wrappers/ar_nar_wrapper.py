"""AR-NAR双头包装器

统一管理AR和NAR的训练和推理，支持：
1. 双头并行训练
2. 单头推理切换
3. 损失权重调度
4. 性能监控
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union, List
import logging
from dataclasses import dataclass

from .swin_temporal import SwinTemporalNAR

logger = logging.getLogger(__name__)


@dataclass
class ARNAROutput:
    """AR-NAR输出结构"""
    ar_pred: Optional[torch.Tensor] = None
    nar_pred: Optional[torch.Tensor] = None
    ar_loss: Optional[torch.Tensor] = None
    nar_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None


class ARNARWrapper(nn.Module):
    """AR-NAR双头包装器
    
    统一管理AR和NAR模型的训练和推理。
    支持动态权重调度、性能监控和推理模式切换。
    
    Args:
        model_config: 模型配置
        loss_config: 损失配置
        training_config: 训练配置
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        super().__init__()
        
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        
        # 创建双头模型
        self.model = SwinTemporalNAR(
            base_kwargs=model_config.get('base_kwargs', {}),
            temporal_cfg=model_config.get('temporal', {}),
            nar_cfg=model_config.get('nar', {}),
            ar_cfg=model_config.get('ar', {}),
            use_ar=model_config.get('use_ar', True),
            use_nar=model_config.get('use_nar', True)
        )
        
        # 损失权重调度
        self.ar_weight_schedule = loss_config.get('ar_weight_schedule', 'constant')
        self.nar_weight_schedule = loss_config.get('nar_weight_schedule', 'constant')
        self.base_ar_weight = loss_config.get('ar_weight', 1.0)
        self.base_nar_weight = loss_config.get('nar_weight', 1.0)
        
        # 推理模式
        self.inference_mode = training_config.get('inference_mode', 'nar')  # 'ar', 'nar', 'ensemble'
        
        # 训练状态
        self.current_epoch = 0
        self.total_epochs = training_config.get('total_epochs', 100)
        
        # 性能监控
        self.enable_monitoring = training_config.get('enable_monitoring', True)
        self.monitoring_interval = training_config.get('monitoring_interval', 100)
        self.step_count = 0
        
        logger.info(f"ARNARWrapper initialized: AR={self.model.use_ar}, NAR={self.model.use_nar}")
    
    def forward(
        self,
        x_seq: torch.Tensor,
        T_out: int = 1,
        teacher_seq: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        target_seq: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, ARNAROutput]:
        """前向传播
        
        Args:
            x_seq: 输入序列 (B,T_in,C,H,W)
            T_out: 输出时间步数
            teacher_seq: 教师信号 (B,T_out,C,H,W)
            compute_loss: 是否计算损失
            target_seq: 目标序列 (B,T_out,C,H,W)，用于损失计算
            
        Returns:
            训练时返回ARNAROutput，推理时返回预测张量
        """
        if self.training and compute_loss:
            return self._forward_train(x_seq, T_out, teacher_seq, target_seq)
        else:
            return self._forward_inference(x_seq, T_out)
    
    def _forward_train(
        self,
        x_seq: torch.Tensor,
        T_out: int,
        teacher_seq: Optional[torch.Tensor],
        target_seq: Optional[torch.Tensor]
    ) -> ARNAROutput:
        """训练前向传播"""
        # 获取双头预测
        ar_pred, nar_pred = self.model(
            x_seq=x_seq,
            T_out=T_out,
            teacher_seq=teacher_seq,
            train_mode=True,
            return_both=True
        )
        
        # 计算损失
        ar_loss = None
        nar_loss = None
        total_loss = None
        
        if target_seq is not None:
            # AR损失
            if ar_pred is not None and self.model.use_ar:
                ar_loss = self._compute_ar_loss(ar_pred, target_seq)
            
            # NAR损失
            if nar_pred is not None and self.model.use_nar:
                nar_loss = self._compute_nar_loss(nar_pred, target_seq)
            
            # 总损失
            total_loss = self._compute_total_loss(ar_loss, nar_loss)
        
        # 计算指标
        metrics = self._compute_metrics(ar_pred, nar_pred, target_seq)
        
        # 性能监控
        if self.enable_monitoring:
            self._update_monitoring(ar_pred, nar_pred, ar_loss, nar_loss)
        
        return ARNAROutput(
            ar_pred=ar_pred,
            nar_pred=nar_pred,
            ar_loss=ar_loss,
            nar_loss=nar_loss,
            total_loss=total_loss,
            metrics=metrics
        )
    
    def _forward_inference(
        self,
        x_seq: torch.Tensor,
        T_out: int
    ) -> torch.Tensor:
        """推理前向传播"""
        if self.inference_mode == 'ar' and self.model.use_ar:
            # 仅使用AR
            ar_pred, _ = self.model(
                x_seq=x_seq,
                T_out=T_out,
                train_mode=False,
                return_both=True
            )
            return ar_pred
        
        elif self.inference_mode == 'nar' and self.model.use_nar:
            # 仅使用NAR
            _, nar_pred = self.model(
                x_seq=x_seq,
                T_out=T_out,
                train_mode=False,
                return_both=True
            )
            return nar_pred
        
        elif self.inference_mode == 'ensemble' and self.model.use_ar and self.model.use_nar:
            # 集成预测
            ar_pred, nar_pred = self.model(
                x_seq=x_seq,
                T_out=T_out,
                train_mode=False,
                return_both=True
            )
            
            # 简单平均集成
            ensemble_weight = self.training_config.get('ensemble_weight', 0.5)
            return ensemble_weight * ar_pred + (1 - ensemble_weight) * nar_pred
        
        else:
            # 回退到可用的模式
            ar_pred, nar_pred = self.model(
                x_seq=x_seq,
                T_out=T_out,
                train_mode=False,
                return_both=True
            )
            return nar_pred if nar_pred is not None else ar_pred
    
    def _compute_ar_loss(
        self,
        ar_pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算AR损失"""
        # 使用MSE损失
        return nn.functional.mse_loss(ar_pred, target)
    
    def _compute_nar_loss(
        self,
        nar_pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算NAR损失"""
        # 使用MSE损失
        return nn.functional.mse_loss(nar_pred, target)
    
    def _compute_total_loss(
        self,
        ar_loss: Optional[torch.Tensor],
        nar_loss: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """计算总损失"""
        # 获取当前权重
        ar_weight = self._get_current_ar_weight()
        nar_weight = self._get_current_nar_weight()
        
        total = torch.tensor(0.0, device=ar_loss.device if ar_loss is not None else nar_loss.device)
        
        if ar_loss is not None:
            total += ar_weight * ar_loss
        
        if nar_loss is not None:
            total += nar_weight * nar_loss
        
        return total
    
    def _get_current_ar_weight(self) -> float:
        """获取当前AR权重"""
        if self.ar_weight_schedule == 'constant':
            return self.base_ar_weight
        elif self.ar_weight_schedule == 'decay':
            # 随训练进度衰减
            progress = self.current_epoch / self.total_epochs
            return self.base_ar_weight * (1 - progress)
        elif self.ar_weight_schedule == 'warmup':
            # 预热增长
            progress = min(1.0, self.current_epoch / (self.total_epochs * 0.3))
            return self.base_ar_weight * progress
        else:
            return self.base_ar_weight
    
    def _get_current_nar_weight(self) -> float:
        """获取当前NAR权重"""
        if self.nar_weight_schedule == 'constant':
            return self.base_nar_weight
        elif self.nar_weight_schedule == 'increase':
            # 随训练进度增长
            progress = self.current_epoch / self.total_epochs
            return self.base_nar_weight * (0.5 + 0.5 * progress)
        elif self.nar_weight_schedule == 'warmup':
            # 预热增长
            progress = min(1.0, self.current_epoch / (self.total_epochs * 0.5))
            return self.base_nar_weight * progress
        else:
            return self.base_nar_weight
    
    def _compute_metrics(
        self,
        ar_pred: Optional[torch.Tensor],
        nar_pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        if target is None:
            return metrics
        
        # AR指标
        if ar_pred is not None:
            ar_mse = nn.functional.mse_loss(ar_pred, target).item()
            ar_mae = nn.functional.l1_loss(ar_pred, target).item()
            metrics.update({
                'ar_mse': ar_mse,
                'ar_mae': ar_mae,
                'ar_rmse': ar_mse ** 0.5
            })
        
        # NAR指标
        if nar_pred is not None:
            nar_mse = nn.functional.mse_loss(nar_pred, target).item()
            nar_mae = nn.functional.l1_loss(nar_pred, target).item()
            metrics.update({
                'nar_mse': nar_mse,
                'nar_mae': nar_mae,
                'nar_rmse': nar_mse ** 0.5
            })
        
        # 比较指标
        if ar_pred is not None and nar_pred is not None:
            ar_mse = metrics.get('ar_mse', float('inf'))
            nar_mse = metrics.get('nar_mse', float('inf'))
            metrics['nar_vs_ar_ratio'] = nar_mse / ar_mse if ar_mse > 0 else 1.0
        
        return metrics
    
    def _update_monitoring(
        self,
        ar_pred: Optional[torch.Tensor],
        nar_pred: Optional[torch.Tensor],
        ar_loss: Optional[torch.Tensor],
        nar_loss: Optional[torch.Tensor]
    ):
        """更新性能监控"""
        self.step_count += 1
        
        if self.step_count % self.monitoring_interval == 0:
            # 记录权重
            ar_weight = self._get_current_ar_weight()
            nar_weight = self._get_current_nar_weight()
            
            logger.info(
                f"Step {self.step_count}: AR_weight={ar_weight:.3f}, NAR_weight={nar_weight:.3f}"
            )
            
            # 记录损失
            if ar_loss is not None:
                logger.info(f"AR_loss: {ar_loss.item():.6f}")
            if nar_loss is not None:
                logger.info(f"NAR_loss: {nar_loss.item():.6f}")
    
    def set_epoch(self, epoch: int, total_epochs: int = None):
        """设置训练epoch"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
        
        # 传递给子模型
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch, total_epochs)
    
    def set_inference_mode(self, mode: str):
        """设置推理模式
        
        Args:
            mode: 'ar', 'nar', 'ensemble'
        """
        if mode not in ['ar', 'nar', 'ensemble']:
            raise ValueError(f"Invalid inference mode: {mode}")
        
        self.inference_mode = mode
        logger.info(f"Inference mode set to: {mode}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        base_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        
        return {
            'wrapper_type': 'ARNARWrapper',
            'inference_mode': self.inference_mode,
            'ar_weight_schedule': self.ar_weight_schedule,
            'nar_weight_schedule': self.nar_weight_schedule,
            'current_ar_weight': self._get_current_ar_weight(),
            'current_nar_weight': self._get_current_nar_weight(),
            'model_info': base_info,
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> Dict[str, int]:
        """计算FLOPs"""
        if hasattr(self.model, 'compute_flops'):
            return self.model.compute_flops(input_shape)
        else:
            # 简单估计
            total_params = sum(p.numel() for p in self.parameters())
            if input_shape is not None:
                B, T, C, H, W = input_shape
                flops = total_params * B * T * H * W
            else:
                flops = total_params * 256 * 256  # 默认估计
            
            return {'total': flops}
    
    def get_memory_usage(self, batch_size: int = 1, T_out: int = 3) -> Dict[str, float]:
        """估算显存使用量"""
        if hasattr(self.model, 'get_memory_usage'):
            return self.model.get_memory_usage(batch_size, T_out)
        else:
            # 简单估计
            total_params = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
            return {'total_MB': total_params}


# 导出接口
__all__ = [
    'ARNARWrapper',
    'ARNAROutput'
]