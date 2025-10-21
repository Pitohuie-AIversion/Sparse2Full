"""AR包装器模块

将单帧模型包装成多步自回归预测模型，支持：
- 训练：teacher forcing（每步用真值作为下一步输入）
- 推理：roll-out（每步用上一步预测作为下一步输入）

兼容现有的baseline/target标准化域处理流程。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ARWrapper(nn.Module):
    """自回归包装器
    
    将单帧模型包装成多步自回归预测：
    - 训练：teacher forcing（每步用真值作为下一步输入）
    - 推理：roll-out（每步用上一步预测作为下一步输入）
    
    Args:
        single_frame_model: 单帧预测模型
        detach_rollout: 推理/评估阶段是否断开梯度（避免梯度累积）
        scheduled_sampling: 是否启用scheduled sampling
        sampling_schedule: scheduled sampling的调度参数
    """
    
    def __init__(
        self, 
        single_frame_model: nn.Module, 
        detach_rollout: bool = True,
        scheduled_sampling: bool = False,
        sampling_schedule: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.m = single_frame_model
        self.detach_rollout = detach_rollout
        self.scheduled_sampling = scheduled_sampling
        
        # Scheduled sampling参数
        if sampling_schedule is None:
            sampling_schedule = {
                'start_prob': 0.0,
                'end_prob': 0.5,
                'schedule_type': 'linear'
            }
        self.sampling_schedule = sampling_schedule
        self.current_epoch = 0
        self.total_epochs = 100  # 默认值，会在训练时更新
        
        # 继承基础模型的属性
        if hasattr(single_frame_model, 'in_channels'):
            self.in_channels = single_frame_model.in_channels
        if hasattr(single_frame_model, 'out_channels'):
            self.out_channels = single_frame_model.out_channels
        if hasattr(single_frame_model, 'img_size'):
            self.img_size = single_frame_model.img_size
    
    def set_epoch(self, epoch: int, total_epochs: int = None):
        """设置当前epoch，用于scheduled sampling"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def get_sampling_prob(self) -> float:
        """获取当前的sampling概率"""
        if not self.scheduled_sampling:
            return 0.0
        
        progress = self.current_epoch / self.total_epochs
        start_prob = self.sampling_schedule['start_prob']
        end_prob = self.sampling_schedule['end_prob']
        
        if self.sampling_schedule['schedule_type'] == 'linear':
            return start_prob + (end_prob - start_prob) * progress
        elif self.sampling_schedule['schedule_type'] == 'exponential':
            # 指数调度
            return start_prob * (end_prob / start_prob) ** progress
        else:
            return start_prob
    
    @torch.no_grad()
    def _rollout(self, x0: torch.Tensor, T_out: int) -> torch.Tensor:
        """推理：以x0作为第1步输入，串行滚动输出T_out帧
        
        Args:
            x0: 初始输入 (B,C,H,W)
            T_out: 输出时间步数
            
        Returns:
            预测序列 (B,T_out,C,H,W)
        """
        self.m.eval()
        last = x0
        outs = []
        
        for t in range(T_out):
            y = self.m(last)  # (B,C,H,W)
            outs.append(y.unsqueeze(1))  # (B,1,C,H,W)
            
            # 准备下一步输入
            last = y if not self.detach_rollout else y.detach()
        
        return torch.cat(outs, dim=1)  # (B,T_out,C,H,W)
    
    def forward(
        self,
        x_in: torch.Tensor,                 # (B,T_in,C,H,W) 或 (B,C,H,W)
        T_out: int = 1,
        teacher: Optional[torch.Tensor] = None,  # (B,T_out,C,H,W), 训练时可传
        train_mode: Optional[bool] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x_in: 输入序列，可以是(B,C,H,W)或(B,T_in,C,H,W)
            T_out: 输出时间步数
            teacher: 教师信号，训练时使用
            train_mode: 是否为训练模式，None时使用self.training
            
        Returns:
            预测序列 (B,T_out,C,H,W)
        """
        if train_mode is None:
            train_mode = self.training
        
        # 处理输入维度
        if x_in.dim() == 4:  # (B,C,H,W)
            last = x_in
        elif x_in.dim() == 5:  # (B,T_in,C,H,W)
            last = x_in[:, -1]  # 使用最后一帧作为初始输入
        else:
            raise ValueError(f"Unsupported input dimension: {x_in.dim()}")
        
        # 推理模式或没有teacher信号时使用roll-out
        if (not train_mode) or (teacher is None):
            return self._rollout(last, T_out)
        
        # 训练模式：teacher forcing (可选scheduled sampling)
        outs = []
        sampling_prob = self.get_sampling_prob() if self.scheduled_sampling else 0.0
        
        for t in range(T_out):
            y = self.m(last)  # (B,C,H,W)
            outs.append(y.unsqueeze(1))  # (B,1,C,H,W)
            
            # 准备下一步输入
            if t < T_out - 1:  # 不是最后一步
                if self.scheduled_sampling and torch.rand(1).item() < sampling_prob:
                    # 使用模型预测作为下一步输入
                    last = y.detach()
                else:
                    # 使用真值作为下一步输入（teacher forcing）
                    last = teacher[:, t]
        
        return torch.cat(outs, dim=1)  # (B,T_out,C,H,W)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        base_info = {}
        if hasattr(self.m, 'get_model_info'):
            base_info = self.m.get_model_info()
        
        # 添加AR包装器信息
        ar_info = {
            'model_type': 'AR_Wrapper',
            'base_model': base_info.get('model_type', type(self.m).__name__),
            'detach_rollout': self.detach_rollout,
            'scheduled_sampling': self.scheduled_sampling,
        }
        
        # 合并信息
        base_info.update(ar_info)
        return base_info
    
    def compute_flops(self, input_shape: tuple = None) -> int:
        """计算FLOPs（粗略估计）
        
        Args:
            input_shape: 输入形状 (B,C,H,W)
            
        Returns:
            FLOPs数量
        """
        if hasattr(self.m, 'compute_flops'):
            base_flops = self.m.compute_flops(input_shape)
        else:
            # 简单估计
            if input_shape is None:
                input_shape = (1, getattr(self, 'in_channels', 4), 
                              getattr(self, 'img_size', 256), 
                              getattr(self, 'img_size', 256))
            
            # 估计基础模型的FLOPs
            param_count = sum(p.numel() for p in self.m.parameters())
            base_flops = param_count * input_shape[0] * input_shape[2] * input_shape[3]
        
        # AR包装器的FLOPs是基础模型的T_out倍（串行执行）
        # 这里使用默认的T_out=3进行估计
        return base_flops * 3
    
    def get_memory_usage(self, batch_size: int = 1, T_out: int = 3) -> Dict[str, float]:
        """估算显存使用量
        
        Args:
            batch_size: 批次大小
            T_out: 输出时间步数
            
        Returns:
            显存使用量信息（MB）
        """
        if hasattr(self.m, 'get_memory_usage'):
            base_memory = self.m.get_memory_usage(batch_size)
        else:
            # 简单估计
            param_memory = sum(p.numel() * p.element_size() for p in self.m.parameters()) / 1024**2
            activation_memory = batch_size * getattr(self, 'in_channels', 4) * \
                               getattr(self, 'img_size', 256)**2 * 4 / 1024**2
            base_memory = {
                'parameters_MB': param_memory,
                'activations_MB': activation_memory,
                'gradients_MB': param_memory,
                'total_MB': param_memory * 2 + activation_memory
            }
        
        # AR包装器需要额外的序列存储空间
        sequence_memory = T_out * base_memory['activations_MB']
        
        return {
            'parameters_MB': base_memory['parameters_MB'],
            'activations_MB': base_memory['activations_MB'] + sequence_memory,
            'gradients_MB': base_memory['gradients_MB'],
            'sequence_MB': sequence_memory,
            'total_MB': base_memory['total_MB'] + sequence_memory
        }
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载预训练权重（仅加载基础模型）"""
        if hasattr(self.m, 'load_pretrained'):
            self.m.load_pretrained(checkpoint_path, strict)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 只加载基础模型的权重
            self.m.load_state_dict(state_dict, strict=strict)
            logger.info(f"Loaded pretrained weights for base model from {checkpoint_path}")
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数（如果基础模型支持）"""
        if hasattr(self.m, 'freeze_encoder'):
            self.m.freeze_encoder()
    
    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def count_parameters(self) -> tuple:
        """统计模型参数
        
        Returns:
            (总参数量, 可训练参数量)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_flops(self, input_shape: tuple = None) -> int:
        """计算FLOPs（兼容接口）
        
        Args:
            input_shape: 输入形状 (B,T_in,C,H,W) 或 (B,C,H,W)
            
        Returns:
            FLOPs数量
        """
        if input_shape is not None and len(input_shape) == 5:
            # 如果是5维输入，转换为4维给基础模型
            base_input_shape = (input_shape[0], input_shape[2], input_shape[3], input_shape[4])
        else:
            base_input_shape = input_shape
        
        return self.compute_flops(base_input_shape)