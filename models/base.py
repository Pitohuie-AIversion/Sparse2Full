"""基础模型接口

定义统一的模型接口，确保所有模型都遵循相同的签名：
forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseModel(nn.Module, ABC):
    """统一模型接口基类
    
    所有模型必须继承此类并实现forward方法，确保接口一致性。
    
    **接口规范**：
    - __init__(in_channels, out_channels, img_size, **kwargs)
    - forward(x[B,C_in,H,W]) → y[B,C_out,H,W]
    - 支持可选的坐标编码、掩码等输入
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        img_size: int, 
        **kwargs
    ):
        """初始化基础模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            img_size: 图像尺寸（假设正方形）
            **kwargs: 其他模型特定参数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        
        # 模型配置
        self.config = kwargs
        
        # 性能统计
        self._param_count = None
        self._flops = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [B, C_in, H, W]
            **kwargs: 可选输入（coords, mask, fourier_pe等）
            
        Returns:
            输出张量 [B, C_out, H, W]
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            包含参数量、FLOPs等信息的字典
        """
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())
        
        info = {
            'name': self.__class__.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'img_size': self.img_size,
            'parameters': self._param_count,
            'parameters_M': self._param_count / 1e6,
        }
        
        if self._flops is not None:
            info['flops'] = self._flops
            info['flops_G'] = self._flops / 1e9
        
        return info
    
    def compute_flops(self, input_shape: Tuple[int, ...] = None) -> int:
        """计算FLOPs
        
        Args:
            input_shape: 输入形状，默认为(1, in_channels, img_size, img_size)
            
        Returns:
            FLOPs数量
        """
        if input_shape is None:
            input_shape = (1, self.in_channels, self.img_size, self.img_size)
        
        # 确保参数计数已初始化
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())
        
        # 简化的FLOPs计算，子类可以重写
        # 这里使用参数量的粗略估计
        self._flops = self._param_count * input_shape[0] * input_shape[2] * input_shape[3]
        
        return self._flops
    
    def get_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """估算显存使用量
        
        Args:
            batch_size: 批次大小
            
        Returns:
            显存使用量信息（MB）
        """
        # 参数显存
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        
        # 激活显存（粗略估计）
        activation_memory = batch_size * self.in_channels * self.img_size**2 * 4 / 1024**2  # float32
        
        # 梯度显存
        grad_memory = param_memory  # 与参数相同
        
        total_memory = param_memory + activation_memory + grad_memory
        
        return {
            'parameters_MB': param_memory,
            'activations_MB': activation_memory,
            'gradients_MB': grad_memory,
            'total_MB': total_memory
        }
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载预训练权重
        
        Args:
            checkpoint_path: 检查点路径
            strict: 是否严格匹配键名
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除不匹配的键
        if not strict:
            model_keys = set(self.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            # 移除多余的键
            extra_keys = checkpoint_keys - model_keys
            for key in extra_keys:
                del state_dict[key]
            
            # 报告缺失的键
            missing_keys = model_keys - checkpoint_keys
            if missing_keys:
                print(f"Missing keys in checkpoint: {missing_keys}")
        
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数（如果适用）"""
        # 子类可以重写此方法
        pass
    
    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True


def create_model(model_name: str, **kwargs) -> BaseModel:
    """模型工厂函数
    
    根据模型名称和参数创建相应的模型实例
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        模型实例
    """
    # 动态导入模型类
    model_name_lower = model_name.lower()
    
    # 基线模型
    if model_name_lower in ['liif']:
        from .liif import LIIFModel
        model_class = LIIFModel
    elif model_name_lower in ['unet']:
        from .unet import UNet
        model_class = UNet
    elif model_name_lower in ['unet_plus_plus', 'unetplusplus']:
        from .unet_plus_plus import UNetPlusPlus
        model_class = UNetPlusPlus
    elif model_name_lower in ['fno2d']:
        from .fno2d import FNO2d
        model_class = FNO2d
    elif model_name_lower in ['ufno_unet', 'ufnounet']:
        from .ufno_unet_bottleneck import UFNOUNet
        model_class = UFNOUNet
    
    # Transformer模型
    elif model_name_lower in ['swin_unet', 'swinunet']:
        from .swin_unet import SwinUNet
        model_class = SwinUNet
    elif model_name_lower in ['segformer_unetformer', 'segformerunetformer']:
        from .segformer_unetformer import SegFormerUNetFormer
        model_class = SegFormerUNetFormer
    elif model_name_lower in ['unetformer']:
        from .unetformer import UNetFormer
        model_class = UNetFormer
    elif model_name_lower in ['segformer']:
        from .segformer import SegFormer
        model_class = SegFormer
    
    # MLP模型
    elif model_name_lower in ['mlp', 'mlpmodel']:
        from .mlp import MLPModel
        model_class = MLPModel
    elif model_name_lower in ['mlp_mixer', 'mlpmixer']:
        from .mlp_mixer import MLPMixer
        model_class = MLPMixer
    elif model_name_lower in ['liif', 'liifmodel']:
        from .liif import LIIFModel
        model_class = LIIFModel
    
    # 混合模型
    elif model_name_lower in ['hybrid', 'hybridmodel']:
        from .hybrid import HybridModel
        model_class = HybridModel
    
    # 新增Transformer模型
    elif model_name_lower in ['vit', 'visiontransformer']:
        from .vit import VisionTransformer
        model_class = VisionTransformer
    elif model_name_lower in ['swint', 'swintransformertiny']:
        from .swin_t import SwinTransformerTiny
        model_class = SwinTransformerTiny
    elif model_name_lower in ['transformer']:
        from .transformer import Transformer
        model_class = Transformer
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 创建模型
    model = model_class(**kwargs)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_model_size(model: nn.Module) -> float:
    """获取模型大小（MB）
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return size_mb