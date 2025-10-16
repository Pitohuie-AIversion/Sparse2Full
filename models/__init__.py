"""模型模块

提供各种深度学习模型的实现，包括基线模型和先进模型。
所有模型遵循统一接口：forward(x[B,C,H,W]) -> y[B,C,H,W]

支持的模型：
- 基线模型：U-Net, FNO2D, U-FNO瓶颈
- Transformer模型：SegFormer/UNetFormer
- MLP模型：MLP-Mixer, LIIF-Head
- 混合模型：SwinUNet, Hybrid, MLP
"""

# 基线模型
from .unet import UNet
from .fno2d import FNO2d

# 扩展基线模型
from .unet_plus_plus import UNetPlusPlus
from .ufno_unet_bottleneck import UFNOUNet

# Transformer模型
from .segformer_unetformer import SegFormerUNetFormer
from .segformer import SegFormer
from .unetformer import UNetFormer

# MLP模型
from .mlp import MLPModel
from .mlp_mixer import MLPMixer
from .liif import LIIFModel

# 混合模型
from .swin_unet import SwinUNet
from .hybrid import HybridModel

# Transformer模型（新增）
from .vit import VisionTransformer, ViT
from .swin_t import SwinTransformerTiny, SwinT
from .transformer import Transformer


def create_model(model_name_or_config, **kwargs):
    """
    根据配置创建模型实例
    
    Args:
        model_name_or_config: 模型名称字符串或配置对象
        **kwargs: 模型参数（当第一个参数是字符串时使用）
        
    Returns:
        torch.nn.Module: 模型实例
        
    Raises:
        ValueError: 当模型名称不支持时
    """
    if isinstance(model_name_or_config, str):
        # 直接传入模型名称和参数
        model_name = model_name_or_config
        model_params = kwargs
    else:
        # 传入配置对象
        config = model_name_or_config
        model_name = config.model.name
        model_params = config.model.params
    
    # 基线模型
    if model_name == "UNet" or model_name == "unet":
        return UNet(**model_params)
    elif model_name == "UNetPlusPlus" or model_name == "unet_plus_plus":
        return UNetPlusPlus(**model_params)
    elif model_name == "FNO2D" or model_name == "fno2d":
        return FNO2d(**model_params)
    elif model_name == "UFNOUNet" or model_name == "ufno_unet":
        return UFNOUNet(**model_params)
    
    # Transformer模型
    elif model_name == "SegFormer" or model_name == "segformer":
        return SegFormer(**model_params)
    elif model_name == "UNetFormer" or model_name == "unetformer":
        return UNetFormer(**model_params)
    elif model_name == "SegFormerUNetFormer" or model_name == "segformer_unetformer":
        return SegFormerUNetFormer(**model_params)
    
    # MLP模型
    elif model_name == "MLPMixer" or model_name == "mlp_mixer":
        return MLPMixer(**model_params)
    elif model_name == "LIIF" or model_name == "liif":
        return LIIFModel(**model_params)
    
    # 混合模型
    elif model_name == "SwinUNet" or model_name == "swin_unet":
        return SwinUNet(**model_params)
    elif model_name == "Hybrid" or model_name == "hybrid":
        return HybridModel(**model_params)
    elif model_name == "MLP" or model_name == "mlp":
        return MLPModel(**model_params)
    
    # 新增Transformer模型
    elif model_name == "ViT" or model_name == "VisionTransformer":
        return VisionTransformer(**model_params)
    elif model_name == "SwinT" or model_name == "SwinTransformerTiny":
        return SwinTransformerTiny(**model_params)
    elif model_name == "Transformer":
        return Transformer(**model_params)
    
    else:
        supported_models = [
            "UNet", "UNetPlusPlus", "FNO2D", "UFNOUNet",
            "SegFormer", "UNetFormer", "SegFormerUNetFormer",
            "MLPMixer", "LIIF",
            "SwinUNet", "Hybrid", "MLP",
            "ViT", "VisionTransformer", "SwinT", "SwinTransformerTiny",
            "Transformer"
        ]
        raise ValueError(f"Unknown model: {model_name}. Supported models: {supported_models}")


__all__ = [
    # 基线模型
    "UNet",
    "UNetPlusPlus",
    "FNO2d",
    "UFNOUNet",
    
    # Transformer模型
    "SegFormerUNetFormer",
    "SegFormer",
    "UNetFormer",
    "VisionTransformer",
    "ViT",
    "SwinTransformerTiny",
    "SwinT",
    "Transformer",
    
    # MLP模型
    "MLPModel",
    "MLPMixer",
    "LIIFModel",
    
    # 混合模型
    "SwinUNet",
    "HybridModel",
    
    # 工厂函数
    "create_model",
    "get_model"
]

# 别名函数，保持向后兼容
def get_model(model_name, **kwargs):
    """获取模型实例（向后兼容）"""
    class Config:
        def __init__(self, name, params):
            self.model = type('Model', (), {'name': name, 'params': params})()
    
    config = Config(model_name, kwargs)
    return create_model(config)