"""
空间预测模型工厂函数
"""

from typing import Dict, Any
from ..base import BaseModel


def create_model(model_name: str, **kwargs) -> BaseModel:
    """
    创建空间预测模型实例
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        BaseModel: 模型实例
    """
    # CNN模型
    if model_name == "UNet" or model_name == "unet":
        from .unet import UNet
        return UNet(**kwargs)
    elif model_name == "UNetPlusPlus" or model_name == "unet_plus_plus":
        from .unet_plus_plus import UNetPlusPlus
        return UNetPlusPlus(**kwargs)
    elif model_name == "FNO2D" or model_name == "fno2d":
        from .fno2d import FNO2d
        return FNO2d(**kwargs)
    elif model_name == "UFNOUNet" or model_name == "ufno_unet":
        from .ufno_unet_bottleneck import UFNOUNet
        return UFNOUNet(**kwargs)
    
    # Transformer模型
    elif model_name == "SegFormer" or model_name == "segformer":
        from .segformer import SegFormer
        return SegFormer(**kwargs)
    elif model_name == "UNetFormer" or model_name == "unetformer":
        from .unetformer import UNetFormer
        return UNetFormer(**kwargs)
    elif model_name == "SegFormerUNetFormer" or model_name == "segformer_unetformer":
        from .segformer_unetformer import SegFormerUNetFormer
        return SegFormerUNetFormer(**kwargs)
    
    # MLP模型
    elif model_name == "MLPMixer" or model_name == "mlp_mixer":
        from .mlp_mixer import MLPMixer
        return MLPMixer(**kwargs)
    elif model_name == "LIIF" or model_name == "liif":
        from .liif import LIIFModel
        return LIIFModel(**kwargs)
    elif model_name == "MLP" or model_name == "mlp":
        from .mlp import MLPModel
        return MLPModel(**kwargs)
    
    # 混合模型
    elif model_name == "SwinUNet" or model_name == "swin_unet":
        from .swin_unet import SwinUNet
        return SwinUNet(**kwargs)
    elif model_name == "Hybrid" or model_name == "hybrid":
        from .hybrid import HybridModel
        return HybridModel(**kwargs)
    
    # 基础Transformer模型
    elif model_name == "ViT" or model_name == "VisionTransformer":
        from .vit import VisionTransformer
        return VisionTransformer(**kwargs)
    elif model_name == "SwinT" or model_name == "SwinTransformerTiny":
        from .swin_t_with_encoder import SwinTWithEncoder
        return SwinTWithEncoder(**kwargs)
    elif model_name == "Transformer":
        from .transformer import Transformer
        return Transformer(**kwargs)
    elif model_name == "RestormerLite" or model_name == "restormer_lite":
        from .restormer_lite import RestormerLite
        return RestormerLite(**kwargs)
    elif model_name == "SwinIRLite" or model_name == "swinir_lite":
        from .swinir_lite import SwinIRLite
        return SwinIRLite(**kwargs)
    elif model_name == "NAFNetLite" or model_name == "nafnet_lite":
        from .nafnet_lite import NAFNetLite
        return NAFNetLite(**kwargs)
    elif model_name == "UformerLite" or model_name == "uformer_lite":
        from .uformer_lite import UformerLite
        return UformerLite(**kwargs)
    
    # 稀疏注意力模型
    elif model_name == "SparseSwinUNet" or model_name == "sparse_swin_unet":
        from .sparse_attention_encoder import SparseSwinUNet
        return SparseSwinUNet(**kwargs)
    elif model_name == "SparseAttentionEncoder" or model_name == "sparse_attention_encoder":
        from .sparse_attention_encoder import SparseAttentionEncoder
        return SparseAttentionEncoder(**kwargs)
    
    else:
        supported_models = [
            "UNet", "UNetPlusPlus", "FNO2D", "UFNOUNet",
            "SegFormer", "UNetFormer", "SegFormerUNetFormer",
            "MLPMixer", "LIIF", "MLP",
            "SwinUNet", "Hybrid",
            "ViT", "VisionTransformer", "SwinT", "SwinTransformerTiny", "Transformer",
            "SparseSwinUNet", "SparseAttentionEncoder"
        ]
        raise ValueError(f"Unknown spatial model: {model_name}. Supported models: {supported_models}")
