"""
空间预测模型模块

采用惰性导入，避免框架初始化阶段预加载无关模型层。
"""


def _getattr(module, name: str):
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __getattr__(name: str):
    if name == "UNet":
        from . import unet
        return _getattr(unet, name)
    elif name == "UNetPlusPlus":
        from . import unet_plus_plus
        return _getattr(unet_plus_plus, name)
    elif name == "FNO2d":
        from . import fno2d
        return _getattr(fno2d, name)
    elif name == "UFNOUNet":
        from . import ufno_unet_bottleneck
        return _getattr(ufno_unet_bottleneck, name)
    elif name == "EDSR":
        from . import edsr
        return _getattr(edsr, name)
    elif name in ("SparseSwinUNet", "SparseAttentionEncoder"):
        from . import sparse_attention_encoder
        return _getattr(sparse_attention_encoder, name)
    elif name == "SegFormer":
        from . import segformer
        return _getattr(segformer, name)
    elif name == "UNetFormer":
        from . import unetformer
        return _getattr(unetformer, name)
    elif name == "SegFormerUNetFormer":
        from . import segformer_unetformer
        return _getattr(segformer_unetformer, name)
    elif name in ("MLPModel", "MLP"):
        from . import mlp
        return _getattr(mlp, name)
    elif name == "MLPMixer":
        from . import mlp_mixer
        return _getattr(mlp_mixer, name)
    elif name == "LIIFModel":
        from . import liif
        return _getattr(liif, name)
    elif name == "HybridModel":
        from . import hybrid
        return _getattr(hybrid, name)
    elif name in ("VisionTransformer", "ViT"):
        from . import vit
        return _getattr(vit, name)
    elif name in ("SwinTransformerTiny", "SwinT"):
        from . import swin_t
        return _getattr(swin_t, name)
    elif name == "SwinUNet":
        from . import swin_unet
        return _getattr(swin_unet, name)
    elif name in ("ResNetLite", "SwinIRLite"):
        from . import resnet
        return _getattr(resnet, name)
    elif name in ("ConvGateLite", "NAFNetLite"):
        from . import conv_gate_lite
        return _getattr(conv_gate_lite, name)
    elif name in ("ConvUNetLite", "UformerLite"):
        from . import conv_unet_lite
        return _getattr(conv_unet_lite, name)
    elif name == "Transformer":
        from . import transformer
        return _getattr(transformer, name)
    elif name == "create_model":
        from .factory import create_model
        return create_model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "UNet",
    "UNetPlusPlus", 
    "FNO2d",
    "UFNOUNet",
    "EDSR",
    "SparseSwinUNet",
    "SegFormer",
    "UNetFormer",
    "SegFormerUNetFormer",
    "MLPMixer",
    "LIIFModel",
    "HybridModel",
    "SwinUNet",
    "create_model",
]
