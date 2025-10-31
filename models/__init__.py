"""模型模块

提供各种深度学习模型的实现，包括基线模型和先进模型。
所有模型遵循统一接口：forward(x[B,C,H,W]) -> y[B,C,H,W]

支持的模型：
- 基线模型：U-Net, FNO2D, U-FNO瓶颈
- Transformer模型：SegFormer/UNetFormer
- MLP模型：MLP-Mixer, LIIF-Head
- 混合模型：SwinUNet, Hybrid, MLP
"""

from typing import Dict

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
from .hybrid import HybridModel

_OPTIONAL_IMPORT_ERRORS: Dict[str, ModuleNotFoundError] = {}


def _register_optional(names, err):
    for name in names:
        _OPTIONAL_IMPORT_ERRORS[name] = err


def _raise_missing_dependency(model_name: str):
    err = _OPTIONAL_IMPORT_ERRORS.get(model_name)
    if err is None:
        raise ImportError(f"Model '{model_name}' is unavailable due to missing optional dependency.")

    missing = getattr(err, "name", None)
    dependency = missing or "required dependency"
    raise ImportError(
        f"Model '{model_name}' requires optional dependency '{dependency}'. "
        f"Please install it (e.g. `pip install {dependency}`)."
    ) from err


try:  # pragma: no cover - 依赖可选
    from .swin_unet import SwinUNet
except ModuleNotFoundError as err:  # pragma: no cover - 测试环境缺失可选依赖
    SwinUNet = None  # type: ignore[assignment]
    _register_optional(["SwinUNet"], err)

try:  # pragma: no cover - 依赖可选
    from .vit import VisionTransformer, ViT
except ModuleNotFoundError as err:  # pragma: no cover
    VisionTransformer = None  # type: ignore[assignment]
    ViT = None  # type: ignore[assignment]
    _register_optional(["VisionTransformer", "ViT"], err)

try:  # pragma: no cover - 依赖可选
    from .swin_t import SwinTransformerTiny, SwinT
except ModuleNotFoundError as err:  # pragma: no cover
    SwinTransformerTiny = None  # type: ignore[assignment]
    SwinT = None  # type: ignore[assignment]
    _register_optional(["SwinTransformerTiny", "SwinT"], err)

try:  # pragma: no cover - 依赖可选
    from .transformer import Transformer
except ModuleNotFoundError as err:  # pragma: no cover
    Transformer = None  # type: ignore[assignment]
    _register_optional(["Transformer"], err)

# AR模型
from .ar.wrapper import ARWrapper

# 时序NAR模型
from .wrappers.swin_temporal import SwinTemporal, SwinTemporalNAR
from .wrappers.ar_nar_wrapper import ARNARWrapper


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
        model_name = config.name
        
        # 处理参数结构
        if 'params' in config:
            # 如果有params字段，使用params中的参数
            model_params = dict(config.params)
            # 如果params中有kwargs，将其合并到主参数中
            if 'kwargs' in model_params:
                kwargs_dict = dict(model_params['kwargs'])
                del model_params['kwargs']
                model_params.update(kwargs_dict)
        else:
            # 过滤掉name字段，其余作为参数
            model_params = {k: v for k, v in config.items() if k != 'name'}
    
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
        if SwinUNet is None:
            _raise_missing_dependency("SwinUNet")
        return SwinUNet(**model_params)
    elif model_name == "Hybrid" or model_name == "hybrid":
        return HybridModel(**model_params)
    elif model_name == "MLP" or model_name == "mlp":
        return MLPModel(**model_params)
    
    # 新增Transformer模型
    elif model_name == "ViT" or model_name == "VisionTransformer":
        if VisionTransformer is None:
            _raise_missing_dependency("VisionTransformer")
        return VisionTransformer(**model_params)
    elif model_name == "SwinT" or model_name == "SwinTransformerTiny":
        if SwinTransformerTiny is None:
            _raise_missing_dependency("SwinTransformerTiny")
        return SwinTransformerTiny(**model_params)
    elif model_name == "Transformer":
        if Transformer is None:
            _raise_missing_dependency("Transformer")
        return Transformer(**model_params)
    
    # AR模型
    elif model_name == "ARWrapper" or model_name == "ar_wrapper":
        return ARWrapper(**model_params)
    
    # 时序NAR模型
    elif model_name == "SwinTemporal" or model_name == "swin_temporal":
        return SwinTemporal(**model_params)
    elif model_name == "SwinTemporalNAR" or model_name == "swin_temporal_nar":
        return SwinTemporalNAR(**model_params)
    elif model_name == "ARNARWrapper" or model_name == "ar_nar_wrapper":
        return ARNARWrapper(**model_params)
    
    else:
        supported_models = [
            "UNet", "UNetPlusPlus", "FNO2D", "UFNOUNet",
            "SegFormer", "UNetFormer", "SegFormerUNetFormer",
            "MLPMixer", "LIIF",
            "SwinUNet" if SwinUNet is not None else None,
            "Hybrid", "MLP",
            "ViT" if VisionTransformer is not None else None,
            "VisionTransformer" if VisionTransformer is not None else None,
            "SwinT" if SwinTransformerTiny is not None else None,
            "SwinTransformerTiny" if SwinTransformerTiny is not None else None,
            "Transformer" if Transformer is not None else None,
            "ARWrapper", "SwinTemporal", "SwinTemporalNAR", "ARNARWrapper"
        ]
        supported_models = [name for name in supported_models if name is not None]
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

    # MLP模型
    "MLPModel",
    "MLPMixer",
    "LIIFModel",

    # 混合模型
    "HybridModel",

    # AR模型
    "ARWrapper",

    # 时序NAR模型
    "SwinTemporal",
    "SwinTemporalNAR",
    "ARNARWrapper",

    # 工厂函数
    "create_model",
    "get_model"
]

if SwinUNet is not None:
    __all__.append("SwinUNet")

if VisionTransformer is not None:
    __all__.extend(["VisionTransformer", "ViT"])

if SwinTransformerTiny is not None:
    __all__.extend(["SwinTransformerTiny", "SwinT"])

if Transformer is not None:
    __all__.append("Transformer")

# 别名函数，保持向后兼容
def get_model(model_name, **kwargs):
    """获取模型实例（向后兼容）"""
    class Config:
        def __init__(self, name, params):
            self.model = type('Model', (), {'name': name, 'params': params})()
    
    config = Config(model_name, kwargs)
    return create_model(config)