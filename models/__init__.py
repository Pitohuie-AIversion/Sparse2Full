"""
模型模块 - 分类组织的空间和时间预测模型

提供各种深度学习模型的实现，按功能分类：
- 空间预测模型：专门处理单帧图像/空间数据
- 时间预测模型：专门处理时间序列数据
- 所有模型遵循统一接口标准

使用示例：
    from models.spatial import UNet, SwinUNet
    from models.temporal import ARWrapper, SwinTemporal
"""

from typing import Dict

# 空间预测模型
from . import spatial

# 时间预测模型  
from . import temporal

# 基础模型和工具
from .base import BaseModel
# 这些模型已经移动到spatial文件夹，从那里导入
try:
    from .spatial.mlp import MLPModel
except ImportError:
    MLPModel = None
try:
    from .spatial.hybrid import HybridModel
except ImportError:
    HybridModel = None
# baseline_models不再使用，跳过导入

# 向后兼容的导入
from .spatial import (
    UNet, UNetPlusPlus, FNO2d, UFNOUNet,
    SegFormer, UNetFormer, SegFormerUNetFormer,
    MLPMixer, LIIFModel,
    SwinUNet, VisionTransformer, SwinTransformerTiny, Transformer,
    SparseAttentionEncoder, SparseSwinUNet
)

from .temporal import (
    ARWrapper, SwinTemporal, SwinTemporalNAR, ARNARWrapper,
    TemporalEncoder, TemporalBlock, NARPredictionHead,
    SequentialSpatiotemporal, SequentialTrainer, SequentialDCConsistency
)

__all__ = [
    # 模块分类
    "spatial",
    "temporal",
    
    # 基础模型
    "BaseModel",
    "MLPModel", 
    "HybridModel",
    
    # 空间预测模型
    "UNet", "UNetPlusPlus", "FNO2d", "UFNOUNet",
    "SegFormer", "UNetFormer", "SegFormerUNetFormer",
    "MLPMixer", "LIIFModel",
    "SwinUNet", "VisionTransformer", "SwinTransformerTiny", "Transformer",
    "SparseAttentionEncoder", "SparseSwinUNet",
    
    # 时间预测模型
    "ARWrapper", "SwinTemporal", "SwinTemporalNAR", "ARNARWrapper",
    "TemporalEncoder", "TemporalBlock", "NARPredictionHead",
    "SequentialSpatiotemporal", "SequentialTrainer", "SequentialDCConsistency"
]

# 工厂函数 - 保持向后兼容
def create_model(model_name_or_config, **kwargs):
    """
    根据配置创建模型实例（向后兼容）
    
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
    
    # 空间预测模型
    spatial_models = {
        "UNet", "unet", "UNetPlusPlus", "unet_plus_plus", 
        "FNO2D", "fno2d", "UFNOUNet", "ufno_unet",
        "SegFormer", "segformer", "UNetFormer", "unetformer", 
        "SegFormerUNetFormer", "segformer_unetformer",
        "MLPMixer", "mlp_mixer", "LIIF", "liif",
        "SwinUNet", "swin_unet", "Hybrid", "hybrid", "MLP", "mlp",
        "ViT", "VisionTransformer", "SwinT", "SwinTransformerTiny", "Transformer",
        "SparseSwinUNet", "sparse_swin_unet", "SparseAttentionEncoder", "sparse_attention_encoder"
    }
    
    # 时间预测模型
    temporal_models = {
        "ARWrapper", "ar_wrapper", "SwinTemporal", "swin_temporal",
        "SwinTemporalNAR", "swin_temporal_nar", "ARNARWrapper", "ar_nar_wrapper"
    }
    
    if model_name in spatial_models:
        return spatial.create_model(model_name, **model_params)
    elif model_name in temporal_models:
        return temporal.create_model(model_name, **model_params)
    else:
        # 尝试在空间模型中查找
        try:
            return spatial.create_model(model_name, **model_params)
        except ValueError:
            # 尝试在时间模型中查找
            try:
                return temporal.create_model(model_name, **model_params)
            except ValueError:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(spatial_models | temporal_models)}")


# 别名函数，保持向后兼容
def get_model(model_name, **kwargs):
    """获取模型实例（向后兼容）"""
    return create_model(model_name, **kwargs)