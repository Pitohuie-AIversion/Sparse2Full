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
# 时间预测模型
from . import spatial, temporal

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
    FNO2d,
    LIIFModel,
    MLPMixer,
    SegFormer,
    SegFormerUNetFormer,
    SparseAttentionEncoder,
    SparseSwinUNet,
    SwinTransformerTiny,
    SwinUNet,
    Transformer,
    UFNOUNet,
    UNet,
    UNetFormer,
    UNetPlusPlus,
    VisionTransformer,
)
from .temporal import (
    ARWrapper,
    NARPredictionHead,
    SequentialDCConsistency,
    SequentialSpatiotemporal,
    SequentialTrainer,
    SwinTemporal,
    SwinTemporalNAR,
    TemporalBlock,
    TemporalEncoder,
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
    "UNet",
    "UNetPlusPlus",
    "FNO2d",
    "UFNOUNet",
    "SegFormer",
    "UNetFormer",
    "SegFormerUNetFormer",
    "MLPMixer",
    "LIIFModel",
    "SwinUNet",
    "VisionTransformer",
    "SwinTransformerTiny",
    "Transformer",
    "SparseAttentionEncoder",
    "SparseSwinUNet",
    # 时间预测模型
    "ARWrapper",
    "SwinTemporal",
    "SwinTemporalNAR",
    "TemporalEncoder",
    "TemporalBlock",
    "NARPredictionHead",
    "SequentialSpatiotemporal",
    "SequentialTrainer",
    "SequentialDCConsistency",
]


# 工厂函数 - 保持向后兼容
def create_model(model_name_or_config, **kwargs):
    from .base import create_model as base_create_model

    return base_create_model(model_name_or_config, **kwargs)


# 别名函数，保持向后兼容
def get_model(model_name, **kwargs):
    """获取模型实例（向后兼容）"""
    return create_model(model_name, **kwargs)
