"""
时间预测模型工厂函数
"""

from typing import Dict, Any
from ..base import BaseModel


def create_model(model_name: str, **kwargs) -> BaseModel:
    """
    创建时间预测模型实例
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        BaseModel: 模型实例
    """
    # AR包装器
    if model_name == "ARWrapper" or model_name == "ar_wrapper":
        from models.ar.wrapper import ARWrapper
        return ARWrapper(**kwargs)
    
    # 时序Swin模型
    elif model_name in ("SwinTemporal", "swin_temporal", "SwinTemporalNAR", "swin_temporal_nar"):
        # 懒加载：仅在选择到对应模型时导入，避免未使用模块的顶层导入告警
        from models.temporal.wrappers.swin_temporal import SwinTemporal, SwinTemporalNAR
        return SwinTemporal(**kwargs) if model_name.lower() == "swin_temporal" else SwinTemporalNAR(**kwargs)
    
    # 混合包装器
    elif model_name == "ARNARWrapper" or model_name == "ar_nar_wrapper":
        from models.temporal.wrappers.ar_nar_wrapper import ARNARWrapper
        return ARNARWrapper(**kwargs)
    
    # 物理感知Transformer模型
    elif model_name == "PhysicsTransformer" or model_name == "physics_transformer":
        from models.temporal.models.physics_transformer import PhysicsTransformerTemporal
        return PhysicsTransformerTemporal(**kwargs)
    
    # 时序组件（通常不直接作为独立模型使用）
    elif model_name == "TemporalEncoder":
        from models.temporal.components.temporal_encoder import TemporalEncoder
        return TemporalEncoder(**kwargs)
    elif model_name == "TemporalBlock":
        from models.temporal.components.temporal_block import TemporalBlock
        return TemporalBlock(**kwargs)
    elif model_name == "NARPredictionHead":
        from models.temporal.components.nar_prediction_head import NARPredictionHead
        return NARPredictionHead(**kwargs)
    elif model_name == "SequentialSpatiotemporal":
        from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporal
        return SequentialSpatiotemporal(**kwargs)
    elif model_name == "SequentialTrainer":
        from models.temporal.components.sequential_trainer import SequentialTrainer
        return SequentialTrainer(**kwargs)
    elif model_name == "SequentialDCConsistency":
        from models.temporal.components.sequential_dc_consistency import SequentialDCConsistency
        return SequentialDCConsistency(**kwargs)
    
    else:
        supported_models = [
            "ARWrapper",
            "SwinTemporal", "SwinTemporalNAR", 
            "ARNARWrapper",
            "PhysicsTransformer",
            "TemporalEncoder", "TemporalBlock", "NARPredictionHead",
            "SequentialSpatiotemporal", "SequentialTrainer", "SequentialDCConsistency"
        ]
        raise ValueError(f"Unknown temporal model: {model_name}. Supported models: {supported_models}")
