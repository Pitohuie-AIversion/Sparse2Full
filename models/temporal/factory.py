"""
时间预测模型工厂函数
"""

from typing import Dict, Any
import torch.nn as nn


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    创建时间预测模型实例
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 模型实例
    """
    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is empty")
    lower = name.lower()

    # 优先使用统一模型注册表创建
    try:
        from models.registry import create_model as registry_create_model
        return registry_create_model(name, **kwargs)
    except Exception:
        pass

    # 时序组件兜底创建（通常不作为独立预测模型）
    if lower == "temporalencoder":
        from models.temporal.components.temporal_encoder import TemporalEncoder
        return TemporalEncoder(**kwargs)

    elif lower == "temporalblock":
        from models.temporal.components.temporal_block import TemporalBlock
        return TemporalBlock(**kwargs)
    elif lower == "narpredictionhead":
        from models.temporal.components.nar_prediction_head import NARPredictionHead
        return NARPredictionHead(**kwargs)
    elif lower == "sequentialspatiotemporal":
        from models.temporal.components.sequential_spatiotemporal import SequentialSpatiotemporalModel as SequentialSpatiotemporal
        return SequentialSpatiotemporal(**kwargs)
    elif lower == "sequentialtrainer":
        from models.temporal.components.sequential_trainer import SequentialTrainer
        return SequentialTrainer(**kwargs)
    elif lower == "sequentialdcconsistency":
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
