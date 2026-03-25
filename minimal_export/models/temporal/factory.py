"""
时间预测模型工厂函数
"""

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
    # AR包装器
    if lower in {"arwrapper", "ar_wrapper"}:
        from models.ar.wrapper import ARWrapper

        return ARWrapper(**kwargs)

    # 时序Swin模型
    elif lower in {
        "swintemporal",
        "swin_temporal",
        "swintemporalnar",
        "swin_temporal_nar",
    }:
        # 懒加载：仅在选择到对应模型时导入，避免未使用模块的顶层导入告警
        from models.temporal.wrappers.swin_temporal import SwinTemporal, SwinTemporalNAR

        return (
            SwinTemporal(**kwargs)
            if lower in {"swintemporal", "swin_temporal"}
            else SwinTemporalNAR(**kwargs)
        )

    # 混合包装器
    elif lower in {"arnarwrapper", "ar_nar_wrapper"}:
        from models.temporal.wrappers.ar_nar_wrapper import ARNARWrapper

        return ARNARWrapper(**kwargs)

    # 物理感知Transformer模型
    elif lower in {"physicstransformer", "physics_transformer"}:
        from models.temporal.models.physics_transformer import (
            PhysicsTransformerTemporal,
        )

        return PhysicsTransformerTemporal(**kwargs)

    # 时序组件（通常不直接作为独立模型使用）
    elif lower == "temporalencoder":
        from models.temporal.components.temporal_encoder import TemporalEncoder

        return TemporalEncoder(**kwargs)
    elif lower == "temporalblock":
        from models.temporal.components.temporal_block import TemporalBlock

        return TemporalBlock(**kwargs)
    elif lower == "narpredictionhead":
        from models.temporal.components.nar_prediction_head import NARPredictionHead

        return NARPredictionHead(**kwargs)
    elif lower == "sequentialspatiotemporal":
        from models.temporal.components.sequential_spatiotemporal import (
            SequentialSpatiotemporalModel as SequentialSpatiotemporal,
        )

        return SequentialSpatiotemporal(**kwargs)
    elif lower == "sequentialtrainer":
        from models.temporal.components.sequential_trainer import SequentialTrainer

        return SequentialTrainer(**kwargs)
    elif lower == "sequentialdcconsistency":
        from models.temporal.components.sequential_dc_consistency import (
            SequentialDCConsistency,
        )

        return SequentialDCConsistency(**kwargs)

    else:
        supported_models = [
            "ARWrapper",
            "SwinTemporal",
            "SwinTemporalNAR",
            "ARNARWrapper",
            "PhysicsTransformer",
            "TemporalEncoder",
            "TemporalBlock",
            "NARPredictionHead",
            "SequentialSpatiotemporal",
            "SequentialTrainer",
            "SequentialDCConsistency",
        ]
        raise ValueError(
            f"Unknown temporal model: {model_name}. Supported models: {supported_models}"
        )
