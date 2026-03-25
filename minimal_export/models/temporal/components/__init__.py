"""
时序组件模块
包含时间预测模型的基础组件
"""

from .temporal_encoder import TemporalEncoder

try:
    from .temporal_block import TemporalBlock
except ImportError:
    from .temporal_block import FiLMTemporalBlock as TemporalBlock
try:
    from .nar_prediction_head import NARPredictionHead
except ImportError:
    from .nar_prediction_head import CrossAttnTimeQueryHead as NARPredictionHead
try:
    from .sequential_spatiotemporal import SequentialSpatiotemporal
except ImportError:
    from .sequential_spatiotemporal import (
        SequentialSpatiotemporalModel as SequentialSpatiotemporal,
    )
try:
    from .sequential_trainer import SequentialTrainer
except ImportError:
    from .sequential_trainer import SequentialSpatiotemporalTrainer as SequentialTrainer
from .sequential_dc_consistency import SequentialDCConsistency

__all__ = [
    "TemporalEncoder",
    "TemporalBlock",
    "NARPredictionHead",
    "SequentialSpatiotemporal",
    "SequentialTrainer",
    "SequentialDCConsistency",
]
