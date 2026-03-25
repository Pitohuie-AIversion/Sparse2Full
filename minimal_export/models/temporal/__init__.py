"""
时间预测模型模块

提供专门用于时间序列预测的深度学习模型，支持以下接口：
- forward(x[B,T,C,H,W]) -> y[B,T,C,H,W] (时间序列预测)
- forward(x[B,T,C,H,W]) -> y[B,C,H,W] (时间聚合后单帧输出)

支持的模型：
- AR包装器：ARWrapper (将空间模型包装为时间预测)
- 时序Swin：SwinTemporal, SwinTemporalNAR
- 混合包装器：ARNARWrapper (AR+NAR组合)
- 时序组件：时序编码器、预测头、时序块

使用示例：
    from models.temporal import ARWrapper, SwinTemporal
    from models.temporal.factory import create_model
    
    model = create_model("ARWrapper", backbone="SwinUNet", T_out=10)
"""

# 基础时序模型类（新增）
from .base_temporal import (
    ARMixin,
    BaseTemporalModel,
    NARMixin,
    TemporalConsistencyMixin,
)

# AR包装器
try:
    from models.ar.wrapper import ARWrapper
except ImportError as e:
    logger = __import__("logging").getLogger(__name__)
    logger.warning(f"Failed to import ARWrapper: {e}")
    ARWrapper = None

# 时序Swin模型
SwinTemporal = None
SwinTemporalNAR = None

# 混合包装器
# from .components.ar_nar_wrapper import ARNARWrapper

# 时序组件
try:
    from models.temporal.components.temporal_encoder import TemporalEncoder
except ImportError as e:
    logger = __import__("logging").getLogger(__name__)
    logger.warning(f"Failed to import TemporalEncoder: {e}")
    TemporalEncoder = None

try:
    from models.temporal.components.temporal_block import TemporalBlock
except ImportError:
    try:
        from models.temporal.components.temporal_block import (
            FiLMTemporalBlock as TemporalBlock,
        )
    except ImportError as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(f"Failed to import TemporalBlock: {e}")
        TemporalBlock = None

try:
    from models.temporal.components.nar_prediction_head import NARPredictionHead
except ImportError:
    try:
        from models.temporal.components.nar_prediction_head import (
            CrossAttnTimeQueryHead as NARPredictionHead,
        )
    except ImportError as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(f"Failed to import NARPredictionHead: {e}")
        NARPredictionHead = None

try:
    from models.temporal.components.sequential_spatiotemporal import (
        SequentialSpatiotemporal,
    )
except ImportError:
    try:
        from models.temporal.components.sequential_spatiotemporal import (
            SequentialSpatiotemporalModel as SequentialSpatiotemporal,
        )
    except ImportError as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(f"Failed to import SequentialSpatiotemporal: {e}")
        SequentialSpatiotemporal = None

try:
    from models.temporal.components.sequential_trainer import SequentialTrainer
except ImportError:
    try:
        from models.temporal.components.sequential_trainer import (
            SequentialSpatiotemporalTrainer as SequentialTrainer,
        )
    except ImportError as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(f"Failed to import SequentialTrainer: {e}")
        SequentialTrainer = None

try:
    from models.temporal.components.sequential_dc_consistency import (
        SequentialDCConsistency,
    )
except ImportError as e:
    logger = __import__("logging").getLogger(__name__)
    logger.warning(f"Failed to import SequentialDCConsistency: {e}")
    SequentialDCConsistency = None

__all__ = [
    # 基础类
    "BaseTemporalModel",
    "ARMixin",
    "NARMixin",
    "TemporalConsistencyMixin",
    # AR包装器
    "ARWrapper",
    # 时序Swin模型
    "SwinTemporal",
    "SwinTemporalNAR",
    # 混合包装器
    # "ARNARWrapper", # 暂时禁用
    # 时序组件
    "TemporalEncoder",
    "TemporalBlock",
    "NARPredictionHead",
    "SequentialSpatiotemporal",
    "SequentialTrainer",
    "SequentialDCConsistency",
]

# 导入工厂函数
from models.temporal.factory import create_model

__all__.append("create_model")
