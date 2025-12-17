"""
空间预测模型模块

提供专门用于空间预测的深度学习模型，所有模型遵循统一接口：
forward(x[B,C,H,W]) -> y[B,C,H,W]

支持的模型：
- CNN模型：U-Net, U-Net++, FNO2D, U-FNO瓶颈
- Transformer模型：SegFormer, UNetFormer, SegFormer-UNetFormer
- MLP模型：MLP-Mixer, LIIF-Head, MLPModel
- 混合模型：SwinUNet, HybridModel
- 基础模型：VisionTransformer, SwinTransformer, Transformer

使用示例：
    from models.spatial import UNet, SwinUNet
    from models.spatial.factory import create_model
    
    model = create_model("UNet", in_ch=3, out_ch=3, features=[32, 64, 128])
"""

# CNN模型
from .unet import UNet
from .unet_plus_plus import UNetPlusPlus
from .fno2d import FNO2d
from .ufno_unet_bottleneck import UFNOUNet

# Transformer模型
from .segformer import SegFormer
from .unetformer import UNetFormer
from .segformer_unetformer import SegFormerUNetFormer

# MLP模型
from .mlp import MLPModel
from .mlp_mixer import MLPMixer
from .liif import LIIFModel

# 混合模型
from .hybrid import HybridModel

# 基础Transformer模型
try:
    from .vit import VisionTransformer, ViT
except ImportError:
    VisionTransformer = None
    ViT = None

try:
    from .swin_t import SwinTransformerTiny, SwinT
except ImportError:
    SwinTransformerTiny = None
    SwinT = None

try:
    from .transformer import Transformer
except ImportError:
    Transformer = None

try:
    from .swin_unet import SwinUNet
except ImportError:
    SwinUNet = None

try:
    from .sparse_attention_encoder import SparseAttentionEncoder, SparseSwinUNet
except ImportError:
    SparseAttentionEncoder = None
    SparseSwinUNet = None

__all__ = [
    # CNN模型
    "UNet",
    "UNetPlusPlus", 
    "FNO2d",
    "UFNOUNet",
    
    # Transformer模型
    "SegFormer",
    "UNetFormer",
    "SegFormerUNetFormer",
    
    # MLP模型
    "MLPModel",
    "MLPMixer",
    "LIIFModel",
    
    # 混合模型
    "HybridModel",
]

# 添加可选模型
if VisionTransformer is not None:
    __all__.extend(["VisionTransformer", "ViT"])
if SwinTransformerTiny is not None:
    __all__.extend(["SwinTransformerTiny", "SwinT"])
if Transformer is not None:
    __all__.append("Transformer")
if SwinUNet is not None:
    __all__.append("SwinUNet")
if SparseAttentionEncoder is not None:
        __all__.extend(["SparseAttentionEncoder", "SparseSwinUNet"])

# 导入工厂函数
from .factory import create_model