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
- 轻量级/SR模型: ConvUNetLite, ResNetLite, CNNAttnLite, ConvGateLite

使用示例：
    from models.spatial import UNet, SwinUNet
    from models.spatial.factory import create_model
    
    model = create_model("UNet", in_ch=3, out_ch=3, features=[32, 64, 128])
"""

# CNN模型
from .edsr import EDSR
from .fno2d import FNO2d

# Transformer模型
from .segformer import SegFormer
from .segformer_unetformer import SegFormerUNetFormer
from .sparse_attention_encoder import SparseSwinUNet
from .ufno_unet_bottleneck import UFNOUNet
from .unet import UNet
from .unet_plus_plus import UNetPlusPlus
from .unetformer import UNetFormer

# MLP模型
try:
    from .mlp import MLP, MLPModel
except ImportError:
    MLPModel = None
    MLP = None
try:
    from .coordinate_encoder import CoordinateEncoder
except ImportError:
    CoordinateEncoder = None
# 混合模型
from .hybrid import HybridModel
from .liif import LIIFModel
from .mlp_mixer import MLPMixer

# Partial Convolution
try:
    from .partialconv_unet import PartialConvUNet
except ImportError:
    PartialConvUNet = None

# 基础Transformer模型
try:
    from .vit import VisionTransformer, ViT
except ImportError:
    VisionTransformer = None
    ViT = None

try:
    from .swin_t import SwinT, SwinTransformerTiny
except ImportError:
    SwinTransformerTiny = None
    SwinT = None

try:
    from .swin_t_with_encoder import SwinTWithEncoder
except ImportError:
    SwinTWithEncoder = None

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

# 轻量级/SR模型
try:
    from .conv_unet_lite import ConvUNetLite
except ImportError:
    ConvUNetLite = None

try:
    from .resnet_lite import ResNetLite
except ImportError:
    ResNetLite = None

try:
    from .cnn_attn_lite import CNNAttnLite
except ImportError:
    CNNAttnLite = None

try:
    from .conv_gate_lite import ConvGateLite
except ImportError:
    ConvGateLite = None

__all__ = [
    # CNN模型
    "UNet",
    "UNetPlusPlus",
    "FNO2d",
    "UFNOUNet",
    "EDSR",
    "SparseSwinUNet",
    # Transformer模型
    "SegFormer",
    "UNetFormer",
    "SegFormerUNetFormer",
    # MLP模型
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
if SwinTWithEncoder is not None:
    __all__.append("SwinTWithEncoder")
if Transformer is not None:
    __all__.append("Transformer")
if SwinUNet is not None:
    __all__.append("SwinUNet")
if SparseAttentionEncoder is not None:
    __all__.extend(["SparseAttentionEncoder", "SparseSwinUNet"])
if ConvUNetLite is not None:
    __all__.append("ConvUNetLite")
if ResNetLite is not None:
    __all__.append("ResNetLite")
if CNNAttnLite is not None:
    __all__.append("CNNAttnLite")
if ConvGateLite is not None:
    __all__.append("ConvGateLite")
if MLPModel is not None:
    __all__.append("MLPModel")
if PartialConvUNet is not None:
    __all__.append("PartialConvUNet")

# 导入工厂函数
from .factory import create_model
