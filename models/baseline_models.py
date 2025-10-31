"""基线模型兼容性包装模块."""
from .unet import UNet
from .fno2d import FNO2d

# 向后兼容的别名
FNO = FNO2d

__all__ = ["UNet", "FNO", "FNO2d"]
