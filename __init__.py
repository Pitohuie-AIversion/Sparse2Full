"""
高性能PDEBench训练脚本
基于AMD EPYC 9654 + NVIDIA L40 x2硬件平台深度优化
"""

__version__ = "1.0.0"
__author__ = "PDEBench Team"
__description__ = "High-performance training script for PDEBench with temporal encoding support"

from .src.models import *
from .src.optimizers import *
from .src.data import *
from .src.utils import *
from .src.monitoring import *