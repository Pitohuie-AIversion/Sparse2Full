"""解码器模块

包含各种解码器实现，支持NAR多步预测。
"""

from .query_head import TimeQueryHead, CrossAttentionQueryHead

__all__ = [
    'TimeQueryHead',
    'CrossAttentionQueryHead'
]