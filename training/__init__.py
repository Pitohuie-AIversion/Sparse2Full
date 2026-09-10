"""Sparse2Full 训练系统子模块 (Training Subsystem)

提供高内聚、低耦合的训练基础组件：
- BatchProcessor: 张量批次构建与物理守恒校验
- CurriculumScheduler: 课程学习阶段调度
- DataOrchestrator: 数据加载与退化一致性验收
- EngineBuilder: 优化器、学习率调度与混合精度初始化
- TrainingArtifactManager: 检查点、可视化样本与论文交付包管理
"""

from .curriculum import CurriculumScheduler
from .batch_processor import BatchProcessor
from .engine_builder import EngineBuilder
from .data_orchestrator import DataOrchestrator
from .artifact_manager import TrainingArtifactManager

__all__ = [
    'CurriculumScheduler',
    'BatchProcessor',
    'EngineBuilder',
    'DataOrchestrator',
    'TrainingArtifactManager',
]
