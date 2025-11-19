#!/usr/bin/env python3
"""
分布式清理路径单测

模拟 torch.distributed 环境，验证 RealDataARTrainer.cleanup_distributed() 会调用 destroy_process_group。
"""

import sys
from pathlib import Path
import types
import pytest


def test_cleanup_distributed_calls_destroy(monkeypatch):
    # 动态导入训练器模块
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "tools" / "training"))
    from tools.training.train_real_data_ar import RealDataARTrainer

    # 创建伪分布式模块接口
    class DummyDist:
        def __init__(self):
            self.destroy_called = False
            self._initialized = True

        def is_available(self):
            return True

        def is_initialized(self):
            return self._initialized

        def barrier(self):
            return None

        def destroy_process_group(self):
            self.destroy_called = True

    dummy = DummyDist()

    # 替换 torch.distributed 引用
    import tools.training.train_real_data_ar as mod
    mod.dist = dummy  # 训练器文件顶部引用

    # 构造训练器实例（无需完整初始化）
    trainer = RealDataARTrainer.__new__(RealDataARTrainer)

    # 调用清理
    trainer.cleanup_distributed()

    assert dummy.destroy_called, "destroy_process_group 未被调用"