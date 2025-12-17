#!/usr/bin/env python3
"""
模块别名：为测试与脚本提供根级导入入口
将 `from train_real_data_ar import RealDataARTrainer` 映射到
`tools.training.train_real_data_ar` 内的实现。
"""

from tools.training.train_real_data_ar import *  # noqa: F401,F403
from tools.training.train_real_data_ar import main

if __name__ == "__main__":
    main()