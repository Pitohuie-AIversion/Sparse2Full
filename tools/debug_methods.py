#!/usr/bin/env python3

import torch
import numpy as np
from tools.visualization.pde_bench_visualizer import PDEBenchVisualizer

# 创建测试数据
input_seq = torch.randn(1, 1, 1, 64, 64)
target_seq = torch.randn(1, 5, 1, 64, 64)
pred_seq = torch.randn(1, 5, 1, 64, 64)

# 创建可视化器
visualizer = PDEBenchVisualizer("debug_output")

print("测试 create_error_analysis...")
try:
    result = visualizer.create_error_analysis(input_seq, target_seq, pred_seq)
    print(f"create_error_analysis 成功: {result}")
except Exception as e:
    print(f"create_error_analysis 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试 create_temporal_analysis...")
try:
    result = visualizer.create_temporal_analysis(pred_seq, target_seq)
    print(f"create_temporal_analysis 成功: {result}")
except Exception as e:
    print(f"create_temporal_analysis 失败: {e}")
    import traceback
    traceback.print_exc()