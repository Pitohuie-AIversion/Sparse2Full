#!/usr/bin/env python3
"""
测试可视化模块导入
"""

try:
    import utils.visualization
    print("SUCCESS: utils.visualization imported")
    
    from utils.visualization import PDEBenchVisualizer
    print("SUCCESS: PDEBenchVisualizer imported")
    
    from utils.visualization import create_quadruplet_visualization
    print("SUCCESS: create_quadruplet_visualization imported")
    
    from utils.visualization import create_combined_quadruplet_visualization
    print("SUCCESS: create_combined_quadruplet_visualization imported")
    
    print("✅ 所有可视化模块导入成功")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()