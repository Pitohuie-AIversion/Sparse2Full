#!/usr/bin/env python3
"""
测试时序NAR模型可视化脚本
"""

import sys
import traceback

def test_imports():
    """测试导入"""
    try:
        print("🔍 测试导入...")
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✅ Matplotlib导入成功")
        
        import numpy as np
        print("✅ NumPy导入成功")
        
        from pathlib import Path
        print("✅ 基础库导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        print("\n🔍 测试模型加载...")
        import torch
        
        model_path = r"f:\Zhaoyang\Sparse2Full\runs\temporal_nar_100epochs\TemporalNAR-DR2D-128-100epochs-s2025\best.pth"
        
        # 检查文件是否存在
        from pathlib import Path
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"✅ 模型文件存在: {model_path}")
        
        # 尝试加载检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ 检查点加载成功")
        
        if isinstance(checkpoint, dict):
            print(f"   检查点键: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"   训练轮次: {checkpoint['epoch']}")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False

def test_visualization_creation():
    """测试可视化创建"""
    try:
        print("\n🔍 测试可视化创建...")
        
        # 导入可视化脚本
        sys.path.insert(0, r"f:\Zhaoyang\Sparse2Full")
        from visualize_pth_models import ModelVisualizer
        
        # 创建可视化器
        output_dir = r"f:\Zhaoyang\Sparse2Full\test_output"
        visualizer = ModelVisualizer(output_dir=output_dir)
        print("✅ 可视化器创建成功")
        
        # 测试模型路径
        model_path = r"f:\Zhaoyang\Sparse2Full\runs\temporal_nar_100epochs\TemporalNAR-DR2D-128-100epochs-s2025\best.pth"
        
        # 处理模型
        result = visualizer.process_model(model_path)
        
        if result:
            print("✅ 模型处理成功")
            print(f"   指标: {result['metrics']}")
            return True
        else:
            print("❌ 模型处理失败")
            return False
            
    except Exception as e:
        print(f"❌ 可视化创建失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试时序NAR模型可视化...")
    
    # 测试导入
    if not test_imports():
        return
    
    # 测试模型加载
    if not test_model_loading():
        return
    
    # 测试可视化创建
    if not test_visualization_creation():
        return
    
    print("\n🎉 所有测试通过！")

if __name__ == "__main__":
    main()