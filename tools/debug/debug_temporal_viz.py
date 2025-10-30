#!/usr/bin/env python3
"""
调试时序NAR模型可视化脚本
"""

import sys
import traceback
import os

def main():
    """主测试函数"""
    print("🚀 开始调试时序NAR模型可视化...")
    
    try:
        print("🔍 测试基础导入...")
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✅ Matplotlib导入成功")
        
        import numpy as np
        print("✅ NumPy导入成功")
        
        from pathlib import Path
        print("✅ 基础库导入成功")
        
        # 检查模型文件
        model_path = r"f:\Zhaoyang\Sparse2Full\runs\temporal_nar_100epochs\TemporalNAR-DR2D-128-100epochs-s2025\best.pth"
        print(f"\n🔍 检查模型文件: {model_path}")
        
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在")
            # 搜索可能的模型文件
            runs_dir = Path(r"f:\Zhaoyang\Sparse2Full\runs")
            if runs_dir.exists():
                print("🔍 搜索runs目录中的.pth文件...")
                pth_files = list(runs_dir.rglob("*.pth"))
                if pth_files:
                    print(f"   找到{len(pth_files)}个.pth文件:")
                    for i, pth_file in enumerate(pth_files[:5]):  # 只显示前5个
                        print(f"     {i+1}. {pth_file}")
                    if len(pth_files) > 5:
                        print(f"     ... 还有{len(pth_files)-5}个文件")
                else:
                    print("   未找到任何.pth文件")
            return
        
        print(f"✅ 模型文件存在")
        
        # 尝试加载检查点
        print("🔍 加载检查点...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ 检查点加载成功")
        
        if isinstance(checkpoint, dict):
            print(f"   检查点键: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"   训练轮次: {checkpoint['epoch']}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"   模型参数数量: {len(state_dict)}")
                # 显示前几个键
                keys = list(state_dict.keys())
                print(f"   前5个参数键: {keys[:5]}")
        
        # 导入并测试可视化脚本
        print("\n🔍 导入可视化脚本...")
        sys.path.insert(0, r"f:\Zhaoyang\Sparse2Full")
        
        try:
            from visualize_pth_models import ModelVisualizer
            print("✅ 可视化脚本导入成功")
            
            # 创建可视化器
            output_dir = r"f:\Zhaoyang\Sparse2Full\debug_output"
            visualizer = ModelVisualizer(output_dir=output_dir)
            print("✅ 可视化器创建成功")
            
            # 处理模型
            print("🔍 处理模型...")
            result = visualizer.process_model(model_path)
            
            if result:
                print("✅ 模型处理成功")
                print(f"   指标: {result['metrics']}")
                print(f"   可视化路径: {result['visualization_path']}")
            else:
                print("❌ 模型处理失败")
                
        except Exception as e:
            print(f"❌ 可视化脚本导入或执行失败: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()