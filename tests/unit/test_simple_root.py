#!/usr/bin/env python3
"""
简单测试脚本，用于调试数据集问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("开始导入模块...")
    import torch
    print("✅ torch导入成功")
    
    import h5py
    print("✅ h5py导入成功")
    
    from omegaconf import OmegaConf
    print("✅ omegaconf导入成功")
    
    from datasets.pdebench import PDEBenchSR
    print("✅ PDEBenchSR导入成功")
    
    # 测试配置文件读取
    config_path = "configs/data/pdebench.yaml"
    config = OmegaConf.load(config_path)
    print(f"✅ 配置文件读取成功: keys={config.keys}")
    
    # 测试HDF5文件访问
    hdf5_file = "E:/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5"
    with h5py.File(hdf5_file, 'r') as f:
        print(f"✅ HDF5文件访问成功: keys={list(f.keys())}")
        if 'tensor' in f:
            print(f"✅ tensor数据存在: shape={f['tensor'].shape}")
    
    # 测试数据集创建
    print("开始创建数据集...")
    dataset = PDEBenchSR(
        data_path=hdf5_file,
        keys=["tensor"],
        scale=4,
        sigma=1.0,
        image_size=128,
        normalize=True,
        split="train"
    )
    print("✅ 数据集创建成功")
    
    # 测试数据读取
    print("测试数据读取...")
    sample = dataset[0]
    print(f"✅ 数据读取成功: sample keys={list(sample.keys())}")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🎉 所有测试通过!")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()