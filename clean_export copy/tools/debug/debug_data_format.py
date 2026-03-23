#!/usr/bin/env python3
"""调试数据格式脚本"""

import torch
from omegaconf import OmegaConf
from datasets.pdebench import PDEBenchDataModule

def debug_data_format():
    """调试数据格式"""
    # 加载配置
    config = OmegaConf.load("configs/experiment/temporal_nar_300epochs.yaml")
    
    # 创建数据模块
    data_module = PDEBenchDataModule(config.data)
    data_module.setup('test')
    
    # 获取测试数据加载器
    test_loader = data_module.test_dataloader()
    
    print("=== 数据格式调试 ===")
    print(f"数据路径: {config.data.data_path}")
    print(f"图像大小: {config.data.image_size}")
    print(f"批次大小: {config.data.batch_size}")
    
    # 打印配置信息
    print(f"配置类型: {type(config.data)}")
    print(f"配置属性: {list(config.data.keys()) if hasattr(config.data, 'keys') else 'N/A'}")
    
    # 跳过keys数量统计，直接进行数据检查
    if hasattr(config.data, 'keys') and config.data.keys:
        print(f"数据键: {config.data.keys}")
    else:
        print("数据键: 未配置")
    
    # 检查第一个batch
    for i, batch in enumerate(test_loader):
        print(f"\n=== Batch {i} ===")
        
        if isinstance(batch, dict):
            print("数据格式: 字典")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)} - {value}")
        elif isinstance(batch, (list, tuple)):
            print(f"数据格式: {type(batch).__name__}")
            for j, item in enumerate(batch):
                if torch.is_tensor(item):
                    print(f"  [{j}]: {item.shape} ({item.dtype})")
                else:
                    print(f"  [{j}]: {type(item)} - {item}")
        else:
            print(f"数据格式: {type(batch)}")
            if torch.is_tensor(batch):
                print(f"  形状: {batch.shape} ({batch.dtype})")
        
        # 只检查第一个batch
        break
    
    print("\n=== 数据集信息 ===")
    print(f"训练集大小: {len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') else 'N/A'}")
    print(f"验证集大小: {len(data_module.val_dataset) if hasattr(data_module, 'val_dataset') else 'N/A'}")
    print(f"测试集大小: {len(data_module.test_dataset) if hasattr(data_module, 'test_dataset') else 'N/A'}")

if __name__ == "__main__":
    debug_data_format()