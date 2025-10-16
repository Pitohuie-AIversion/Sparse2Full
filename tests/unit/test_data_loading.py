#!/usr/bin/env python3
"""
测试数据加载功能
"""

from datasets.pdebench import PDEBenchDataModule
from omegaconf import DictConfig, OmegaConf

def main():
    # 加载配置
    config = OmegaConf.load('configs/train.yaml')
    print('配置加载成功')

    # 创建数据模块
    data_module = PDEBenchDataModule(config.data)
    print('数据模块创建成功')

    # 设置数据模块
    data_module.setup()
    print('数据模块设置完成')

    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f'训练集批次数: {len(train_loader)}')
    print(f'验证集批次数: {len(val_loader)}')
    print(f'测试集批次数: {len(test_loader)}')

    # 测试一个批次
    batch = next(iter(train_loader))
    print(f'批次类型: {type(batch)}')
    
    if isinstance(batch, dict):
        print(f'批次键: {list(batch.keys())}')
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f'{key}: {value.shape}')
            else:
                print(f'{key}: {type(value)}')
    else:
        print(f'批次长度: {len(batch)}')
        for i, item in enumerate(batch):
            if hasattr(item, 'shape'):
                print(f'项目 {i}: {item.shape}')
            else:
                print(f'项目 {i}: {type(item)}')
    
    print('✅ 数据加载验证成功！')

if __name__ == "__main__":
    main()