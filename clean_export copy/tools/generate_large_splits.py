#!/usr/bin/env python3
"""
生成大规模数据分割文件
将Darcy Flow数据集的10000个样本分割为：
- 训练集：1000个样本
- 验证集：100个样本  
- 测试集：100个样本
"""

import os
import random
import numpy as np
from pathlib import Path

def generate_splits(total_samples=10000, train_size=1000, val_size=100, test_size=100, seed=2025):
    """
    生成数据分割
    
    Args:
        total_samples: 总样本数
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 生成所有样本索引
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    # 分割数据
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:train_size + val_size + test_size]
    
    print(f"训练集: {len(train_indices)} 个样本")
    print(f"验证集: {len(val_indices)} 个样本")
    print(f"测试集: {len(test_indices)} 个样本")
    print(f"总计使用: {len(train_indices) + len(val_indices) + len(test_indices)} / {total_samples} 个样本")
    
    return train_indices, val_indices, test_indices

def write_split_files(train_indices, val_indices, test_indices, output_dir="data/pdebench/splits"):
    """
    写入分割文件
    
    Args:
        train_indices: 训练集索引
        val_indices: 验证集索引
        test_indices: 测试集索引
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入训练集
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w') as f:
        for idx in train_indices:
            f.write(f"{idx}\n")
    print(f"✅ 训练集文件已保存: {train_file}")
    
    # 写入验证集
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, 'w') as f:
        for idx in val_indices:
            f.write(f"{idx}\n")
    print(f"✅ 验证集文件已保存: {val_file}")
    
    # 写入测试集
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, 'w') as f:
        for idx in test_indices:
            f.write(f"{idx}\n")
    print(f"✅ 测试集文件已保存: {test_file}")

def main():
    """主函数"""
    print("🚀 开始生成大规模数据分割...")
    
    # 生成分割
    train_indices, val_indices, test_indices = generate_splits(
        total_samples=10000,
        train_size=1000,
        val_size=100,
        test_size=100,
        seed=2025
    )
    
    # 写入文件
    write_split_files(train_indices, val_indices, test_indices)
    
    print("🎉 数据分割生成完成！")
    
    # 验证文件
    splits_dir = "data/pdebench/splits"
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()
            print(f"📊 {split_name}.txt: {len(lines)} 行")
        else:
            print(f"❌ {split_file} 不存在")

if __name__ == "__main__":
    main()