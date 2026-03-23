#!/usr/bin/env python3
"""
创建测试用的PDEBench数据集
生成简单的2D扩散反应方程数据用于训练测试
"""

import os
import h5py
import numpy as np
from pathlib import Path

def create_2d_diffusion_reaction_data():
    """创建2D扩散反应方程测试数据"""
    
    # 数据参数
    nx, ny = 128, 128  # 空间分辨率
    nt = 50           # 时间步数
    n_samples = 100   # 样本数量
    
    # 物理参数
    dx = 1.0 / nx
    dy = 1.0 / ny
    dt = 0.01
    D = 0.1  # 扩散系数
    k = 0.1  # 反应系数
    
    print(f"创建测试数据: {n_samples}个样本, 空间分辨率{nx}x{ny}, 时间步数{nt}")
    
    # 创建数据目录
    data_dir = Path("data/pdebench/2D/diff-react")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建HDF5文件
    h5_path = data_dir / "2D_diff-react_NA_NA.h5"
    
    with h5py.File(h5_path, 'w') as f:
        # 创建数据集
        u_data = np.zeros((n_samples, nt, nx, ny), dtype=np.float32)
        
        for i in range(n_samples):
            print(f"生成样本 {i+1}/{n_samples}")
            
            # 初始条件：随机高斯分布
            u = np.random.normal(0.5, 0.1, (nx, ny))
            u = np.clip(u, 0, 1)  # 限制在[0,1]范围
            
            # 时间演化
            for t in range(nt):
                u_data[i, t] = u.copy()
                
                if t < nt - 1:
                    # 简单的扩散反应方程数值求解
                    # du/dt = D * ∇²u + k * u * (1 - u)
                    
                    # 计算拉普拉斯算子（5点差分）
                    laplacian = np.zeros_like(u)
                    laplacian[1:-1, 1:-1] = (
                        u[2:, 1:-1] + u[:-2, 1:-1] + 
                        u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
                    ) / (dx * dy)
                    
                    # 边界条件：零通量
                    laplacian[0, :] = laplacian[1, :]
                    laplacian[-1, :] = laplacian[-2, :]
                    laplacian[:, 0] = laplacian[:, 1]
                    laplacian[:, -1] = laplacian[:, -2]
                    
                    # 反应项
                    reaction = k * u * (1 - u)
                    
                    # 时间步进
                    u = u + dt * (D * laplacian + reaction)
                    u = np.clip(u, 0, 1)  # 保持物理约束
        
        # 保存数据
        f.create_dataset('u', data=u_data, compression='gzip', compression_opts=9)
        
        # 添加元数据
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nt'] = nt
        f.attrs['n_samples'] = n_samples
        f.attrs['dx'] = dx
        f.attrs['dy'] = dy
        f.attrs['dt'] = dt
        f.attrs['D'] = D
        f.attrs['k'] = k
        f.attrs['description'] = '2D Diffusion-Reaction equation test data'
    
    print(f"数据已保存到: {h5_path}")
    print(f"数据形状: {u_data.shape}")
    print(f"数据范围: [{u_data.min():.3f}, {u_data.max():.3f}]")
    
    return h5_path

def create_data_splits():
    """创建数据切分文件"""
    splits_dir = Path("data/pdebench/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # 100个样本的切分：70训练，15验证，15测试
    n_samples = 100
    indices = list(range(n_samples))
    
    train_indices = indices[:70]
    val_indices = indices[70:85]
    test_indices = indices[85:]
    
    # 保存切分文件
    with open(splits_dir / "train.txt", 'w') as f:
        for idx in train_indices:
            f.write(f"{idx}\n")
    
    with open(splits_dir / "val.txt", 'w') as f:
        for idx in val_indices:
            f.write(f"{idx}\n")
    
    with open(splits_dir / "test.txt", 'w') as f:
        for idx in test_indices:
            f.write(f"{idx}\n")
    
    print(f"数据切分已保存到: {splits_dir}")
    print(f"训练集: {len(train_indices)}个样本")
    print(f"验证集: {len(val_indices)}个样本")
    print(f"测试集: {len(test_indices)}个样本")

if __name__ == "__main__":
    print("开始创建PDEBench测试数据...")
    
    # 创建数据
    h5_path = create_2d_diffusion_reaction_data()
    
    # 创建切分
    create_data_splits()
    
    print("\n测试数据创建完成！")
    print("现在可以开始训练了。")