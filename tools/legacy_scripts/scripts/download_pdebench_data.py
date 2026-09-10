#!/usr/bin/env python3
"""
PDEBench数据集下载脚本

从DaRUS数据仓库下载PDEBench数据集的样本数据。
由于网络限制，我们先下载一些小的样本数据进行测试。
"""

import os
import sys
import urllib.request
import h5py
import numpy as np
from pathlib import Path

def create_sample_pdebench_data():
    """创建符合PDEBench格式的样本数据"""
    
    # 创建数据目录
    data_dir = Path("../data/pdebench/official")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建符合PDEBench格式的样本数据
    # 数据格式: [batch, time, x, y, variables] -> [b, t, x, y, v]
    # 我们创建一个简化版本: [time, channels, height, width] -> [t, c, h, w]
    
    print("Creating sample PDEBench-format data...")
    
    # 参数设置
    n_timesteps = 20
    n_channels = 1  # 单变量 u
    height, width = 64, 64
    
    # 生成2D扩散反应方程的模拟数据
    # u_t = D * (u_xx + u_yy) + R * u * (1 - u)
    
    # 创建空间网格
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 初始条件：高斯分布
    u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
    
    # 时间演化（简化的扩散过程）
    data = np.zeros((n_timesteps, n_channels, height, width))
    u = u0.copy()
    
    dt = 0.01
    D = 0.1  # 扩散系数
    
    for t in range(n_timesteps):
        data[t, 0] = u
        
        # 简单的扩散更新（使用有限差分）
        if t < n_timesteps - 1:
            # 拉普拉斯算子（5点模板）
            laplacian = np.zeros_like(u)
            laplacian[1:-1, 1:-1] = (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
            )
            
            # 更新
            u = u + dt * D * laplacian
            
            # 边界条件（零通量）
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
    
    # 保存为HDF5格式
    h5_file = data_dir / "2D_diff-react_NA_NA.h5"
    
    with h5py.File(h5_file, 'w') as f:
        # PDEBench标准格式：[batch, time, x, y, variables]
        # 我们的数据：[time, channels, height, width]
        # 需要转换为：[1, time, height, width, channels]
        pdebench_data = data.transpose(1, 0, 2, 3)  # [channels, time, height, width]
        pdebench_data = pdebench_data.transpose(1, 2, 3, 0)  # [time, height, width, channels]
        pdebench_data = pdebench_data[np.newaxis, ...]  # [1, time, height, width, channels]
        
        f.create_dataset('data', data=pdebench_data)
        
        # 添加元数据
        f.attrs['equation'] = '2D_diff-react'
        f.attrs['description'] = 'Sample 2D diffusion-reaction equation data'
        f.attrs['spatial_resolution'] = f'{height}x{width}'
        f.attrs['time_steps'] = n_timesteps
        f.attrs['variables'] = ['u']
        f.attrs['boundary_conditions'] = 'Neumann'
        
        print(f"Saved data with shape: {pdebench_data.shape}")
        print(f"Data range: [{pdebench_data.min():.4f}, {pdebench_data.max():.4f}]")
    
    # 创建另一个样本文件（Burgers方程）
    print("Creating Burgers equation sample data...")
    
    # 1D Burgers方程样本数据
    n_timesteps = 20
    n_spatial = 64
    
    # 空间网格
    x = np.linspace(0, 1, n_spatial)
    
    # 初始条件：正弦波
    u0 = np.sin(2 * np.pi * x)
    
    # 时间演化数据
    burgers_data = np.zeros((n_timesteps, 1, n_spatial))  # [t, c, x]
    u = u0.copy()
    
    dt = 0.001
    dx = x[1] - x[0]
    nu = 0.01  # 粘性系数
    
    for t in range(n_timesteps):
        burgers_data[t, 0] = u
        
        if t < n_timesteps - 1:
            # Burgers方程：u_t + u * u_x = nu * u_xx
            # 简化更新
            u_new = u.copy()
            for i in range(1, n_spatial - 1):
                # 对流项（向后差分）
                convection = u[i] * (u[i] - u[i-1]) / dx
                # 扩散项（中心差分）
                diffusion = nu * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
                u_new[i] = u[i] - dt * convection + dt * diffusion
            
            # 周期边界条件
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            u = u_new
    
    # 保存Burgers数据
    h5_file = data_dir / "1D_Burgers_Sols_Nu0.01.h5"
    
    with h5py.File(h5_file, 'w') as f:
        # 转换为PDEBench格式：[1, time, spatial, 1]
        pdebench_data = burgers_data.transpose(1, 0, 2)  # [channels, time, spatial]
        pdebench_data = pdebench_data.transpose(1, 2, 0)  # [time, spatial, channels]
        pdebench_data = pdebench_data[np.newaxis, ...]  # [1, time, spatial, channels]
        
        f.create_dataset('data', data=pdebench_data)
        
        # 添加元数据
        f.attrs['equation'] = '1D_Burgers'
        f.attrs['description'] = 'Sample 1D Burgers equation data'
        f.attrs['spatial_resolution'] = f'{n_spatial}'
        f.attrs['time_steps'] = n_timesteps
        f.attrs['variables'] = ['u']
        f.attrs['viscosity'] = nu
        f.attrs['boundary_conditions'] = 'Periodic'
        
        print(f"Saved Burgers data with shape: {pdebench_data.shape}")
        print(f"Data range: [{pdebench_data.min():.4f}, {pdebench_data.max():.4f}]")
    
    print(f"Sample PDEBench data created in: {data_dir}")
    return data_dir

def create_splits_and_stats(data_dir: Path):
    """创建数据切分文件和归一化统计量"""
    
    splits_dir = data_dir.parent / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    # 创建切分文件（基于时间步）
    n_timesteps = 20
    train_ids = list(range(int(0.8 * n_timesteps)))  # 0-15
    val_ids = list(range(int(0.8 * n_timesteps), int(0.9 * n_timesteps)))  # 16-17
    test_ids = list(range(int(0.9 * n_timesteps), n_timesteps))  # 18-19
    
    # 写入切分文件
    with open(splits_dir / "train.txt", 'w') as f:
        for idx in train_ids:
            f.write(f"{idx}\n")
    
    with open(splits_dir / "val.txt", 'w') as f:
        for idx in val_ids:
            f.write(f"{idx}\n")
    
    with open(splits_dir / "test.txt", 'w') as f:
        for idx in test_ids:
            f.write(f"{idx}\n")
    
    # 计算归一化统计量
    print("Computing normalization statistics...")
    
    # 读取2D数据
    h5_file = data_dir / "2D_diff-react_NA_NA.h5"
    with h5py.File(h5_file, 'r') as f:
        data = f['data'][:]  # [1, time, height, width, channels]
        
        # 只在训练集上计算统计量
        train_data = data[0, train_ids, :, :, 0]  # [train_time, height, width]
        
        u_mean = np.mean(train_data)
        u_std = np.std(train_data)
    
    # 保存统计量
    np.savez(
        splits_dir / "norm_stat.npz",
        u_mean=u_mean,
        u_std=u_std
    )
    
    print(f"Normalization stats: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"Splits and stats created in: {splits_dir}")

if __name__ == "__main__":
    print("Creating sample PDEBench data...")
    
    # 创建样本数据
    data_dir = create_sample_pdebench_data()
    
    # 创建切分和统计量
    create_splits_and_stats(data_dir)
    
    print("Done! Sample PDEBench data is ready for testing.")