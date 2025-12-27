#!/usr/bin/env python3
"""
详细分析test_data.h5的数据内容和统计信息
"""

import h5py
import numpy as np

def detailed_analysis(file_path):
    """详细分析H5文件的数据内容"""
    print(f"详细分析文件: {file_path}")
    print("=" * 80)
    
    with h5py.File(file_path, 'r') as f:
        root_keys = list(f.keys())
        print(f"样本总数: {len(root_keys)}")
        print(f"样本键范围: {root_keys[0]} 到 {root_keys[-1]}")
        print()
        
        # 分析所有样本的数据形状一致性
        print("数据形状一致性检查:")
        shapes = []
        data_ranges = []
        means = []
        
        for sample_key in root_keys:
            if 'data' in f[sample_key]:
                data = f[sample_key]['data']
                shapes.append(data.shape)
                data_ranges.append([np.min(data[:]), np.max(data[:])])
                means.append(np.mean(data[:]))
        
        # 检查形状一致性
        unique_shapes = list(set(shapes))
        print(f"  唯一形状数量: {len(unique_shapes)}")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"    {shape}: {count} 个样本")
        
        print()
        
        # 分析数据范围统计
        print("数据范围统计:")
        all_ranges = np.array(data_ranges)
        global_min = np.min(all_ranges[:, 0])
        global_max = np.max(all_ranges[:, 1])
        mean_min = np.mean(all_ranges[:, 0])
        mean_max = np.mean(all_ranges[:, 1])
        
        print(f"  全局最小值: {global_min:.4f}")
        print(f"  全局最大值: {global_max:.4f}")
        print(f"  平均最小值: {mean_min:.4f}")
        print(f"  平均最大值: {mean_max:.4f}")
        print(f"  全局范围: [{global_min:.4f}, {global_max:.4f}]")
        
        # 分析通道信息
        if len(unique_shapes) > 0:
            sample_shape = unique_shapes[0]
            if len(sample_shape) >= 4:
                print()
                print("通道分析:")
                print(f"  总时间步: {sample_shape[0]}")
                print(f"  空间分辨率: {sample_shape[1]} x {sample_shape[2]}")
                print(f"  通道数: {sample_shape[3]}")
                
                # 分析每个通道的特性
                print("  各通道统计:")
                for sample_key in root_keys[:3]:  # 只分析前3个样本
                    if 'data' in f[sample_key]:
                        data = f[sample_key]['data'][:]
                        print(f"    样本 {sample_key}:")
                        for ch in range(sample_shape[3]):
                            channel_data = data[:, :, :, ch]
                            print(f"      通道 {ch}: 范围=[{np.min(channel_data):.4f}, {np.max(channel_data):.4f}], 均值={np.mean(channel_data):.4f}")
                        break
        
        # 分析坐标信息
        print()
        print("坐标信息分析:")
        coord_samples = 0
        for sample_key in root_keys:
            sample_group = f[sample_key]
            if 'x' in sample_group and 'y' in sample_group:
                coord_samples += 1
                if coord_samples == 1:  # 只分析第一个有坐标的样本
                    x_coords = sample_group['x'][:]
                    y_coords = sample_group['y'][:]
                    print(f"  样本 {sample_key} 坐标:")
                    print(f"    x坐标范围: [{np.min(x_coords):.4f}, {np.max(x_coords):.4f}]")
                    print(f"    y坐标范围: [{np.min(y_coords):.4f}, {np.max(y_coords):.4f}]")
                    print(f"    x坐标步长: {np.diff(x_coords)[:5]}")  # 前5个步长
                    print(f"    y坐标步长: {np.diff(y_coords)[:5]}")  # 前5个步长
        
        if coord_samples == 0:
            print("  未找到坐标信息")
        else:
            print(f"  有坐标信息的样本数: {coord_samples}")
        
        # 数据质量检查
        print()
        print("数据质量检查:")
        nan_samples = 0
        inf_samples = 0
        zero_variance_samples = 0
        
        for sample_key in root_keys:
            if 'data' in f[sample_key]:
                data = f[sample_key]['data'][:]
                if np.any(np.isnan(data)):
                    nan_samples += 1
                if np.any(np.isinf(data)):
                    inf_samples += 1
                if np.var(data) == 0:
                    zero_variance_samples += 1
        
        print(f"  含NaN的样本数: {nan_samples}")
        print(f"  含Inf的样本数: {inf_samples}")
        print(f"  零方差样本数: {zero_variance_samples}")
        
        if nan_samples == 0 and inf_samples == 0 and zero_variance_samples == 0:
            print("  ✓ 数据质量良好")
        else:
            print("  ⚠ 发现数据质量问题")
        
        # 时间序列分析
        if len(unique_shapes) > 0 and len(unique_shapes[0]) >= 4:
            print()
            print("时间序列特性:")
            sample_shape = unique_shapes[0]
            
            # 分析时间相关性
            if sample_shape[0] > 1:
                correlations = []
                for sample_key in root_keys[:5]:  # 分析前5个样本
                    if 'data' in f[sample_key]:
                        data = f[sample_key]['data'][:]
                        # 计算相邻时间步的相关性
                        for ch in range(sample_shape[3]):
                            channel_data = data[:, :, :, ch]
                            # 重塑为 (time, spatial)
                            spatial_flat = channel_data.reshape(sample_shape[0], -1)
                            if sample_shape[0] > 1:
                                corr = np.corrcoef(spatial_flat[:-1], spatial_flat[1:])[0, 1]
                                correlations.append(corr)
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    print(f"  平均时间相关性: {avg_corr:.4f}")
                    if avg_corr > 0.9:
                        print("  ✓ 强时间相关性（适合AR模型）")
                    elif avg_corr > 0.5:
                        print("  ✓ 中等时间相关性")
                    else:
                        print("  ⚠ 弱时间相关性")

if __name__ == "__main__":
    file_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/test_data.h5"
    detailed_analysis(file_path)