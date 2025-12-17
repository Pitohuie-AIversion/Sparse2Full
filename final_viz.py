#!/usr/bin/env python3
"""
最终可视化脚本
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    data_path = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/....data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
    
    print(f"正在读取数据: {data_path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            # 获取第一个样本（0000）
            sample_key = '0000'
            if sample_key in f:
                sample_group = f[sample_key]
                print(f"样本 {sample_key} 的内容:")
                for key in sample_group.keys():
                    item = sample_group[key]
                    if hasattr(item, 'shape'):
                        print(f"  {key}: {item.shape}")
                    else:
                        print(f"  {key}: group")
                
                # 获取数据
                if 'data' in sample_group:
                    data = sample_group['data'][:]
                elif 'tensor' in sample_group:
                    data = sample_group['tensor'][:]
                else:
                    # 取第一个数据集
                    data_keys = [k for k in sample_group.keys() if hasattr(sample_group[k], 'shape')]
                    if data_keys:
                        data = sample_group[data_keys[0]][:]
                    else:
                        raise ValueError("找不到数据")
                
                print(f"数据形状: {data.shape}")
                
                # 确保数据是4D: [time, height, width, channels]
                if len(data.shape) == 3:
                    data = data[..., np.newaxis]  # 添加通道维度
                
                # 生成图像
                plt.figure(figsize=(8, 6))
                plt.imshow(data[0, :, :, 0], cmap='viridis')
                plt.title('初始状态 (t=0)')
                plt.colorbar()
                plt.savefig('sample_0_initial.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ 生成: sample_0_initial.png")
                
                # 中间状态
                mid_t = min(50, data.shape[0] - 1)
                plt.figure(figsize=(8, 6))
                plt.imshow(data[mid_t, :, :, 0], cmap='viridis')
                plt.title(f'中间状态 (t={mid_t})')
                plt.colorbar()
                plt.savefig('sample_0_middle.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ 生成: sample_0_middle.png")
                
                # 最终状态
                final_t = data.shape[0] - 1
                plt.figure(figsize=(8, 6))
                plt.imshow(data[final_t, :, :, 0], cmap='viridis')
                plt.title(f'最终状态 (t={final_t})')
                plt.colorbar()
                plt.savefig('sample_0_final.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ 生成: sample_0_final.png")
                
                # 通道对比（如果有多个通道）
                if data.shape[-1] > 1:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    im1 = axes[0].imshow(data[0, :, :, 0], cmap='viridis')
                    axes[0].set_title('通道1 - 初始')
                    plt.colorbar(im1, ax=axes[0])
                    
                    im2 = axes[1].imshow(data[0, :, :, 1], cmap='viridis')
                    axes[1].set_title('通道2 - 初始')
                    plt.colorbar(im2, ax=axes[1])
                    
                    plt.tight_layout()
                    plt.savefig('sample_0_channels_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("✓ 生成: sample_0_channels_comparison.png")
                else:
                    # 单通道时间对比
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    times = [0, mid_t, final_t]
                    labels = ['初始', '中间', '最终']
                    
                    for i, (t, label) in enumerate(zip(times, labels)):
                        im = axes[i].imshow(data[t, :, :, 0], cmap='viridis')
                        axes[i].set_title(f'{label} (t={t})')
                        plt.colorbar(im, ax=axes[i])
                    
                    plt.tight_layout()
                    plt.savefig('sample_0_channels_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("✓ 生成: sample_0_channels_comparison.png")
                
                print("\n🎉 所有图像生成完成!")
                
            else:
                print(f"❌ 找不到样本 {sample_key}")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()