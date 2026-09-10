#!/usr/bin/env python3
"""
快速可视化脚本 - 生成2D反应扩散数据的可视化图像
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

def main():
    # 数据文件路径
    project_root = Path(__file__).resolve().parents[1]
    data_path = str(project_root / "data/DR2D/2D_diff-react_NA_NA.h5")
    
    print("🔍 正在读取2D反应扩散数据...")
    print(f"📁 数据路径: {data_path}")
    print(f"📂 当前工作目录: {os.getcwd()}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            # 获取第一个样本 (0000)
            sample_key = '0000'
            print(f"📊 读取样本: {sample_key}")
            
            if sample_key in f:
                sample_group = f[sample_key]
                
                # 查找数据
                data_found = False
                for key in sample_group.keys():
                    item = sample_group[key]
                    if hasattr(item, 'shape') and len(item.shape) >= 3:
                        data = item[:]
                        data_found = True
                        print(f"✅ 找到数据: {key}, 形状: {data.shape}")
                        break
                
                if not data_found:
                    print("❌ 未找到合适的数据")
                    return
                
                # 确保数据是4D格式 [time, height, width, channels]
                if len(data.shape) == 3:
                    data = data[..., np.newaxis]
                
                print(f"📐 数据维度: {data.shape}")
                
                # 设置matplotlib参数
                plt.rcParams['font.size'] = 10
                plt.rcParams['figure.dpi'] = 100
                
                # 1. 初始状态
                print("🎨 生成初始状态图像...")
                plt.figure(figsize=(8, 6))
                plt.imshow(data[0, :, :, 0], cmap='viridis', origin='lower')
                plt.title('Initial State (t=0)', fontsize=14, fontweight='bold')
                plt.colorbar(label='Concentration')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                plt.savefig('initial_state.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 保存: initial_state.png")
                
                # 2. 中间状态
                mid_t = min(data.shape[0] // 2, data.shape[0] - 1)
                print(f"🎨 生成中间状态图像 (t={mid_t})...")
                plt.figure(figsize=(8, 6))
                plt.imshow(data[mid_t, :, :, 0], cmap='viridis', origin='lower')
                plt.title(f'Middle State (t={mid_t})', fontsize=14, fontweight='bold')
                plt.colorbar(label='Concentration')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                plt.savefig('middle_state.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 保存: middle_state.png")
                
                # 3. 最终状态
                final_t = data.shape[0] - 1
                print(f"🎨 生成最终状态图像 (t={final_t})...")
                plt.figure(figsize=(8, 6))
                plt.imshow(data[final_t, :, :, 0], cmap='viridis', origin='lower')
                plt.title(f'Final State (t={final_t})', fontsize=14, fontweight='bold')
                plt.colorbar(label='Concentration')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                plt.savefig('final_state.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 保存: final_state.png")
                
                # 4. 时间演化对比
                print("🎨 生成时间演化对比图...")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                times = [0, mid_t, final_t]
                titles = ['Initial', 'Middle', 'Final']
                
                for i, (t, title) in enumerate(zip(times, titles)):
                    im = axes[i].imshow(data[t, :, :, 0], cmap='viridis', origin='lower')
                    axes[i].set_title(f'{title} (t={t})', fontweight='bold')
                    axes[i].set_xlabel('X')
                    axes[i].set_ylabel('Y')
                    plt.colorbar(im, ax=axes[i], label='Concentration')
                
                plt.suptitle('2D Reaction-Diffusion Evolution', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig('evolution_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ 保存: evolution_comparison.png")
                
                # 5. 如果有多个通道，生成通道对比
                if data.shape[-1] > 1:
                    print("🎨 生成多通道对比图...")
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    for ch in range(min(2, data.shape[-1])):
                        im = axes[ch].imshow(data[0, :, :, ch], cmap='viridis', origin='lower')
                        axes[ch].set_title(f'Channel {ch+1} (t=0)', fontweight='bold')
                        axes[ch].set_xlabel('X')
                        axes[ch].set_ylabel('Y')
                        plt.colorbar(im, ax=axes[ch], label='Concentration')
                    
                    plt.suptitle('Multi-Channel Comparison', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig('channels_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("✅ 保存: channels_comparison.png")
                
                print("\n🎉 所有可视化图像生成完成!")
                print("📁 生成的文件:")
                print("   - initial_state.png")
                print("   - middle_state.png") 
                print("   - final_state.png")
                print("   - evolution_comparison.png")
                if data.shape[-1] > 1:
                    print("   - channels_comparison.png")
                
            else:
                print(f"❌ 找不到样本 {sample_key}")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()