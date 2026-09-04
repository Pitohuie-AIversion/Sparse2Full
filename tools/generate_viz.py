#!/usr/bin/env python3
"""
简单直接的2D反应扩散数据可视化脚本
直接生成PNG图像文件到当前目录
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("🔍 开始生成2D反应扩散数据可视化...")
    
    # 数据文件路径
    project_root = Path(__file__).resolve().parents[1]
    data_path = str(project_root / "data/DR2D/2D_diff-react_NA_NA.h5")
    
    print(f"📁 数据路径: {data_path}")
    print(f"📂 当前工作目录: {os.getcwd()}")
    
    try:
        # 读取HDF5文件
        with h5py.File(data_path, 'r') as f:
            print("✅ 成功打开HDF5文件")
            
            # 获取样本0000
            sample_key = '0000'
            if sample_key not in f:
                print(f"❌ 找不到样本 {sample_key}")
                return
            
            sample_group = f[sample_key]
            print(f"📊 样本组键: {list(sample_group.keys())}")
            
            # 查找数据
            data = None
            for key in sample_group.keys():
                item = sample_group[key]
                if hasattr(item, 'shape') and len(item.shape) >= 3:
                    data = item[:]
                    print(f"✅ 找到数据: {key}, 形状: {data.shape}")
                    break
            
            if data is None:
                print("❌ 未找到合适的数据")
                return
            
            # 确保数据是4D格式
            if len(data.shape) == 3:
                data = data[..., np.newaxis]
            
            print(f"📐 最终数据维度: {data.shape}")
            
            # 设置matplotlib基本参数
            plt.style.use('default')
            
            # 1. 生成初始状态图像
            print("🎨 生成初始状态图像...")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data[0, :, :, 0], cmap='viridis')
            ax.set_title('Initial State (t=0)', fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Concentration')
            plt.tight_layout()
            plt.savefig('viz_initial.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 保存: viz_initial.png")
            
            # 2. 生成中间状态图像
            mid_t = data.shape[0] // 2
            print(f"🎨 生成中间状态图像 (t={mid_t})...")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data[mid_t, :, :, 0], cmap='viridis')
            ax.set_title(f'Middle State (t={mid_t})', fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Concentration')
            plt.tight_layout()
            plt.savefig('viz_middle.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 保存: viz_middle.png")
            
            # 3. 生成最终状态图像
            final_t = data.shape[0] - 1
            print(f"🎨 生成最终状态图像 (t={final_t})...")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data[final_t, :, :, 0], cmap='viridis')
            ax.set_title(f'Final State (t={final_t})', fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Concentration')
            plt.tight_layout()
            plt.savefig('viz_final.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 保存: viz_final.png")
            
            # 4. 生成对比图像
            print("🎨 生成演化对比图像...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            times = [0, mid_t, final_t]
            titles = ['Initial', 'Middle', 'Final']
            
            for i, (t, title) in enumerate(zip(times, titles)):
                im = axes[i].imshow(data[t, :, :, 0], cmap='viridis')
                axes[i].set_title(f'{title} (t={t})')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                plt.colorbar(im, ax=axes[i])
            
            plt.suptitle('2D Reaction-Diffusion Evolution', fontsize=16)
            plt.tight_layout()
            plt.savefig('viz_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ 保存: viz_comparison.png")
            
            print("\n🎉 所有图像生成完成!")
            print("📁 生成的文件:")
            print("   - viz_initial.png")
            print("   - viz_middle.png")
            print("   - viz_final.png")
            print("   - viz_comparison.png")
            
            # 验证文件是否存在
            files = ['viz_initial.png', 'viz_middle.png', 'viz_final.png', 'viz_comparison.png']
            for file in files:
                if os.path.exists(file):
                    size = os.path.getsize(file)
                    print(f"✅ {file} - {size} bytes")
                else:
                    print(f"❌ {file} - 文件不存在")
                    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()