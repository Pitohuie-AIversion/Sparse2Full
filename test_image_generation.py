#!/usr/bin/env python3
"""
简单的测试图像生成脚本
用于验证matplotlib图像生成和保存功能是否正常
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def create_test_images():
    """创建几个简单的测试图像"""
    
    # 设置matplotlib后端
    plt.switch_backend('Agg')
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # 图像1: 简单的正弦波
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    Z1 = np.sin(X) * np.cos(Y)
    im1 = ax1.imshow(Z1, cmap='viridis', extent=[0, 10, 0, 10])
    ax1.set_title('Test Image 1: Sine Wave Pattern')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    filename1 = 'test_image_1.png'
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ 已保存: {filename1}")
    
    # 图像2: 高斯分布
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    Z2 = np.exp(-((X-5)**2 + (Y-5)**2)/4)
    im2 = ax2.imshow(Z2, cmap='plasma', extent=[0, 10, 0, 10])
    ax2.set_title('Test Image 2: Gaussian Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    filename2 = 'test_image_2.png'
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ 已保存: {filename2}")
    
    # 图像3: 对比图
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左侧子图
    im3a = ax3a.imshow(Z1, cmap='coolwarm', extent=[0, 10, 0, 10])
    ax3a.set_title('Pattern A')
    ax3a.set_xlabel('X')
    ax3a.set_ylabel('Y')
    plt.colorbar(im3a, ax=ax3a)
    
    # 右侧子图
    im3b = ax3b.imshow(Z2, cmap='coolwarm', extent=[0, 10, 0, 10])
    ax3b.set_title('Pattern B')
    ax3b.set_xlabel('X')
    ax3b.set_ylabel('Y')
    plt.colorbar(im3b, ax=ax3b)
    
    plt.tight_layout()
    filename3 = 'test_comparison.png'
    plt.savefig(filename3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ 已保存: {filename3}")
    
    return [filename1, filename2, filename3]

def verify_files(filenames):
    """验证文件是否成功生成"""
    print("\n=== 文件验证 ===")
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    for filename in filenames:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename} - 大小: {size} bytes")
        else:
            print(f"✗ {filename} - 文件不存在!")
    
    # 列出所有PNG文件
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    print(f"\n当前目录下的所有PNG文件: {png_files}")

if __name__ == "__main__":
    print("开始生成测试图像...")
    print(f"工作目录: {os.getcwd()}")
    
    try:
        filenames = create_test_images()
        verify_files(filenames)
        print("\n✓ 测试图像生成完成!")
        
    except Exception as e:
        print(f"✗ 生成图像时出错: {e}")
        import traceback
        traceback.print_exc()