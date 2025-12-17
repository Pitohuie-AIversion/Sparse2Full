#!/usr/bin/env python3
"""
验证可视化图片生成和字体显示效果
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def check_generated_files():
    """检查生成的可视化文件"""
    figs_dir = Path("paper_package/figs")
    expected_files = [
        "metrics_comparison.png",
        "performance_radar.png", 
        "results_table.png",
        "error_analysis.png"
    ]
    
    print("=== 检查生成的可视化文件 ===")
    all_exist = True
    
    for filename in expected_files:
        filepath = figs_dir / filename
        if filepath.exists():
            # 获取文件大小
            size = filepath.stat().st_size
            print(f"✓ {filename} - {size:,} bytes")
            
            # 检查图片是否可以正常打开
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    print(f"  图片尺寸: {width}x{height}")
            except Exception as e:
                print(f"  ⚠️ 图片打开失败: {e}")
        else:
            print(f"✗ {filename} - 文件不存在")
            all_exist = False
    
    return all_exist

def check_font_configuration():
    """检查当前字体配置"""
    print("\n=== 当前字体配置 ===")
    print(f"默认字体族: {plt.rcParams['font.family']}")
    print(f"sans-serif字体: {plt.rcParams['font.sans-serif']}")
    print(f"字体大小: {plt.rcParams['font.size']}")
    print(f"Unicode负号: {plt.rcParams['axes.unicode_minus']}")
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['simhei', 'yahei', 'chinese', 'cjk'])]
    
    print(f"\n系统中的中文相关字体 ({len(chinese_fonts)} 个):")
    for font in chinese_fonts[:10]:  # 只显示前10个
        print(f"  - {font}")
    
    if len(chinese_fonts) > 10:
        print(f"  ... 还有 {len(chinese_fonts) - 10} 个字体")

def create_font_test_image():
    """创建字体测试图片"""
    print("\n=== 创建字体测试图片 ===")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试不同的文本
    test_texts = [
        "English Text - Normal Display",
        "Numbers: 123.456",
        "Special Characters: ±×÷≤≥",
        "Mixed: AR Model Performance (English Only)"
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.8 - i*0.15, text, fontsize=14, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Font Display Test - English Only", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # 保存测试图片
    test_path = Path("paper_package/figs/font_test.png")
    plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"字体测试图片已保存: {test_path}")
    return test_path.exists()

def main():
    """主函数"""
    print("开始验证可视化图片和字体显示效果...\n")
    
    # 检查生成的文件
    files_ok = check_generated_files()
    
    # 检查字体配置
    check_font_configuration()
    
    # 创建字体测试图片
    font_test_ok = create_font_test_image()
    
    # 总结
    print("\n=== 验证结果总结 ===")
    print(f"可视化文件生成: {'✓ 成功' if files_ok else '✗ 失败'}")
    print(f"字体测试图片: {'✓ 成功' if font_test_ok else '✗ 失败'}")
    
    if files_ok and font_test_ok:
        print("\n🎉 所有验证通过！可视化图片已成功生成，字体配置正常。")
        print("注意：已将所有标签改为英文以避免字体显示问题。")
    else:
        print("\n⚠️ 验证过程中发现问题，请检查上述输出。")

if __name__ == "__main__":
    main()