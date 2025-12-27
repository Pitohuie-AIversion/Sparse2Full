#!/usr/bin/env python3
"""
生成AR训练结果可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# 统一安全字体配置
from utils.font_config import apply_safe_matplotlib_fonts
apply_safe_matplotlib_fonts(prefer_chinese=True, base_font_size=10)
sns.set_style("whitegrid")

# 使用更兼容的样式
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

def load_test_results():
    """加载测试结果数据"""
    results_path = Path("runs/AR-DR2D-T20-SwinUNet-s2025/test_results.json")
    
    if not results_path.exists():
        print(f"未找到测试结果文件: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data

def create_metrics_comparison_chart(metrics_data, save_path):
    """创建指标对比柱状图"""
    # 提取指标数据
    metrics = metrics_data['final_test_metrics']
    
    # 准备数据
    metric_names = []
    metric_values = []
    
    for key, value in metrics.items():
        if key != 'test_loss':  # 排除test_loss，因为它的量级不同
            metric_names.append(key.upper())
            metric_values.append(value)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建柱状图
    bars = ax.bar(metric_names, metric_values, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                         '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE'])
    
    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 设置标题和标签
    ax.set_title('AR模型测试指标对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"指标对比图已保存到: {save_path}")

def create_performance_summary_chart(metrics_data, save_path):
    """创建性能总结雷达图"""
    metrics = metrics_data['final_test_metrics']
    
    # 选择主要指标进行雷达图展示
    radar_metrics = {
        'REL_L2': 1.0 / (1.0 + metrics['rel_l2']),  # 转换为越大越好
        'MAE': 1.0 / (1.0 + metrics['mae']),        # 转换为越大越好
        'PSNR': metrics['psnr'] / 50.0,             # 归一化到0-1
        'SSIM': metrics['ssim'],                     # 已经是0-1
        'FRMSE_LOW': 1.0 / (1.0 + metrics['frmse_low']),  # 转换为越大越好
        'BRMSE': 1.0 / (1.0 + metrics['brmse'])     # 转换为越大越好
    }
    
    # 准备雷达图数据
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    values += values[:1]  # 闭合图形
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, label='AR-SwinUNet', color='#FF6B6B')
    ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    
    # 添加网格线
    ax.grid(True)
    
    # 设置标题
    ax.set_title('AR模型性能雷达图', size=16, fontweight='bold', pad=30)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"性能雷达图已保存到: {save_path}")

def create_training_summary_table(metrics_data, save_path):
    """创建训练结果总结表格图"""
    metrics = metrics_data['final_test_metrics']
    
    # 准备表格数据
    table_data = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table_data.append([key.upper(), f"{value:.6f}"])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=['指标名称', '数值'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.4, 0.6])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # 设置标题
    plt.title('AR模型测试结果详细表格', fontsize=16, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结果表格已保存到: {save_path}")

def create_error_analysis_chart(metrics_data, save_path):
    """创建误差分析图"""
    metrics = metrics_data['final_test_metrics']
    
    # 准备误差数据
    error_types = ['REL_L2', 'MAE', 'FRMSE_LOW', 'FRMSE_MID', 'FRMSE_HIGH', 'BRMSE', 'CRMSE']
    error_values = [
        metrics['rel_l2'],
        metrics['mae'],
        metrics['frmse_low'],
        metrics['frmse_mid'],
        metrics['frmse_high'],
        metrics['brmse'],
        metrics['crmse']
    ]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：误差类型对比
    colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))
    bars = ax1.bar(error_types, error_values, color=colors)
    
    # 添加数值标签
    for bar, value in zip(bars, error_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('不同类型误差对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('误差类型')
    ax1.set_ylabel('误差值')
    ax1.tick_params(axis='x', rotation=45)
    
    # 右图：频域误差分析
    freq_errors = ['FRMSE_LOW', 'FRMSE_MID', 'FRMSE_HIGH']
    freq_values = [metrics['frmse_low'], metrics['frmse_mid'], metrics['frmse_high']]
    
    wedges, texts, autotexts = ax2.pie(freq_values, labels=freq_errors, autopct='%1.3f',
                                      colors=['#FF9999', '#66B2FF', '#99FF99'])
    ax2.set_title('频域误差分布', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"误差分析图已保存到: {save_path}")

def main():
    """主函数"""
    print("开始生成AR训练结果可视化图表...")
    
    # 创建输出目录
    output_dir = Path("paper_package/figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试结果
    test_data = load_test_results()
    if test_data is None:
        print("无法加载测试结果，退出程序")
        return
    
    print("测试结果加载成功，开始生成可视化图表...")
    
    # 生成各种图表
    create_metrics_comparison_chart(test_data, output_dir / "metrics_comparison.png")
    create_performance_summary_chart(test_data, output_dir / "performance_radar.png")
    create_training_summary_table(test_data, output_dir / "results_table.png")
    create_error_analysis_chart(test_data, output_dir / "error_analysis.png")
    
    print("\n所有可视化图表生成完成！")
    print(f"图表保存位置: {output_dir.absolute()}")
    
    # 列出生成的文件
    generated_files = list(output_dir.glob("*.png"))
    print(f"\n生成的文件 ({len(generated_files)} 个):")
    for file in generated_files:
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()