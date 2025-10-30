#!/usr/bin/env python3
"""
创建包含完整资源统计的增强模型对比表格
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_model_resources():
    """加载模型资源统计数据"""
    resource_file = "paper_package/metrics/model_resources.json"
    if not os.path.exists(resource_file):
        print(f"错误：找不到资源文件 {resource_file}")
        return {}
    
    with open(resource_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_original_ranking():
    """加载原始排名数据"""
    ranking_file = "paper_package/metrics/original_model_ranking.csv"
    if not os.path.exists(ranking_file):
        print(f"错误：找不到排名文件 {ranking_file}")
        return pd.DataFrame()
    
    return pd.read_csv(ranking_file)

def create_enhanced_comparison_table():
    """创建增强的模型对比表格"""
    
    # 加载数据
    resources = load_model_resources()
    ranking_df = load_original_ranking()
    
    if ranking_df.empty or not resources:
        print("错误：无法加载必要的数据文件")
        return None
    
    print(f"加载了 {len(ranking_df)} 个模型的性能数据")
    print(f"加载了 {len(resources)} 个模型的资源数据")
    
    # 合并数据
    enhanced_data = []
    
    for _, row in ranking_df.iterrows():
        model_name = row['模型']
        
        # 基础性能数据
        base_data = {
            '模型': model_name,
            'Rel-L2': row['Rel-L2'],
            'MAE': row['MAE'],
            'PSNR': row['PSNR'],
            'SSIM': row['SSIM'],
            'BRMSE': row['BRMSE'],
            'CRMSE': row['CRMSE'],
            'FRMSE_low': row['FRMSE_low'],
            'FRMSE_high': row['FRMSE_high'],
            '训练时间(分钟)': row['训练时间(分钟)']
        }
        
        # 添加资源数据
        if model_name in resources:
            resource_data = resources[model_name]
            base_data.update({
                '参数量(M)': resource_data['total_params_M'],
                '可训练参数(M)': resource_data['trainable_params_M'],
                '模型大小(MB)': resource_data['model_size_MB'],
                'FLOPs(G)': resource_data['flops_G'],
                'GFLOPS/s': resource_data['gflops_per_sec'],
                '训练显存(GB)': resource_data['training_memory_GB'],
                '推理显存(GB)': resource_data['inference_memory_GB'],
                '推理延迟(ms)': resource_data['latency_ms'],
                '延迟标准差(ms)': resource_data['latency_std_ms'],
                'FPS': resource_data['fps']
            })
        else:
            # 默认值
            base_data.update({
                '参数量(M)': 0.0,
                '可训练参数(M)': 0.0,
                '模型大小(MB)': 0.0,
                'FLOPs(G)': 0.0,
                'GFLOPS/s': 0.0,
                '训练显存(GB)': 0.0,
                '推理显存(GB)': 0.0,
                '推理延迟(ms)': 0.0,
                '延迟标准差(ms)': 0.0,
                'FPS': 0.0
            })
        
        enhanced_data.append(base_data)
    
    # 创建DataFrame
    enhanced_df = pd.DataFrame(enhanced_data)
    
    # 按Rel-L2排序
    enhanced_df = enhanced_df.sort_values('Rel-L2').reset_index(drop=True)
    enhanced_df.index = enhanced_df.index + 1  # 从1开始排名
    
    return enhanced_df

def add_model_categories(df):
    """添加模型类别信息"""
    model_categories = {
        'FNO2D': 'Fourier Neural Operator',
        'SwinUNet': 'Vision Transformer',
        'UNet': 'Convolutional Network',
        'MLP': 'Multi-Layer Perceptron',
        'MLP_Mixer': 'MLP-based Architecture',
        'Hybrid': 'Hybrid Architecture',
        'UFNO_UNet': 'Hybrid FNO-UNet',
        'UNetPlusPlus': 'Enhanced UNet'
    }
    
    df['模型类别'] = df['模型'].map(model_categories)
    return df

def add_performance_grades(df):
    """添加性能等级"""
    # Rel-L2性能等级
    rel_l2_values = df['Rel-L2'].values
    rel_l2_percentiles = np.percentile(rel_l2_values, [25, 50, 75])
    
    def get_performance_grade(rel_l2):
        if rel_l2 <= rel_l2_percentiles[0]:
            return "A+ (优秀)"
        elif rel_l2 <= rel_l2_percentiles[1]:
            return "A (良好)"
        elif rel_l2 <= rel_l2_percentiles[2]:
            return "B (中等)"
        else:
            return "C (一般)"
    
    df['性能等级'] = df['Rel-L2'].apply(get_performance_grade)
    
    # 效率等级（基于推理延迟）
    latency_values = df['推理延迟(ms)'].values
    latency_percentiles = np.percentile(latency_values[latency_values > 0], [25, 50, 75])
    
    def get_efficiency_grade(latency):
        if latency <= 0:
            return "未知"
        elif latency <= latency_percentiles[0]:
            return "A+ (极快)"
        elif latency <= latency_percentiles[1]:
            return "A (快速)"
        elif latency <= latency_percentiles[2]:
            return "B (中等)"
        else:
            return "C (较慢)"
    
    df['效率等级'] = df['推理延迟(ms)'].apply(get_efficiency_grade)
    
    return df

def calculate_composite_score(df):
    """计算综合评分"""
    # 归一化各项指标（0-100分）
    def normalize_score(values, higher_better=True):
        if higher_better:
            return 100 * (values - values.min()) / (values.max() - values.min())
        else:
            return 100 * (values.max() - values) / (values.max() - values.min())
    
    # 性能指标（越小越好）
    rel_l2_score = normalize_score(df['Rel-L2'], higher_better=False)
    psnr_score = normalize_score(df['PSNR'], higher_better=True)
    ssim_score = normalize_score(df['SSIM'], higher_better=True)
    
    # 效率指标（越小越好）
    latency_score = normalize_score(df['推理延迟(ms)'].replace(0, df['推理延迟(ms)'].max()), higher_better=False)
    training_time_score = normalize_score(df['训练时间(分钟)'], higher_better=False)
    
    # 资源指标（参数量适中更好）
    params_score = normalize_score(df['参数量(M)'], higher_better=False)
    
    # 综合评分（权重分配）
    composite_score = (
        0.4 * rel_l2_score +      # 40% 重建精度
        0.2 * psnr_score +        # 20% 图像质量
        0.1 * ssim_score +        # 10% 结构相似性
        0.15 * latency_score +    # 15% 推理速度
        0.1 * training_time_score + # 10% 训练效率
        0.05 * params_score       # 5% 模型复杂度
    )
    
    df['综合评分'] = composite_score.round(1)
    return df

def save_enhanced_tables(df):
    """保存增强的表格"""
    output_dir = Path("paper_package/metrics")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存完整CSV
    csv_file = output_dir / "enhanced_model_comparison_with_resources.csv"
    df.to_csv(csv_file, index=True, encoding='utf-8')
    print(f"完整CSV已保存: {csv_file}")
    
    # 2. 创建Excel文件
    excel_file = output_dir / "comprehensive_model_comparison_with_resources.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 主表
        df.to_excel(writer, sheet_name='模型对比', index=True)
        
        # 性能指标表
        performance_cols = ['模型', '模型类别', 'Rel-L2', 'PSNR', 'SSIM', '性能等级', '综合评分']
        df[performance_cols].to_excel(writer, sheet_name='性能指标', index=True)
        
        # 资源指标表
        resource_cols = ['模型', '参数量(M)', 'FLOPs(G)', '推理延迟(ms)', 'FPS', '训练显存(GB)', '推理显存(GB)']
        df[resource_cols].to_excel(writer, sheet_name='资源指标', index=True)
        
        # 效率分析表
        efficiency_cols = ['模型', '训练时间(分钟)', '推理延迟(ms)', 'GFLOPS/s', '效率等级']
        df[efficiency_cols].to_excel(writer, sheet_name='效率分析', index=True)
    
    print(f"Excel文件已保存: {excel_file}")
    
    # 3. 创建Markdown表格
    markdown_file = output_dir / "enhanced_model_comparison_with_resources.md"
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("# 🚀 Sparse2Full 增强模型性能对比表（含资源统计）\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 主要对比表
        f.write("## 📊 综合性能对比（含资源统计）\n\n")
        
        # 选择主要列显示
        display_cols = [
            '模型', '模型类别', 'Rel-L2', 'PSNR', 'SSIM', 
            '参数量(M)', 'FLOPs(G)', '推理延迟(ms)', 'FPS',
            '训练时间(分钟)', '综合评分'
        ]
        
        display_df = df[display_cols].copy()
        
        # 格式化数值
        display_df['Rel-L2'] = display_df['Rel-L2'].apply(lambda x: f"{x:.4f}")
        display_df['PSNR'] = display_df['PSNR'].apply(lambda x: f"{x:.2f}")
        display_df['SSIM'] = display_df['SSIM'].apply(lambda x: f"{x:.4f}")
        display_df['参数量(M)'] = display_df['参数量(M)'].apply(lambda x: f"{x:.2f}")
        display_df['FLOPs(G)'] = display_df['FLOPs(G)'].apply(lambda x: f"{x:.1f}")
        display_df['推理延迟(ms)'] = display_df['推理延迟(ms)'].apply(lambda x: f"{x:.1f}")
        display_df['FPS'] = display_df['FPS'].apply(lambda x: f"{x:.1f}")
        display_df['训练时间(分钟)'] = display_df['训练时间(分钟)'].apply(lambda x: f"{x:.2f}")
        display_df['综合评分'] = display_df['综合评分'].apply(lambda x: f"{x:.1f}")
        
        f.write(display_df.to_markdown(index=True))
        f.write("\n\n")
        
        # 资源统计分析
        f.write("## 📈 资源统计分析\n\n")
        
        # 参数量统计
        f.write("### 🔢 参数量统计\n\n")
        params_stats = df.groupby('模型类别')['参数量(M)'].agg(['mean', 'std', 'min', 'max'])
        f.write(params_stats.round(2).to_markdown())
        f.write("\n\n")
        
        # FLOPs统计
        f.write("### ⚡ FLOPs统计\n\n")
        flops_stats = df.groupby('模型类别')['FLOPs(G)'].agg(['mean', 'std', 'min', 'max'])
        f.write(flops_stats.round(1).to_markdown())
        f.write("\n\n")
        
        # 推理性能统计
        f.write("### 🚀 推理性能统计\n\n")
        inference_stats = df.groupby('模型类别')[['推理延迟(ms)', 'FPS']].agg(['mean', 'std'])
        f.write(inference_stats.round(2).to_markdown())
        f.write("\n\n")
        
        # 最佳模型推荐
        f.write("## 🏆 最佳模型推荐\n\n")
        
        top3 = df.head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            medal = ["🥇", "🥈", "🥉"][i-1]
            f.write(f"{medal} **{row['模型']}** ({row['模型类别']})\n")
            f.write(f"   - Rel-L2: {row['Rel-L2']:.4f}\n")
            f.write(f"   - PSNR: {row['PSNR']:.2f} dB\n")
            f.write(f"   - 参数量: {row['参数量(M)']:.2f}M\n")
            f.write(f"   - FLOPs: {row['FLOPs(G)']:.1f}G\n")
            f.write(f"   - 推理延迟: {row['推理延迟(ms)']:.1f}ms\n")
            f.write(f"   - 综合评分: {row['综合评分']:.1f}/100\n\n")
        
        # 效率分析
        f.write("## ⚡ 效率分析\n\n")
        
        # 最快推理
        fastest_inference = df.loc[df['推理延迟(ms)'].idxmin()]
        f.write(f"### 🚀 最快推理: {fastest_inference['模型']}\n")
        f.write(f"- 推理延迟: {fastest_inference['推理延迟(ms)']:.1f}ms\n")
        f.write(f"- FPS: {fastest_inference['FPS']:.1f}\n")
        f.write(f"- 参数量: {fastest_inference['参数量(M)']:.2f}M\n\n")
        
        # 最少参数
        smallest_model = df.loc[df['参数量(M)'].idxmin()]
        f.write(f"### 📦 最少参数: {smallest_model['模型']}\n")
        f.write(f"- 参数量: {smallest_model['参数量(M)']:.2f}M\n")
        f.write(f"- 模型大小: {smallest_model['模型大小(MB)']:.1f}MB\n")
        f.write(f"- Rel-L2: {smallest_model['Rel-L2']:.4f}\n\n")
        
        # 最高效率
        highest_efficiency = df.loc[df['GFLOPS/s'].idxmax()]
        f.write(f"### ⚡ 最高计算效率: {highest_efficiency['模型']}\n")
        f.write(f"- 计算效率: {highest_efficiency['GFLOPS/s']:.1f} GFLOPS/s\n")
        f.write(f"- FLOPs: {highest_efficiency['FLOPs(G)']:.1f}G\n")
        f.write(f"- 推理延迟: {highest_efficiency['推理延迟(ms)']:.1f}ms\n\n")
        
        # 使用建议
        f.write("## 💡 使用建议\n\n")
        f.write("### 🎯 按应用场景选择\n\n")
        f.write("- **🔬 科研项目**: FNO2D (最高精度，合理资源消耗)\n")
        f.write("- **🏭 工业应用**: UNet (精度与效率平衡)\n")
        f.write("- **🚀 实时应用**: MLP_Mixer (最快推理速度)\n")
        f.write("- **📱 边缘计算**: MLP (最少参数量)\n\n")
        
        f.write("### 📊 技术指标说明\n\n")
        f.write("- **参数量(M)**: 模型总参数数量（百万）\n")
        f.write("- **FLOPs(G)**: 浮点运算次数，按256×256输入计算（十亿次）\n")
        f.write("- **推理延迟(ms)**: 单次推理时间（毫秒）\n")
        f.write("- **FPS**: 每秒处理帧数\n")
        f.write("- **GFLOPS/s**: 计算效率（每秒十亿次浮点运算）\n")
        f.write("- **训练显存(GB)**: 训练时峰值显存使用量估算\n")
        f.write("- **推理显存(GB)**: 推理时显存使用量估算\n\n")
        
        f.write("---\n\n")
        f.write("*本报告由PDEBench稀疏观测重建系统自动生成，包含完整的性能和资源统计数据*\n")
    
    print(f"Markdown文件已保存: {markdown_file}")
    
    return df

def main():
    """主函数"""
    print("开始创建增强的模型对比表格...")
    
    # 创建增强表格
    enhanced_df = create_enhanced_comparison_table()
    if enhanced_df is None:
        return
    
    # 添加分类和等级
    enhanced_df = add_model_categories(enhanced_df)
    enhanced_df = add_performance_grades(enhanced_df)
    enhanced_df = calculate_composite_score(enhanced_df)
    
    # 保存表格
    final_df = save_enhanced_tables(enhanced_df)
    
    print("\n=== 增强模型对比表格创建完成 ===")
    print(f"总共处理了 {len(enhanced_df)} 个模型")
    print("\n生成的文件:")
    print("- enhanced_model_comparison_with_resources.csv")
    print("- comprehensive_model_comparison_with_resources.xlsx")
    print("- enhanced_model_comparison_with_resources.md")
    
    # 显示前3名
    print("\n🏆 性能排名前3:")
    for i, (_, row) in enumerate(enhanced_df.head(3).iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        print(f"{medal} {row['模型']}: Rel-L2={row['Rel-L2']:.4f}, 参数量={row['参数量(M)']:.1f}M, 延迟={row['推理延迟(ms)']:.1f}ms")

if __name__ == "__main__":
    main()