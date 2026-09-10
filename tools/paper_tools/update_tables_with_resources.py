import json
import pandas as pd
import os

# 加载资源数据
with open("paper_package/metrics/model_resources.json", 'r') as f:
    resources = json.load(f)

# 加载原始排名数据
ranking_df = pd.read_csv("paper_package/metrics/original_model_ranking.csv")

print(f"加载了 {len(ranking_df)} 个模型的性能数据")
print(f"加载了 {len(resources)} 个模型的资源数据")

# 合并数据
enhanced_data = []

for _, row in ranking_df.iterrows():
    model_name = row['模型']
    
    # 基础数据
    base_data = {
        '排名': row.name + 1,
        '模型': model_name,
        'Rel-L2': row['Rel-L2'],
        'MAE': row['MAE'],
        'PSNR': row['PSNR'],
        'SSIM': row['SSIM'],
        'BRMSE': row['BRMSE'],
        'CRMSE': row['CRMSE'],
        '最佳验证损失': row['最佳验证损失'],
        '训练时间(分钟)': row['训练时间(分钟)']
    }
    
    # 添加资源数据
    if model_name in resources:
        resource_data = resources[model_name]
        base_data.update({
            '参数量(M)': round(resource_data['total_params_M'], 2),
            '可训练参数(M)': round(resource_data['trainable_params_M'], 2),
            '模型大小(MB)': round(resource_data['model_size_MB'], 1),
            'FLOPs(G)': round(resource_data['flops_G'], 1),
            'GFLOPS/s': round(resource_data['gflops_per_sec'], 1),
            '训练显存(GB)': round(resource_data['training_memory_GB'], 2),
            '推理显存(GB)': round(resource_data['inference_memory_GB'], 2),
            '推理延迟(ms)': round(resource_data['latency_ms'], 1),
            '延迟标准差(ms)': round(resource_data['latency_std_ms'], 2),
            'FPS': round(resource_data['fps'], 1)
        })
    else:
        print(f"警告：未找到模型 {model_name} 的资源数据")
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

# 保存增强的CSV
csv_file = "paper_package/metrics/enhanced_model_comparison_with_resources.csv"
enhanced_df.to_csv(csv_file, index=False, encoding='utf-8')
print(f"增强CSV已保存: {csv_file}")

# 保存Excel文件
excel_file = "paper_package/metrics/comprehensive_model_comparison_with_resources.xlsx"
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # 主表
    enhanced_df.to_excel(writer, sheet_name='完整对比', index=False)
    
    # 性能指标表
    performance_cols = ['排名', '模型', 'Rel-L2', 'PSNR', 'SSIM', '训练时间(分钟)']
    enhanced_df[performance_cols].to_excel(writer, sheet_name='性能指标', index=False)
    
    # 资源指标表
    resource_cols = ['排名', '模型', '参数量(M)', 'FLOPs(G)', '推理延迟(ms)', 'FPS', '训练显存(GB)', '推理显存(GB)']
    enhanced_df[resource_cols].to_excel(writer, sheet_name='资源指标', index=False)

print(f"Excel文件已保存: {excel_file}")

# 创建Markdown表格
markdown_file = "paper_package/metrics/enhanced_model_comparison_with_resources.md"

with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write("# 🚀 Sparse2Full 增强模型性能对比表（含资源统计）\n\n")
    f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 主要对比表
    f.write("## 📊 综合性能对比（含资源统计）\n\n")
    
    # 选择主要列显示
    display_cols = [
        '排名', '模型', 'Rel-L2', 'PSNR', 'SSIM', 
        '参数量(M)', 'FLOPs(G)', '推理延迟(ms)', 'FPS',
        '训练时间(分钟)'
    ]
    
    display_df = enhanced_df[display_cols].copy()
    f.write(display_df.to_markdown(index=False))
    f.write("\n\n")
    
    # 资源统计分析
    f.write("## 📈 资源统计分析\n\n")
    
    # 参数量统计
    f.write("### 🔢 参数量统计\n\n")
    f.write(f"- 最少参数: {enhanced_df['参数量(M)'].min():.2f}M ({enhanced_df.loc[enhanced_df['参数量(M)'].idxmin(), '模型']})\n")
    f.write(f"- 最多参数: {enhanced_df['参数量(M)'].max():.2f}M ({enhanced_df.loc[enhanced_df['参数量(M)'].idxmax(), '模型']})\n")
    f.write(f"- 平均参数: {enhanced_df['参数量(M)'].mean():.2f}M\n\n")
    
    # FLOPs统计
    f.write("### ⚡ FLOPs统计\n\n")
    f.write(f"- 最少FLOPs: {enhanced_df['FLOPs(G)'].min():.1f}G ({enhanced_df.loc[enhanced_df['FLOPs(G)'].idxmin(), '模型']})\n")
    f.write(f"- 最多FLOPs: {enhanced_df['FLOPs(G)'].max():.1f}G ({enhanced_df.loc[enhanced_df['FLOPs(G)'].idxmax(), '模型']})\n")
    f.write(f"- 平均FLOPs: {enhanced_df['FLOPs(G)'].mean():.1f}G\n\n")
    
    # 推理性能统计
    f.write("### 🚀 推理性能统计\n\n")
    valid_latency = enhanced_df[enhanced_df['推理延迟(ms)'] > 0]
    if not valid_latency.empty:
        f.write(f"- 最快推理: {valid_latency['推理延迟(ms)'].min():.1f}ms ({valid_latency.loc[valid_latency['推理延迟(ms)'].idxmin(), '模型']})\n")
        f.write(f"- 最慢推理: {valid_latency['推理延迟(ms)'].max():.1f}ms ({valid_latency.loc[valid_latency['推理延迟(ms)'].idxmax(), '模型']})\n")
        f.write(f"- 平均延迟: {valid_latency['推理延迟(ms)'].mean():.1f}ms\n\n")
    
    # 最佳模型推荐
    f.write("## 🏆 最佳模型推荐\n\n")
    
    top3 = enhanced_df.head(3)
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][i-1]
        f.write(f"{medal} **{row['模型']}**\n")
        f.write(f"   - Rel-L2: {row['Rel-L2']:.4f}\n")
        f.write(f"   - PSNR: {row['PSNR']:.2f} dB\n")
        f.write(f"   - 参数量: {row['参数量(M)']:.2f}M\n")
        f.write(f"   - FLOPs: {row['FLOPs(G)']:.1f}G\n")
        f.write(f"   - 推理延迟: {row['推理延迟(ms)']:.1f}ms\n\n")
    
    # 使用建议
    f.write("## 💡 使用建议\n\n")
    f.write("### 🎯 按应用场景选择\n\n")
    
    best_accuracy = enhanced_df.iloc[0]
    fastest_model = enhanced_df.loc[enhanced_df['推理延迟(ms)'].idxmin()] if enhanced_df['推理延迟(ms)'].max() > 0 else None
    smallest_model = enhanced_df.loc[enhanced_df['参数量(M)'].idxmin()]
    
    f.write(f"- **🔬 科研项目**: {best_accuracy['模型']} (最高精度)\n")
    if fastest_model is not None:
        f.write(f"- **🚀 实时应用**: {fastest_model['模型']} (最快推理)\n")
    f.write(f"- **📱 边缘计算**: {smallest_model['模型']} (最少参数)\n\n")
    
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