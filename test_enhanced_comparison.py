#!/usr/bin/env python3
"""
测试增强模型对比表格创建
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

def test_load_data():
    """测试数据加载"""
    print("测试数据加载...")
    
    # 测试资源文件
    resource_file = "paper_package/metrics/model_resources.json"
    if os.path.exists(resource_file):
        print(f"✓ 资源文件存在: {resource_file}")
        try:
            with open(resource_file, 'r', encoding='utf-8') as f:
                resources = json.load(f)
            print(f"✓ 成功加载 {len(resources)} 个模型的资源数据")
            print(f"  模型列表: {list(resources.keys())}")
        except Exception as e:
            print(f"✗ 加载资源文件失败: {e}")
            return False
    else:
        print(f"✗ 资源文件不存在: {resource_file}")
        return False
    
    # 测试排名文件
    ranking_file = "paper_package/metrics/original_model_ranking.csv"
    if os.path.exists(ranking_file):
        print(f"✓ 排名文件存在: {ranking_file}")
        try:
            ranking_df = pd.read_csv(ranking_file)
            print(f"✓ 成功加载 {len(ranking_df)} 个模型的排名数据")
            print(f"  列名: {list(ranking_df.columns)}")
            print(f"  模型列表: {list(ranking_df['模型'])}")
        except Exception as e:
            print(f"✗ 加载排名文件失败: {e}")
            return False
    else:
        print(f"✗ 排名文件不存在: {ranking_file}")
        return False
    
    return True

def create_simple_enhanced_table():
    """创建简化的增强表格"""
    print("\n创建简化的增强表格...")
    
    try:
        # 加载数据
        with open("paper_package/metrics/model_resources.json", 'r', encoding='utf-8') as f:
            resources = json.load(f)
        
        ranking_df = pd.read_csv("paper_package/metrics/original_model_ranking.csv")
        
        # 合并数据
        enhanced_data = []
        
        for _, row in ranking_df.iterrows():
            model_name = row['模型']
            
            # 基础数据
            base_data = {
                '模型': model_name,
                'Rel-L2': row['Rel-L2'],
                'PSNR': row['PSNR'],
                'SSIM': row['SSIM'],
                '训练时间(分钟)': row['训练时间(分钟)']
            }
            
            # 添加资源数据
            if model_name in resources:
                resource_data = resources[model_name]
                base_data.update({
                    '参数量(M)': resource_data['total_params_M'],
                    'FLOPs(G)': resource_data['flops_G'],
                    '推理延迟(ms)': resource_data['latency_ms'],
                    'FPS': resource_data['fps'],
                    '训练显存(GB)': resource_data['training_memory_GB'],
                    '推理显存(GB)': resource_data['inference_memory_GB']
                })
            else:
                base_data.update({
                    '参数量(M)': 0.0,
                    'FLOPs(G)': 0.0,
                    '推理延迟(ms)': 0.0,
                    'FPS': 0.0,
                    '训练显存(GB)': 0.0,
                    '推理显存(GB)': 0.0
                })
            
            enhanced_data.append(base_data)
        
        # 创建DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        enhanced_df = enhanced_df.sort_values('Rel-L2').reset_index(drop=True)
        enhanced_df.index = enhanced_df.index + 1
        
        print(f"✓ 成功创建增强表格，包含 {len(enhanced_df)} 个模型")
        
        # 保存CSV
        output_dir = Path("paper_package/metrics")
        csv_file = output_dir / "enhanced_model_comparison_simple.csv"
        enhanced_df.to_csv(csv_file, index=True, encoding='utf-8')
        print(f"✓ CSV文件已保存: {csv_file}")
        
        # 保存Excel
        excel_file = output_dir / "enhanced_model_comparison_simple.xlsx"
        enhanced_df.to_excel(excel_file, index=True)
        print(f"✓ Excel文件已保存: {excel_file}")
        
        # 创建简化Markdown
        markdown_file = output_dir / "enhanced_model_comparison_simple.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write("# 增强模型性能对比表（含资源统计）\n\n")
            f.write("## 综合对比\n\n")
            
            # 格式化显示
            display_df = enhanced_df.copy()
            for col in ['Rel-L2', 'PSNR', 'SSIM']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            for col in ['参数量(M)', 'FLOPs(G)', '推理延迟(ms)', 'FPS', '训练显存(GB)', '推理显存(GB)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            
            f.write(display_df.to_markdown(index=True))
            f.write("\n\n")
            
            # 前3名
            f.write("## 🏆 性能排名前3\n\n")
            for i, (_, row) in enumerate(enhanced_df.head(3).iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][i-1]
                f.write(f"{medal} **{row['模型']}**\n")
                f.write(f"   - Rel-L2: {row['Rel-L2']:.4f}\n")
                f.write(f"   - 参数量: {row['参数量(M)']:.2f}M\n")
                f.write(f"   - 推理延迟: {row['推理延迟(ms)']:.1f}ms\n\n")
        
        print(f"✓ Markdown文件已保存: {markdown_file}")
        
        return enhanced_df
        
    except Exception as e:
        print(f"✗ 创建增强表格失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=== 测试增强模型对比表格创建 ===\n")
    
    # 测试数据加载
    if not test_load_data():
        print("数据加载测试失败，退出")
        return
    
    # 创建简化表格
    enhanced_df = create_simple_enhanced_table()
    if enhanced_df is not None:
        print(f"\n✓ 成功创建增强表格，包含 {len(enhanced_df)} 个模型")
        print("\n前3名模型:")
        for i, (_, row) in enumerate(enhanced_df.head(3).iterrows(), 1):
            print(f"{i}. {row['模型']}: Rel-L2={row['Rel-L2']:.4f}, 参数量={row['参数量(M)']:.1f}M")
    else:
        print("✗ 创建增强表格失败")

if __name__ == "__main__":
    main()