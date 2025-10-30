#!/usr/bin/env python3
"""
批量训练结果分析脚本

分析批量训练的结果，生成性能对比报告和可视化
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# 设置项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_training_summary(batch_dir: str) -> Dict[str, Any]:
    """加载训练汇总信息"""
    summary_path = Path(batch_dir) / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"训练汇总文件不存在: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_model_metrics(batch_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
    """提取单个模型的指标"""
    model_dir = Path(batch_dir) / model_name.lower()
    
    # 查找最佳模型文件
    best_model_path = model_dir / "checkpoints" / "best.pth"
    if not best_model_path.exists():
        print(f"警告: 模型 {model_name} 的最佳检查点不存在")
        return None
    
    # 查找训练日志
    train_log_path = model_dir / "train.log"
    if not train_log_path.exists():
        print(f"警告: 模型 {model_name} 的训练日志不存在")
        return None
    
    # 解析训练日志获取最终指标
    metrics = parse_training_log(train_log_path)
    
    # 添加模型信息
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        metrics.update({
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'model_params': checkpoint.get('model_params', 0)
        })
    
    return metrics

def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """解析训练日志获取指标"""
    metrics = {
        'final_train_loss': None,
        'final_val_loss': None,
        'best_val_loss': float('inf'),
        'final_rel_l2': None,
        'final_mae': None,
        'final_psnr': None,
        'final_ssim': None
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 从后往前查找最后的验证指标
        for line in reversed(lines):
            if 'val_loss' in line and 'rel_l2' in line:
                # 解析验证指标行
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part.startswith('val_loss='):
                        metrics['final_val_loss'] = float(part.split('=')[1])
                    elif part.startswith('rel_l2='):
                        metrics['final_rel_l2'] = float(part.split('=')[1])
                    elif part.startswith('mae='):
                        metrics['final_mae'] = float(part.split('=')[1])
                    elif part.startswith('psnr='):
                        metrics['final_psnr'] = float(part.split('=')[1])
                    elif part.startswith('ssim='):
                        metrics['final_ssim'] = float(part.split('=')[1])
                break
        
        # 查找最佳验证损失
        for line in lines:
            if 'Best val_loss' in line or 'best_val_loss' in line:
                try:
                    # 提取数值
                    import re
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        metrics['best_val_loss'] = float(numbers[0])
                except:
                    pass
    
    except Exception as e:
        print(f"解析训练日志失败 {log_path}: {e}")
    
    return metrics

def create_performance_comparison(results: Dict[str, Dict[str, Any]], output_dir: Path):
    """创建性能对比图表"""
    # 准备数据
    models = []
    rel_l2_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []
    val_loss_values = []
    training_times = []
    
    for model_name, data in results.items():
        if data['status'] == 'success' and data.get('metrics'):
            models.append(model_name)
            rel_l2_values.append(data['metrics'].get('final_rel_l2', 0))
            mae_values.append(data['metrics'].get('final_mae', 0))
            psnr_values.append(data['metrics'].get('final_psnr', 0))
            ssim_values.append(data['metrics'].get('final_ssim', 0))
            val_loss_values.append(data['metrics'].get('final_val_loss', 0))
            training_times.append(data.get('training_time', 0) / 60)  # 转换为分钟
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型性能对比分析', fontsize=16, fontweight='bold')
    
    # Rel-L2误差对比
    axes[0, 0].bar(models, rel_l2_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('相对L2误差 (越小越好)')
    axes[0, 0].set_ylabel('Rel-L2')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE对比
    axes[0, 1].bar(models, mae_values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('平均绝对误差 (越小越好)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # PSNR对比
    axes[0, 2].bar(models, psnr_values, color='orange', alpha=0.7)
    axes[0, 2].set_title('峰值信噪比 (越大越好)')
    axes[0, 2].set_ylabel('PSNR (dB)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # SSIM对比
    axes[1, 0].bar(models, ssim_values, color='pink', alpha=0.7)
    axes[1, 0].set_title('结构相似性 (越大越好)')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 验证损失对比
    axes[1, 1].bar(models, val_loss_values, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('验证损失 (越小越好)')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 训练时间对比
    axes[1, 2].bar(models, training_times, color='gold', alpha=0.7)
    axes[1, 2].set_title('训练时间 (分钟)')
    axes[1, 2].set_ylabel('时间 (分钟)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ranking_table(results: Dict[str, Dict[str, Any]], output_dir: Path):
    """创建模型排名表"""
    # 准备数据
    data = []
    for model_name, result in results.items():
        if result['status'] == 'success' and result.get('metrics'):
            metrics = result['metrics']
            data.append({
                '模型': model_name,
                'Rel-L2': metrics.get('final_rel_l2', 0),
                'MAE': metrics.get('final_mae', 0),
                'PSNR': metrics.get('final_psnr', 0),
                'SSIM': metrics.get('final_ssim', 0),
                '验证损失': metrics.get('final_val_loss', 0),
                '训练时间(分钟)': round(result.get('training_time', 0) / 60, 2),
                '参数量': result.get('model_params', 0)
            })
    
    if not data:
        print("警告: 没有成功的模型数据用于排名")
        return
    
    df = pd.DataFrame(data)
    
    # 按Rel-L2排序（越小越好）
    df_sorted = df.sort_values('Rel-L2')
    
    # 保存为CSV
    df_sorted.to_csv(output_dir / 'model_ranking.csv', index=False, encoding='utf-8-sig')
    
    # 创建排名表图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 创建表格
    table_data = df_sorted.values
    table = ax.table(cellText=table_data, 
                    colLabels=df_sorted.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(df_sorted.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置最佳性能行的颜色
    if len(table_data) > 0:
        for i in range(len(df_sorted.columns)):
            table[(1, i)].set_facecolor('#E8F5E8')  # 浅绿色表示最佳
    
    ax.axis('off')
    ax.set_title('模型性能排名表 (按Rel-L2排序)', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'model_ranking_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_sorted

def generate_analysis_report(summary: Dict[str, Any], results: Dict[str, Dict[str, Any]], 
                           ranking_df: pd.DataFrame, output_dir: Path):
    """生成分析报告"""
    report_path = output_dir / 'batch_training_analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 批量训练结果分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 训练概览
        f.write("## 📊 训练概览\n\n")
        f.write(f"- **总模型数**: {summary['batch_training_info']['total_models']}\n")
        f.write(f"- **成功训练**: {summary['batch_training_info']['successful_models']}\n")
        f.write(f"- **训练失败**: {summary['batch_training_info']['failed_models']}\n")
        f.write(f"- **成功率**: {summary['batch_training_info']['successful_models'] / summary['batch_training_info']['total_models'] * 100:.1f}%\n\n")
        
        # 成功模型列表
        f.write("### ✅ 成功训练的模型\n\n")
        for model in summary['successful_models']:
            f.write(f"- {model}\n")
        f.write("\n")
        
        # 失败模型列表
        if summary['failed_models']:
            f.write("### ❌ 训练失败的模型\n\n")
            for model in summary['failed_models']:
                f.write(f"- {model}\n")
            f.write("\n")
        
        # 性能排名
        f.write("## 🏆 性能排名 (按Rel-L2排序)\n\n")
        if not ranking_df.empty:
            f.write("| 排名 | 模型 | Rel-L2 | MAE | PSNR | SSIM | 验证损失 | 训练时间(分钟) |\n")
            f.write("|------|------|--------|-----|------|------|----------|----------------|\n")
            
            for idx, row in ranking_df.iterrows():
                rank = ranking_df.index.get_loc(idx) + 1
                f.write(f"| {rank} | {row['模型']} | {row['Rel-L2']:.4f} | {row['MAE']:.4f} | "
                       f"{row['PSNR']:.2f} | {row['SSIM']:.4f} | {row['验证损失']:.4f} | {row['训练时间(分钟)']} |\n")
        f.write("\n")
        
        # 关键发现
        f.write("## 🔍 关键发现\n\n")
        if not ranking_df.empty:
            best_model = ranking_df.iloc[0]
            f.write(f"### 最佳模型: {best_model['模型']}\n\n")
            f.write(f"- **Rel-L2误差**: {best_model['Rel-L2']:.4f}\n")
            f.write(f"- **PSNR**: {best_model['PSNR']:.2f} dB\n")
            f.write(f"- **SSIM**: {best_model['SSIM']:.4f}\n")
            f.write(f"- **训练时间**: {best_model['训练时间(分钟)']} 分钟\n\n")
        
        # 训练配置
        f.write("## ⚙️ 训练配置\n\n")
        stable_config = summary['batch_training_info']['stable_config']
        f.write("### 损失函数配置\n")
        f.write(f"- rec_weight: {stable_config['loss']['rec_weight']}\n")
        f.write(f"- spec_weight: {stable_config['loss']['spec_weight']}\n")
        f.write(f"- dc_weight: {stable_config['loss']['dc_weight']}\n\n")
        
        f.write("### 训练参数\n")
        f.write(f"- batch_size: {stable_config['training']['batch_size']}\n")
        f.write(f"- learning_rate: {stable_config['training']['lr']}\n")
        f.write(f"- epochs: {stable_config['training']['epochs']}\n")
        f.write(f"- use_amp: {stable_config['training']['use_amp']}\n\n")
        
        # 文件说明
        f.write("## 📁 生成文件说明\n\n")
        f.write("- `performance_comparison.png`: 模型性能对比图表\n")
        f.write("- `model_ranking_table.png`: 模型排名表\n")
        f.write("- `model_ranking.csv`: 详细排名数据\n")
        f.write("- `batch_training_analysis_report.md`: 本分析报告\n\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量训练结果分析')
    parser.add_argument('--batch_dir', type=str, required=True, help='批量训练结果目录')
    parser.add_argument('--output_dir', type=str, help='输出目录（默认为batch_dir/analysis）')
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"错误: 批量训练目录不存在: {batch_dir}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = batch_dir / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 开始分析批量训练结果...")
    print(f"📁 批量训练目录: {batch_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    try:
        # 加载训练汇总
        summary = load_training_summary(batch_dir)
        print(f"✅ 加载训练汇总成功")
        
        # 提取每个模型的指标
        results = {}
        for model_info in summary['detailed_log']:
            model_name = model_info['model']
            print(f"🔄 分析模型: {model_name}")
            
            result = {
                'status': model_info['status'],
                'training_time': model_info['training_time'],
                'model_params': 0  # 默认值
            }
            
            if model_info['status'] == 'success':
                metrics = extract_model_metrics(batch_dir, model_name)
                result['metrics'] = metrics
            
            results[model_name] = result
        
        # 创建性能对比图
        print("🔄 创建性能对比图...")
        create_performance_comparison(results, output_dir)
        
        # 创建排名表
        print("🔄 创建排名表...")
        ranking_df = create_ranking_table(results, output_dir)
        
        # 生成分析报告
        print("🔄 生成分析报告...")
        generate_analysis_report(summary, results, ranking_df, output_dir)
        
        print(f"\n🎉 分析完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"📊 生成的文件:")
        for file_path in sorted(output_dir.glob("*")):
            print(f"  - {file_path.name}")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 导入必要的库
    import torch
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    main()