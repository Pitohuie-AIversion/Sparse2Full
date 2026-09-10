#!/usr/bin/env python3
"""
提取批量训练模型的性能指标

从训练日志中提取每个模型的最终性能指标
"""

import os
import sys
import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def extract_best_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """从训练日志中提取最佳指标"""
    metrics = {
        'best_val_loss': None,
        'rel_l2': None,
        'mae': None,
        'psnr': None,
        'ssim': None,
        'frmse_low': None,
        'frmse_mid': None,
        'frmse_high': None,
        'brmse': None,
        'crmse': None
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最佳验证损失
        best_loss_match = re.search(r'Best validation loss: ([\d.]+)', content)
        if best_loss_match:
            metrics['best_val_loss'] = float(best_loss_match.group(1))
        
        # 查找最佳验证指标部分 - 处理多行格式
        best_metrics_pattern = r'Best validation metrics: \{(.*?)\}'
        best_metrics_match = re.search(best_metrics_pattern, content, re.DOTALL)
        if best_metrics_match:
            metrics_text = best_metrics_match.group(1)
            
            # 提取各种指标的平均值
            def extract_tensor_mean(pattern, text):
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    tensor_content = match.group(1)
                    # 提取数值，包括科学计数法
                    numbers = re.findall(r'[\d.]+(?:e[+-]?\d+)?', tensor_content)
                    if numbers:
                        values = [float(n) for n in numbers]
                        return sum(values) / len(values)  # 平均值
                return None
            
            # 更新正则表达式以匹配多行tensor格式
            metrics['rel_l2'] = extract_tensor_mean(r"'rel_l2': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['mae'] = extract_tensor_mean(r"'mae': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['psnr'] = extract_tensor_mean(r"'psnr': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['ssim'] = extract_tensor_mean(r"'ssim': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['frmse_low'] = extract_tensor_mean(r"'frmse_low': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['frmse_mid'] = extract_tensor_mean(r"'frmse_mid': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['frmse_high'] = extract_tensor_mean(r"'frmse_high': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['brmse'] = extract_tensor_mean(r"'brmse': tensor\(\[\[(.*?)\]\]", metrics_text)
            metrics['crmse'] = extract_tensor_mean(r"'crmse': tensor\(\[\[(.*?)\]\]", metrics_text)
    
    except Exception as e:
        print(f"解析训练日志失败 {log_path}: {e}")
    
    return metrics

def get_model_params_from_checkpoint(checkpoint_path: Path) -> int:
    """从检查点获取模型参数数量"""
    try:
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_params' in checkpoint:
                return checkpoint['model_params']
            elif 'model' in checkpoint:
                # 计算参数数量
                model_state = checkpoint['model']
                total_params = sum(p.numel() for p in model_state.values())
                return total_params
    except Exception as e:
        print(f"读取检查点失败 {checkpoint_path}: {e}")
    
    return 0

def analyze_batch_results(batch_dir: str) -> Dict[str, Any]:
    """分析批量训练结果"""
    batch_path = Path(batch_dir)
    
    # 加载训练汇总
    summary_path = batch_path / "training_summary.json"
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    results = {}
    
    # 分析每个成功的模型
    for model_info in summary['detailed_log']:
        model_name = model_info['model']
        
        if model_info['status'] == 'success':
            print(f"🔄 分析模型: {model_name}")
            
            # 模型目录
            model_dir = batch_path / model_name.lower()
            
            # 提取指标
            log_path = model_dir / "train.log"
            metrics = extract_best_metrics_from_log(log_path)
            
            # 获取模型参数数量
            checkpoint_path = model_dir / "checkpoints" / "best.pth"
            model_params = get_model_params_from_checkpoint(checkpoint_path)
            
            results[model_name] = {
                'status': 'success',
                'training_time': model_info['training_time'],
                'metrics': metrics,
                'model_params': model_params,
                'model_dir': str(model_dir)
            }
        else:
            results[model_name] = {
                'status': 'failed',
                'training_time': model_info['training_time'],
                'return_code': model_info.get('return_code', -1)
            }
    
    return {
        'summary': summary,
        'results': results
    }

def create_performance_comparison(analysis_data: Dict[str, Any], output_dir: Path):
    """创建性能对比图表"""
    results = analysis_data['results']
    
    # 准备数据
    models = []
    rel_l2_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []
    val_loss_values = []
    training_times = []
    param_counts = []
    
    for model_name, data in results.items():
        if data['status'] == 'success' and data.get('metrics'):
            metrics = data['metrics']
            if metrics['rel_l2'] is not None:  # 确保有有效指标
                models.append(model_name)
                rel_l2_values.append(metrics['rel_l2'])
                mae_values.append(metrics['mae'] or 0)
                psnr_values.append(metrics['psnr'] or 0)
                ssim_values.append(metrics['ssim'] or 0)
                val_loss_values.append(metrics['best_val_loss'] or 0)
                training_times.append(data['training_time'] / 60)  # 转换为分钟
                param_counts.append(data['model_params'] / 1e6)  # 转换为百万参数
    
    if not models:
        print("警告: 没有有效的模型数据用于可视化")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('批量训练模型性能对比分析', fontsize=16, fontweight='bold')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
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
    axes[0, 3].bar(models, ssim_values, color='pink', alpha=0.7)
    axes[0, 3].set_title('结构相似性 (越大越好)')
    axes[0, 3].set_ylabel('SSIM')
    axes[0, 3].tick_params(axis='x', rotation=45)
    
    # 验证损失对比
    axes[1, 0].bar(models, val_loss_values, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('最佳验证损失 (越小越好)')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 训练时间对比
    axes[1, 1].bar(models, training_times, color='gold', alpha=0.7)
    axes[1, 1].set_title('训练时间 (分钟)')
    axes[1, 1].set_ylabel('时间 (分钟)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 参数数量对比
    axes[1, 2].bar(models, param_counts, color='lightblue', alpha=0.7)
    axes[1, 2].set_title('模型参数数量 (百万)')
    axes[1, 2].set_ylabel('参数数量 (M)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 效率对比 (PSNR/训练时间)
    efficiency = [p/t if t > 0 else 0 for p, t in zip(psnr_values, training_times)]
    axes[1, 3].bar(models, efficiency, color='lightsteelblue', alpha=0.7)
    axes[1, 3].set_title('训练效率 (PSNR/分钟)')
    axes[1, 3].set_ylabel('PSNR/分钟')
    axes[1, 3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 性能对比图保存到: {output_dir / 'performance_comparison.png'}")

def create_ranking_table(analysis_data: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """创建模型排名表"""
    results = analysis_data['results']
    
    # 准备数据
    data = []
    for model_name, result in results.items():
        if result['status'] == 'success' and result.get('metrics'):
            metrics = result['metrics']
            if metrics['rel_l2'] is not None:  # 确保有有效指标
                data.append({
                    '模型': model_name,
                    'Rel-L2': metrics['rel_l2'],
                    'MAE': metrics['mae'] or 0,
                    'PSNR': metrics['psnr'] or 0,
                    'SSIM': metrics['ssim'] or 0,
                    '最佳验证损失': metrics['best_val_loss'] or 0,
                    '训练时间(分钟)': round(result['training_time'] / 60, 2),
                    '参数量(M)': round(result['model_params'] / 1e6, 2),
                    'BRMSE': metrics['brmse'] or 0,
                    'CRMSE': metrics['crmse'] or 0
                })
    
    if not data:
        print("警告: 没有成功的模型数据用于排名")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # 按Rel-L2排序（越小越好）
    df_sorted = df.sort_values('Rel-L2')
    
    # 保存为CSV
    df_sorted.to_csv(output_dir / 'model_ranking.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 排名表保存到: {output_dir / 'model_ranking.csv'}")
    return df_sorted

def generate_analysis_report(analysis_data: Dict[str, Any], ranking_df: pd.DataFrame, output_dir: Path):
    """生成分析报告"""
    summary = analysis_data['summary']
    results = analysis_data['results']
    
    report_path = output_dir / 'batch_training_analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 批量训练结果分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 训练概览
        f.write("## 📊 训练概览\n\n")
        batch_info = summary['batch_training_info']
        f.write(f"- **总模型数**: {batch_info['total_models']}\n")
        f.write(f"- **成功训练**: {batch_info['successful_models']}\n")
        f.write(f"- **训练失败**: {batch_info['failed_models']}\n")
        f.write(f"- **成功率**: {batch_info['successful_models'] / batch_info['total_models'] * 100:.1f}%\n\n")
        
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
            f.write("| 排名 | 模型 | Rel-L2 | MAE | PSNR | SSIM | 最佳验证损失 | 训练时间(分钟) | 参数量(M) |\n")
            f.write("|------|------|--------|-----|------|------|-------------|----------------|----------|\n")
            
            for idx, (_, row) in enumerate(ranking_df.iterrows()):
                rank = idx + 1
                f.write(f"| {rank} | {row['模型']} | {row['Rel-L2']:.4f} | {row['MAE']:.4f} | "
                       f"{row['PSNR']:.2f} | {row['SSIM']:.4f} | {row['最佳验证损失']:.4f} | "
                       f"{row['训练时间(分钟)']} | {row['参数量(M)']} |\n")
        f.write("\n")
        
        # 关键发现
        f.write("## 🔍 关键发现\n\n")
        if not ranking_df.empty:
            best_model = ranking_df.iloc[0]
            f.write(f"### 🥇 最佳模型: {best_model['模型']}\n\n")
            f.write(f"- **Rel-L2误差**: {best_model['Rel-L2']:.4f}\n")
            f.write(f"- **PSNR**: {best_model['PSNR']:.2f} dB\n")
            f.write(f"- **SSIM**: {best_model['SSIM']:.4f}\n")
            f.write(f"- **训练时间**: {best_model['训练时间(分钟)']} 分钟\n")
            f.write(f"- **参数量**: {best_model['参数量(M)']} M\n\n")
            
            # 性能分析
            f.write("### 📈 性能分析\n\n")
            f.write("**Top 3 模型对比:**\n\n")
            for idx, (_, row) in enumerate(ranking_df.head(3).iterrows()):
                rank = idx + 1
                f.write(f"{rank}. **{row['模型']}**: Rel-L2={row['Rel-L2']:.4f}, "
                       f"PSNR={row['PSNR']:.2f}dB, 训练时间={row['训练时间(分钟)']}分钟\n")
            f.write("\n")
        
        # 训练配置
        f.write("## ⚙️ 训练配置\n\n")
        stable_config = batch_info['stable_config']
        f.write("### 损失函数配置 (稳定配置)\n")
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
        f.write("- `model_ranking.csv`: 详细排名数据\n")
        f.write("- `batch_training_analysis_report.md`: 本分析报告\n")
        f.write("- `analysis_results.json`: 完整分析数据\n\n")
    
    print(f"✅ 分析报告保存到: {report_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='提取批量训练模型的性能指标')
    parser.add_argument('--batch_dir', type=str, required=True, help='批量训练结果目录')
    parser.add_argument('--output_dir', type=str, help='输出目录（默认为batch_dir/analysis）')
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"❌ 错误: 批量训练目录不存在: {batch_dir}")
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
        # 分析批量训练结果
        print("🔄 提取模型指标...")
        analysis_data = analyze_batch_results(batch_dir)
        
        # 保存完整分析数据
        with open(output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            # 转换不可序列化的对象
            serializable_data = {}
            for key, value in analysis_data.items():
                if key == 'results':
                    serializable_results = {}
                    for model_name, model_data in value.items():
                        serializable_results[model_name] = {
                            k: (v if not isinstance(v, torch.Tensor) else v.tolist() if hasattr(v, 'tolist') else str(v))
                            for k, v in model_data.items()
                        }
                    serializable_data[key] = serializable_results
                else:
                    serializable_data[key] = value
            
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # 创建性能对比图
        print("🔄 创建性能对比图...")
        create_performance_comparison(analysis_data, output_dir)
        
        # 创建排名表
        print("🔄 创建排名表...")
        ranking_df = create_ranking_table(analysis_data, output_dir)
        
        # 生成分析报告
        print("🔄 生成分析报告...")
        generate_analysis_report(analysis_data, ranking_df, output_dir)
        
        print(f"\n🎉 分析完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"📊 生成的文件:")
        for file_path in sorted(output_dir.glob("*")):
            print(f"  - {file_path.name}")
        
        # 显示排名摘要
        if not ranking_df.empty:
            print(f"\n🏆 模型性能排名 (Top 3):")
            for idx, (_, row) in enumerate(ranking_df.head(3).iterrows()):
                rank = idx + 1
                print(f"  {rank}. {row['模型']}: Rel-L2={row['Rel-L2']:.4f}, PSNR={row['PSNR']:.2f}dB")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()