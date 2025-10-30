#!/usr/bin/env python3
"""
训练结果可视化脚本

使用工作空间现有的可视化工具对SwinUNet训练结果进行可视化
包括训练损失曲线、验证指标变化、预测结果对比图等
"""

import os
import sys
import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from utils.visualization import PDEBenchVisualizer, create_training_curves, create_field_comparison
from tools.visualize import VisualizationTool


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """解析训练日志文件，提取损失和指标数据
    
    Args:
        log_path: 训练日志文件路径
        
    Returns:
        包含训练数据的字典
    """
    data = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'rel_l2': []
    }
    
    if not os.path.exists(log_path):
        print(f"警告: 日志文件不存在: {log_path}")
        return data
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配训练损失和验证指标的行
            # 格式: Epoch   X - Train Loss: Y Val Loss: Z Val Rel-L2: W
            match = re.search(r'Epoch\s+(\d+)\s+-\s+Train Loss:\s+([\d.]+)\s+Val Loss:\s+([\d.]+)\s+Val Rel-L2:\s+([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                rel_l2 = float(match.group(4))
                
                data['epochs'].append(epoch)
                data['train_losses'].append(train_loss)
                data['val_losses'].append(val_loss)
                data['rel_l2'].append(rel_l2)
    
    return data


def create_training_visualization(run_dir: str, output_dir: str = None) -> Dict[str, str]:
    """为指定的训练运行创建可视化
    
    Args:
        run_dir: 训练运行目录
        output_dir: 输出目录，默认为run_dir/visualizations
        
    Returns:
        生成的可视化文件路径字典
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        raise ValueError(f"训练目录不存在: {run_dir}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = run_path / "visualizations"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"为训练运行创建可视化: {run_path.name}")
    print(f"输出目录: {output_path}")
    
    # 解析训练日志
    log_file = run_path / "train.log"
    training_data = parse_training_log(str(log_file))
    
    if not training_data['epochs']:
        print("警告: 未找到训练数据，跳过训练曲线可视化")
        return {}
    
    # 创建可视化器
    visualizer = PDEBenchVisualizer(str(output_path), output_format='png')
    
    generated_files = {}
    
    # 1. 训练损失曲线
    print("生成训练损失曲线...")
    try:
        additional_metrics = {'rel_l2': training_data['rel_l2']}
        loss_curve_path = visualizer.plot_training_curves(
            training_data['train_losses'],
            training_data['val_losses'],
            save_name="training_curves",
            additional_metrics=additional_metrics
        )
        generated_files['training_curves'] = loss_curve_path
        print(f"✓ 训练曲线已保存: {loss_curve_path}")
    except Exception as e:
        print(f"✗ 生成训练曲线失败: {e}")
    
    # 2. 单独的Rel-L2指标图
    print("生成Rel-L2指标图...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(training_data['epochs'], training_data['rel_l2'], 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Rel-L2')
        plt.title('Validation Rel-L2 over Training')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        rel_l2_path = output_path / "rel_l2_curve.png"
        plt.savefig(rel_l2_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        generated_files['rel_l2_curve'] = str(rel_l2_path)
        print(f"✓ Rel-L2曲线已保存: {rel_l2_path}")
    except Exception as e:
        print(f"✗ 生成Rel-L2曲线失败: {e}")
    
    # 3. 训练统计摘要
    print("生成训练统计摘要...")
    try:
        if training_data['epochs']:
            stats = {
                'total_epochs': len(training_data['epochs']),
                'final_train_loss': training_data['train_losses'][-1],
                'final_val_loss': training_data['val_losses'][-1],
                'final_rel_l2': training_data['rel_l2'][-1],
                'best_rel_l2': min(training_data['rel_l2']),
                'best_rel_l2_epoch': training_data['epochs'][np.argmin(training_data['rel_l2'])],
                'min_train_loss': min(training_data['train_losses']),
                'min_val_loss': min(training_data['val_losses'])
            }
            
            stats_path = output_path / "training_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            generated_files['training_stats'] = str(stats_path)
            print(f"✓ 训练统计已保存: {stats_path}")
            
            # 打印关键统计信息
            print(f"\n=== 训练统计摘要 ===")
            print(f"总轮数: {stats['total_epochs']}")
            print(f"最终训练损失: {stats['final_train_loss']:.6f}")
            print(f"最终验证损失: {stats['final_val_loss']:.6f}")
            print(f"最终Rel-L2: {stats['final_rel_l2']:.6f}")
            print(f"最佳Rel-L2: {stats['best_rel_l2']:.6f} (第{stats['best_rel_l2_epoch']}轮)")
    except Exception as e:
        print(f"✗ 生成训练统计失败: {e}")
    
    # 4. 检查是否有样本数据用于预测结果可视化
    samples_dir = run_path / "samples"
    if samples_dir.exists():
        print("检查样本数据...")
        sample_epochs = list(samples_dir.glob("epoch_*"))
        if sample_epochs:
            print(f"找到 {len(sample_epochs)} 个样本epoch目录")
            # 这里可以添加样本可视化的代码
            # 由于需要加载实际的张量数据，暂时跳过
        else:
            print("未找到样本数据")
    
    return generated_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练结果可视化工具')
    parser.add_argument('--run_dir', type=str, help='训练运行目录路径')
    parser.add_argument('--output_dir', type=str, help='输出目录路径（可选）')
    parser.add_argument('--list_runs', action='store_true', help='列出所有可用的训练运行')
    
    args = parser.parse_args()
    
    # 项目根目录
    project_root = Path(__file__).parent
    runs_dir = project_root / "runs"
    
    if args.list_runs:
        print("可用的训练运行:")
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if run_path.is_dir() and (run_path / "train.log").exists():
                    print(f"  - {run_path.name}")
        else:
            print("  未找到runs目录")
        return
    
    if args.run_dir:
        run_dir = args.run_dir
    else:
        # 自动查找最近的SwinUNet训练
        swin_runs = []
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if run_path.is_dir() and "swin" in run_path.name.lower():
                    log_file = run_path / "train.log"
                    if log_file.exists():
                        swin_runs.append(run_path)
        
        if not swin_runs:
            print("未找到SwinUNet训练运行，请使用--run_dir指定目录")
            return
        
        # 选择最新的运行
        run_dir = str(max(swin_runs, key=lambda x: x.stat().st_mtime))
        print(f"自动选择最新的SwinUNet运行: {Path(run_dir).name}")
    
    try:
        generated_files = create_training_visualization(run_dir, args.output_dir)
        
        print(f"\n=== 可视化完成 ===")
        print(f"生成了 {len(generated_files)} 个可视化文件:")
        for name, path in generated_files.items():
            print(f"  - {name}: {path}")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()