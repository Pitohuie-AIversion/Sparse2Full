#!/usr/bin/env python3
"""
训练结果可视化脚本

使用matplotlib直接创建训练曲线可视化
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """解析训练日志文件，提取损失和指标数据"""
    data = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'rel_l2': []
    }
    
    if not os.path.exists(log_path):
        print(f"警告: 日志文件不存在: {log_path}")
        return data
    
    print(f"解析日志文件: {log_path}")
    
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
    
    print(f"解析完成，找到 {len(data['epochs'])} 个训练记录")
    return data


def create_training_curves(training_data: Dict[str, List[float]], output_dir: str):
    """创建训练曲线可视化"""
    if not training_data['epochs']:
        print("没有训练数据，跳过可视化")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置matplotlib参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = training_data['epochs']
    
    # 训练损失
    axes[0, 0].plot(epochs, training_data['train_losses'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 验证损失
    axes[0, 1].plot(epochs, training_data['val_losses'], 'r-', linewidth=2, label='Val Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Rel-L2指标
    axes[1, 0].plot(epochs, training_data['rel_l2'], 'g-', linewidth=2, label='Rel-L2')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Rel-L2')
    axes[1, 0].set_title('Validation Rel-L2')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 训练和验证损失对比
    axes[1, 1].plot(epochs, training_data['train_losses'], 'b-', linewidth=2, label='Train Loss')
    axes[1, 1].plot(epochs, training_data['val_losses'], 'r-', linewidth=2, label='Val Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training vs Validation Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图像
    save_path = output_path / "training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练曲线已保存: {save_path}")
    
    # 生成统计信息
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
        
        print(f"✓ 训练统计已保存: {stats_path}")
        
        # 打印关键统计信息
        print(f"\n=== 训练统计摘要 ===")
        print(f"总轮数: {stats['total_epochs']}")
        print(f"最终训练损失: {stats['final_train_loss']:.6f}")
        print(f"最终验证损失: {stats['final_val_loss']:.6f}")
        print(f"最终Rel-L2: {stats['final_rel_l2']:.6f}")
        print(f"最佳Rel-L2: {stats['best_rel_l2']:.6f} (第{stats['best_rel_l2_epoch']}轮)")


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent
    runs_dir = project_root / "runs"
    
    print("查找可用的训练运行...")
    
    # 查找包含train.log的目录
    available_runs = []
    if runs_dir.exists():
        for run_path in runs_dir.iterdir():
            if run_path.is_dir():
                log_file = run_path / "train.log"
                if log_file.exists():
                    available_runs.append(run_path)
    
    if not available_runs:
        print("未找到任何训练运行")
        return
    
    print(f"找到 {len(available_runs)} 个训练运行:")
    for i, run_path in enumerate(available_runs):
        print(f"  {i+1}. {run_path.name}")
    
    # 选择最新的运行（按修改时间）
    latest_run = max(available_runs, key=lambda x: (x / "train.log").stat().st_mtime)
    print(f"\n选择最新的运行: {latest_run.name}")
    
    # 解析训练日志
    log_file = latest_run / "train.log"
    training_data = parse_training_log(str(log_file))
    
    if not training_data['epochs']:
        print("未找到训练数据")
        return
    
    # 创建可视化
    output_dir = latest_run / "visualizations"
    print(f"\n创建可视化，输出目录: {output_dir}")
    
    create_training_curves(training_data, str(output_dir))
    
    print(f"\n=== 可视化完成 ===")
    print(f"可视化文件保存在: {output_dir}")


if __name__ == "__main__":
    main()