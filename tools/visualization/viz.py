#!/usr/bin/env python3
"""
简单的训练结果可视化脚本
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path):
    print(f"解析日志: {log_path}")
    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_rel_l2': []
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配训练日志格式
            if "Epoch" in line and "Train Loss" in line and "Val Loss" in line and "Val Rel-L2" in line:
                # 解析格式: Epoch   0 - Train Loss: 241301.744141 Val Loss: 311400.562500 Val Rel-L2: 3.386388
                try:
                    # 使用正则表达式提取数值
                    import re
                    epoch_match = re.search(r'Epoch\s+(\d+)', line)
                    train_loss_match = re.search(r'Train Loss:\s+([\d.]+)', line)
                    val_loss_match = re.search(r'Val Loss:\s+([\d.]+)', line)
                    rel_l2_match = re.search(r'Val Rel-L2:\s+([\d.]+)', line)
                    
                    if all([epoch_match, train_loss_match, val_loss_match, rel_l2_match]):
                        epoch = int(epoch_match.group(1))
                        train_loss = float(train_loss_match.group(1))
                        val_loss = float(val_loss_match.group(1))
                        val_rel_l2 = float(rel_l2_match.group(1))
                        
                        data['epochs'].append(epoch)
                        data['train_loss'].append(train_loss)
                        data['val_loss'].append(val_loss)
                        data['val_rel_l2'].append(val_rel_l2)
                except (ValueError, AttributeError) as e:
                    print(f"解析错误: {line.strip()}, 错误: {e}")
                    continue
    
    print(f"找到 {len(data['epochs'])} 条记录")
    return data


def create_plots(data, output_dir):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    axes[0, 0].plot(data['epochs'], data['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 验证损失
    axes[0, 1].plot(data['epochs'], data['val_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rel-L2
    axes[1, 0].plot(data['epochs'], data['val_rel_l2'], 'g-', linewidth=2)
    axes[1, 0].set_title('Rel-L2')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Rel-L2')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 对比
    axes[1, 1].plot(data['epochs'], data['train_loss'], 'b-', linewidth=2, label='Train')
    axes[1, 1].plot(data['epochs'], data['val_loss'], 'r-', linewidth=2, label='Val')
    axes[1, 1].set_title('Loss Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {save_path}")
    
    # 统计信息
    if data['epochs']:
        stats = {
            'total_epochs': len(data['epochs']),
            'final_train_loss': data['train_loss'][-1],
            'final_val_loss': data['val_loss'][-1],
            'final_rel_l2': data['val_rel_l2'][-1],
            'best_rel_l2': min(data['val_rel_l2']),
        }
        
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"统计信息已保存: {stats_path}")
        print(f"最终Rel-L2: {stats['final_rel_l2']:.6f}")
        print(f"最佳Rel-L2: {stats['best_rel_l2']:.6f}")


def main():
    # 指定使用MLP训练运行
    target_run = Path("runs/SRx4-DarcyFlow-128-MLP-quick-s2025-20250111")
    
    if not target_run.exists():
        print(f"目标运行不存在: {target_run}")
        return
    
    print(f"使用运行: {target_run.name}")
    
    # 解析日志
    log_file = target_run / "train.log"
    data = parse_log(str(log_file))
    
    if not data['epochs']:
        print("未找到训练数据")
        return
    
    print(f"找到 {len(data['epochs'])} 个训练记录")
    
    # 创建可视化
    output_dir = target_run / "visualizations"
    create_plots(data, str(output_dir))


if __name__ == "__main__":
    main()