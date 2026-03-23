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
    """解析训练日志，支持多种格式"""
    print(f"解析日志: {log_path}")
    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_rel_l2': [],
        'learning_rates': [],
        'curriculum_stages': [],
        'val_metrics': []
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配标准训练日志格式
            if "Epoch" in line and "Train Loss" in line and "Val Loss" in line and "Val Rel-L2" in line:
                # 解析格式: Epoch   0 - Train Loss: 241301.744141 Val Loss: 311400.562500 Val Rel-L2: 3.386388
                try:
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
            
            # 匹配AR训练日志格式
            elif "Epoch" in line and "Loss:" in line:
                # AR训练格式: Epoch 10/100 - Loss: 0.001234 - Val Loss: 0.002345 - LR: 1e-4 - T_out: 8
                try:
                    epoch_match = re.search(r'Epoch\s+(\d+)', line)
                    loss_match = re.search(r'Loss:\s+([\d.e-]+)', line)
                    val_loss_match = re.search(r'Val Loss:\s+([\d.e-]+)', line)
                    lr_match = re.search(r'LR:\s+([\d.e-]+)', line)
                    t_out_match = re.search(r'T_out:\s+(\d+)', line)
                    
                    if epoch_match and loss_match:
                        epoch = int(epoch_match.group(1))
                        train_loss = float(loss_match.group(1))
                        
                        data['epochs'].append(epoch)
                        data['train_loss'].append(train_loss)
                        
                        if val_loss_match:
                            val_loss = float(val_loss_match.group(1))
                            data['val_loss'].append(val_loss)
                        
                        if lr_match:
                            lr = float(lr_match.group(1))
                            data['learning_rates'].append(lr)
                        
                        if t_out_match:
                            t_out = int(t_out_match.group(1))
                            data['curriculum_stages'].append({'epoch': epoch, 'T_out': t_out})
                            
                except (ValueError, AttributeError) as e:
                    print(f"AR日志解析错误: {line.strip()}, 错误: {e}")
                    continue
            
            # 匹配课程学习阶段信息
            elif "课程学习" in line or "Curriculum" in line:
                try:
                    stage_match = re.search(r'阶段\s*(\d+)', line)
                    t_out_match = re.search(r'T_out[=:]\s*(\d+)', line)
                    if stage_match and t_out_match:
                        stage = int(stage_match.group(1))
                        t_out = int(t_out_match.group(1))
                        data['curriculum_stages'].append({'stage': stage, 'T_out': t_out})
                except (ValueError, AttributeError):
                    continue
    
    print(f"找到 {len(data['epochs'])} 条训练记录")
    if data['curriculum_stages']:
        print(f"找到 {len(data['curriculum_stages'])} 条课程学习记录")
    return data


def create_plots(data, output_dir):
    """创建可视化图表，支持AR训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体支持
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 尝试设置中文字体
    try:
        import matplotlib.font_manager as fm
        # 查找系统中可用的中文字体
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            if any(name in font.name.lower() for name in ['simhei', 'simsun', 'microsoft yahei', 'noto sans cjk']):
                chinese_fonts.append(font.name)
        
        if chinese_fonts:
            matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial']
            print(f"使用中文字体: {chinese_fonts[0]}")
        else:
            print("警告: 未找到中文字体，可能显示为方框")
    except Exception as e:
        print(f"字体配置警告: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    # 动态调整子图布局
    has_curriculum = len(data.get('curriculum_stages', [])) > 0
    has_lr = len(data.get('learning_rates', [])) > 0
    has_rel_l2 = len(data.get('val_rel_l2', [])) > 0
    
    # 计算需要的子图数量
    n_plots = 2  # 基础的train/val loss
    if has_lr:
        n_plots += 1
    if has_curriculum:
        n_plots += 1
    if has_rel_l2:
        n_plots += 1
    
    # 创建子图布局
    if n_plots <= 3:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 确保axes是列表
    if n_plots == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    plot_idx = 0
        
    fig.suptitle('Training Process Visualization', fontsize=16, fontweight='bold')
    
    # 1. 训练和验证损失
    axes[plot_idx].plot(data['epochs'], data['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[plot_idx].plot(data['epochs'], data['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[plot_idx].set_xlabel('Epoch')
    axes[plot_idx].set_ylabel('Loss')
    axes[plot_idx].set_title('Training and Validation Loss')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_yscale('log')
    plot_idx += 1
    
    # 2. 验证Rel-L2误差
    if has_rel_l2:
        axes[plot_idx].plot(data['epochs'], data['val_rel_l2'], 'g-', label='Val Rel-L2', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Rel-L2 Error')
        axes[plot_idx].set_title('Validation Relative L2 Error')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')
        plot_idx += 1
    
    # 3. 学习率变化
    if has_lr:
        axes[plot_idx].plot(data['epochs'], data['learning_rates'], 'purple', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')
        plot_idx += 1
    
    # 4. 课程学习进度
    if has_curriculum:
        t_out_values = [stage.get('T_out', 0) for stage in data['curriculum_stages']]
        stage_epochs = list(range(len(t_out_values)))
        axes[plot_idx].plot(stage_epochs, t_out_values, 'orange', marker='o', linewidth=2, markersize=6)
        axes[plot_idx].set_xlabel('Curriculum Stage')
        axes[plot_idx].set_ylabel('T_out')
        axes[plot_idx].set_title('Curriculum Learning Progress (T_out)')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 5. 训练统计信息（如果有额外空间）
    if plot_idx < len(axes):
        stats_text = f"""Training Statistics:
Total Epochs: {len(data['epochs'])}
Final Train Loss: {data['train_loss'][-1]:.6f}
Final Val Loss: {data['val_loss'][-1]:.6f}
Best Val Loss: {min(data['val_loss']):.6f}"""
        
        if has_rel_l2:
            stats_text += f"\nFinal Rel-L2: {data['val_rel_l2'][-1]:.6f}"
            stats_text += f"\nBest Rel-L2: {min(data['val_rel_l2']):.6f}"
        
        axes[plot_idx].text(0.1, 0.9, stats_text, transform=axes[plot_idx].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[plot_idx].set_title('Training Statistics')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {save_path}")
    
    # 统计信息
    if data['epochs']:
        stats = {
            'total_epochs': len(data['epochs']),
            'final_train_loss': data['train_loss'][-1] if data['train_loss'] else None,
            'final_val_loss': data['val_loss'][-1] if data['val_loss'] else None,
            'best_val_loss': min(data['val_loss']) if data['val_loss'] else None,
        }
        
        # 只有当val_rel_l2不为空时才添加相关统计
        if data.get('val_rel_l2') and len(data['val_rel_l2']) > 0:
            stats['final_rel_l2'] = data['val_rel_l2'][-1]
            stats['best_rel_l2'] = min(data['val_rel_l2'])
        
        stats_path = Path(output_dir) / "training_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"统计信息已保存: {stats_path}")
        if 'final_rel_l2' in stats:
            print(f"最终Rel-L2: {stats['final_rel_l2']:.6f}")
            print(f"最佳Rel-L2: {stats['best_rel_l2']:.6f}")


def parse_training_history(history_path):
    """解析training_history.json文件"""
    print(f"解析训练历史文件: {history_path}")
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 转换为viz.py期望的格式
        data = {
            'epochs': history.get('epochs', []),
            'train_loss': history.get('train_losses', []),
            'val_loss': history.get('val_losses', []),
            'val_rel_l2': [],
            'learning_rates': history.get('learning_rates', []),
            'curriculum_stages': history.get('curriculum_stages', []),
            'val_metrics': history.get('val_metrics', [])
        }
        
        # 从val_metrics中提取rel_l2
        if data['val_metrics']:
            for metrics in data['val_metrics']:
                if isinstance(metrics, dict) and 'rel_l2' in metrics:
                    data['val_rel_l2'].append(metrics['rel_l2'])
        
        print(f"成功解析训练历史: {len(data['epochs'])} 个epoch")
        return data
        
    except Exception as e:
        print(f"解析训练历史文件失败: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--log_path', type=str, help='训练日志路径')
    parser.add_argument('--history_path', type=str, help='训练历史JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='./viz_output', help='输出目录')
    
    args = parser.parse_args()
    
    data = None
    
    # 优先使用training_history.json
    if args.history_path and os.path.exists(args.history_path):
        data = parse_training_history(args.history_path)
    elif args.log_path and os.path.exists(args.log_path):
        data = parse_log(args.log_path)
    else:
        print("错误: 请提供有效的日志文件路径 (--log_path) 或训练历史文件路径 (--history_path)")
        return
    
    if not data or not data['epochs']:
        print("警告: 未找到有效的训练数据")
        return
    
    # 创建可视化
    create_plots(data, args.output_dir)
    
    print("可视化完成！")


if __name__ == "__main__":
    main()