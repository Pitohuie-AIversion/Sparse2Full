#!/usr/bin/env python3
"""
时序NAR训练结果可视化脚本
生成专业的训练可视化报告，包括训练曲线、收敛分析等
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置matplotlib后端和中文字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(json_path):
    """加载训练历史数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_training_curves(data, save_dir):
    """创建训练损失曲线图"""
    train_losses = data['train_losses']
    val_losses = data.get('val_losses', [])
    epochs = range(1, len(train_losses) + 1)
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal NAR Model Training Visualization Report', fontsize=20, fontweight='bold')
    
    # 1. 训练损失曲线
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    if val_losses:
        ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # 添加平滑曲线
    if len(train_losses) > 10:
        # 简单移动平均代替高斯滤波
        window = 5
        smooth_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(train_losses)+1), smooth_train, 'b--', linewidth=1.5, alpha=0.6, label='Training Loss (Smoothed)')
    
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(train_losses))
    
    # 2. 损失分布直方图
    ax2.hist(train_losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(train_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(train_losses):.4f}')
    ax2.axvline(np.median(train_losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(train_losses):.4f}')
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Training Loss Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收敛分析 - 移动平均
    window_sizes = [5, 10, 20]
    colors = ['orange', 'green', 'purple']
    
    for window, color in zip(window_sizes, colors):
        if len(train_losses) >= window:
            moving_avg = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            ax3.plot(range(window, len(train_losses)+1), moving_avg, 
                    color=color, linewidth=2, label=f'{window}-epoch Moving Average')
    
    ax3.plot(epochs, train_losses, 'lightgray', alpha=0.5, label='Original Loss')
    ax3.set_xlabel('Training Epochs')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Convergence Trend Analysis (Moving Average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练稳定性分析
    if len(train_losses) > 20:
        # 计算损失变化率
        loss_changes = np.diff(train_losses)
        ax4.plot(epochs[1:], loss_changes, 'purple', alpha=0.6, label='Loss Change Rate')
        ax4.axhline(0, color='red', linestyle='--', alpha=0.8)
        
        # 添加趋势线
        z = np.polyfit(epochs[1:], loss_changes, 1)
        p = np.poly1d(z)
        ax4.plot(epochs[1:], p(epochs[1:]), "r--", alpha=0.8, label=f'Trend Line (slope: {z[0]:.6f})')
        
        ax4.set_xlabel('Training Epochs')
        ax4.set_ylabel('Loss Change Rate')
        ax4.set_title('Training Stability Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / 'training_curves_comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Training curves saved: {save_path}")
    
    return fig

def create_convergence_analysis(data, save_dir):
    """创建详细的收敛分析图"""
    train_losses = data['train_losses']
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Convergence Deep Analysis', fontsize=20, fontweight='bold')
    
    # 1. 对数尺度损失曲线
    ax1.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss (Log Scale)')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss Value (Log Scale)')
    ax1.set_title('Log Scale Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失改善率
    if len(train_losses) > 1:
        improvement_rate = [(train_losses[i-1] - train_losses[i]) / train_losses[i-1] * 100 
                           for i in range(1, len(train_losses))]
        ax2.plot(epochs[1:], improvement_rate, 'g-', linewidth=2, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel('Loss Improvement Rate (%)')
        ax2.set_title('Per-Epoch Loss Improvement Rate')
        ax2.grid(True, alpha=0.3)
    
    # 3. 滚动标准差 (训练稳定性)
    window = min(10, len(train_losses) // 4)
    if window >= 3:
        rolling_std = []
        for i in range(window, len(train_losses) + 1):
            rolling_std.append(np.std(train_losses[i-window:i]))
        
        ax3.plot(range(window, len(train_losses) + 1), rolling_std, 'orange', linewidth=2)
        ax3.set_xlabel('Training Epochs')
        ax3.set_ylabel('Rolling Standard Deviation')
        ax3.set_title(f'Training Stability Analysis (Window Size: {window})')
        ax3.grid(True, alpha=0.3)
    
    # 4. 收敛速度分析
    if len(train_losses) > 10:
        # 计算到最小值的距离
        min_loss = min(train_losses)
        distance_to_min = [abs(loss - min_loss) for loss in train_losses]
        
        ax4.semilogy(epochs, distance_to_min, 'purple', linewidth=2, label='Distance to Minimum Loss')
        ax4.set_xlabel('Training Epochs')
        ax4.set_ylabel('Distance to Min Loss (Log Scale)')
        ax4.set_title('Convergence Speed Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / 'convergence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Convergence analysis saved: {save_path}")
    
    return fig

def generate_training_report(data, save_dir):
    """生成训练报告摘要"""
    train_losses = data['train_losses']
    val_losses = data.get('val_losses', [])
    
    # 计算统计信息
    stats_info = {
        'Total Training Epochs': len(train_losses),
        'Final Training Loss': train_losses[-1],
        'Best Training Loss': min(train_losses),
        'Average Training Loss': np.mean(train_losses),
        'Training Loss Std': np.std(train_losses),
        'Loss Improvement': train_losses[0] - train_losses[-1],
        'Improvement Percentage': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    }
    
    if val_losses:
        stats_info.update({
            'Best Validation Loss': min(val_losses),
            'Final Validation Loss': val_losses[-1]
        })
    
    # 创建报告图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 标题
    fig.suptitle('🎯 Temporal NAR Training Report Summary', fontsize=24, fontweight='bold', y=0.95)
    
    # 创建表格
    table_data = []
    for key, value in stats_info.items():
        if isinstance(value, float):
            table_data.append([key, f'{value:.6f}'])
        else:
            table_data.append([key, str(value)])
    
    # 添加表格
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.4, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    
    # 美化表格
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            cell.set_edgecolor('gray')
            cell.set_linewidth(1)
    
    # 添加训练状态评估
    if train_losses[-1] < train_losses[0] * 0.9:
        status = "✅ Training Successfully Converged"
        color = 'green'
    elif train_losses[-1] < train_losses[0]:
        status = "⚠️ Training Improved but May Need More Epochs"
        color = 'orange'
    else:
        status = "❌ Training May Have Issues"
        color = 'red'
    
    ax.text(0.5, 0.15, status, transform=ax.transAxes, 
            fontsize=18, fontweight='bold', ha='center', color=color)
    
    # 保存报告
    save_path = save_dir / 'training_report_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Training report summary saved: {save_path}")
    
    return fig

def main():
    """主函数"""
    # 设置路径
    base_dir = Path("f:/Zhaoyang/Sparse2Full")
    data_path = base_dir / "runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/training_history.json"
    save_dir = base_dir / "runs/temporal_nar_100epochs/visualizations"
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载数据
        print("📊 Loading training data...")
        data = load_training_data(data_path)
        print(f"✅ Successfully loaded {len(data['train_losses'])} epochs of training data")
        
        # 生成可视化
        print("\n🎨 Generating training curves...")
        fig1 = create_training_curves(data, save_dir)
        plt.close(fig1)  # 关闭图形以释放内存
        
        print("\n📈 Generating convergence analysis...")
        fig2 = create_convergence_analysis(data, save_dir)
        plt.close(fig2)  # 关闭图形以释放内存
        
        print("\n📋 Generating training report...")
        fig3 = generate_training_report(data, save_dir)
        plt.close(fig3)  # 关闭图形以释放内存
        
        print(f"\n🎉 All visualization charts have been generated successfully!")
        print(f"📁 Save location: {save_dir}")
        print("\nGenerated files:")
        print("  - training_curves_comprehensive.png (Comprehensive Training Curves)")
        print("  - convergence_analysis.png (Convergence Analysis)")
        print("  - training_report_summary.png (Training Report Summary)")
        
    except FileNotFoundError:
        print(f"❌ Error: Training data file not found {data_path}")
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()