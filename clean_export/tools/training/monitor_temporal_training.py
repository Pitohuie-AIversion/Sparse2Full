#!/usr/bin/env python3
"""
时序NAR训练监控脚本
实时监控300轮训练的进度和效果
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def monitor_training():
    """监控训练进度"""
    
    print("=" * 60)
    print("🚀 时序NAR模型300轮训练监控")
    print("=" * 60)
    
    # 训练历史文件路径
    history_file = Path("runs/temporal_nar_300epochs/TemporalNAR-DR2D-128-300epochs-s2025/training_history.json")
    
    if not history_file.exists():
        print("❌ 训练历史文件不存在，请确保训练已开始")
        return
    
    # 读取训练历史
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    
    current_epoch = len(train_losses)
    
    print(f"📊 当前训练进度: {current_epoch}/300 轮")
    
    if current_epoch > 0:
        print(f"📈 最新训练损失: {train_losses[-1]:.6f}")
        if val_losses:
            print(f"📉 最新验证损失: {val_losses[-1]:.6f}")
        
        # 计算收敛趋势
        if current_epoch >= 10:
            recent_train = np.mean(train_losses[-10:])
            early_train = np.mean(train_losses[:10])
            improvement = (early_train - recent_train) / early_train * 100
            print(f"🎯 损失改善: {improvement:.2f}%")
        
        # 生成训练曲线图
        create_training_plot(train_losses, val_losses, current_epoch)
        
        # 分析训练状态
        analyze_training_status(train_losses, val_losses)
    
    print("=" * 60)

def create_training_plot(train_losses, val_losses, current_epoch):
    """创建训练曲线图"""
    
    plt.figure(figsize=(12, 8))
    
    # 训练损失曲线
    plt.subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.7)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.7)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title(f'训练曲线 (当前: {current_epoch}/300)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失对数图
    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, train_losses, 'b-', label='训练损失', alpha=0.7)
    if val_losses:
        plt.semilogy(epochs, val_losses, 'r-', label='验证损失', alpha=0.7)
    plt.xlabel('轮次')
    plt.ylabel('损失 (对数)')
    plt.title('损失对数图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最近50轮的详细图
    if len(train_losses) > 50:
        plt.subplot(2, 2, 3)
        recent_epochs = epochs[-50:]
        recent_train = train_losses[-50:]
        plt.plot(recent_epochs, recent_train, 'b-', label='训练损失', alpha=0.7)
        if len(val_losses) >= 50:
            recent_val = val_losses[-50:]
            plt.plot(recent_epochs, recent_val, 'r-', label='验证损失', alpha=0.7)
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('最近50轮详细图')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 收敛速度分析
    plt.subplot(2, 2, 4)
    if len(train_losses) > 10:
        # 计算滑动平均
        window = min(10, len(train_losses) // 4)
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = range(window, len(train_losses) + 1)
        plt.plot(smooth_epochs, smoothed, 'g-', label=f'{window}轮滑动平均', linewidth=2)
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('收敛趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("runs/temporal_nar_300epochs/TemporalNAR-DR2D-128-300epochs-s2025/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / f"training_progress_epoch_{current_epoch:03d}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练曲线已保存: {output_dir}/training_progress_epoch_{current_epoch:03d}.png")

def analyze_training_status(train_losses, val_losses):
    """分析训练状态"""
    
    print("\n📋 训练状态分析:")
    
    if len(train_losses) < 5:
        print("   ⏳ 训练刚开始，数据不足以分析")
        return
    
    # 检查是否收敛
    recent_variance = np.var(train_losses[-10:]) if len(train_losses) >= 10 else np.var(train_losses)
    
    if recent_variance < 1e-6:
        print("   ✅ 训练已收敛")
    elif recent_variance < 1e-4:
        print("   🎯 训练接近收敛")
    else:
        print("   📈 训练仍在进行中")
    
    # 检查过拟合
    if val_losses and len(val_losses) >= 10:
        recent_train_trend = np.mean(train_losses[-5:]) - np.mean(train_losses[-10:-5])
        recent_val_trend = np.mean(val_losses[-5:]) - np.mean(val_losses[-10:-5])
        
        if recent_train_trend < 0 and recent_val_trend > 0:
            print("   ⚠️  可能存在过拟合")
        else:
            print("   ✅ 训练健康")
    
    # 学习率建议
    if len(train_losses) >= 20:
        early_loss = np.mean(train_losses[:10])
        recent_loss = np.mean(train_losses[-10:])
        improvement_rate = (early_loss - recent_loss) / len(train_losses)
        
        if improvement_rate < 1e-5:
            print("   💡 建议: 可能需要调整学习率")
        else:
            print("   ✅ 学习率合适")

if __name__ == "__main__":
    monitor_training()