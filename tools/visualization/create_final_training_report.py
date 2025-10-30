#!/usr/bin/env python3
"""
生成300轮时序NAR训练的最终报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def create_final_report():
    """生成最终训练报告"""
    
    print("=" * 60)
    print("📊 生成300轮时序NAR训练最终报告")
    print("=" * 60)
    
    # 读取训练历史
    history_file = Path("runs/temporal_nar_300epochs/TemporalNAR-DR2D-128-300epochs-s2025/training_history.json")
    
    if not history_file.exists():
        print("❌ 训练历史文件不存在")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    train_losses = history.get('train_losses', [])
    val_losses = history.get('val_losses', [])
    best_val_loss = history.get('best_val_loss', None)
    
    total_epochs = len(train_losses)
    
    print(f"✅ 训练完成情况: {total_epochs}/300 轮")
    print(f"📈 最终训练损失: {train_losses[-1]:.8f}")
    if val_losses:
        print(f"📉 最终验证损失: {val_losses[-1]:.8f}")
    if best_val_loss:
        print(f"🏆 最佳验证损失: {best_val_loss:.8f}")
    
    # 计算训练统计
    calculate_training_stats(train_losses, val_losses)
    
    # 生成完整的训练报告图
    create_comprehensive_plot(train_losses, val_losses, total_epochs)
    
    # 生成收敛分析
    analyze_convergence(train_losses, val_losses)
    
    print("=" * 60)

def calculate_training_stats(train_losses, val_losses):
    """计算训练统计信息"""
    
    print("\n📊 训练统计信息:")
    
    # 基本统计
    print(f"   初始训练损失: {train_losses[0]:.6f}")
    print(f"   最终训练损失: {train_losses[-1]:.6f}")
    print(f"   总体改善: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.2f}%")
    
    # 收敛分析
    if len(train_losses) >= 50:
        # 最后50轮的方差
        final_variance = np.var(train_losses[-50:])
        print(f"   最后50轮方差: {final_variance:.2e}")
        
        # 收敛速度
        mid_point = len(train_losses) // 2
        first_half_avg = np.mean(train_losses[:mid_point])
        second_half_avg = np.mean(train_losses[mid_point:])
        print(f"   前半段平均损失: {first_half_avg:.6f}")
        print(f"   后半段平均损失: {second_half_avg:.6f}")
        
        # 学习效率
        learning_efficiency = (first_half_avg - second_half_avg) / len(train_losses)
        print(f"   学习效率: {learning_efficiency:.2e} 每轮")
    
    # 验证损失统计
    if val_losses:
        print(f"   初始验证损失: {val_losses[0]:.6f}")
        print(f"   最终验证损失: {val_losses[-1]:.6f}")
        print(f"   验证改善: {(val_losses[0] - val_losses[-1]) / val_losses[0] * 100:.2f}%")

def create_comprehensive_plot(train_losses, val_losses, total_epochs):
    """创建综合训练报告图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 1. 完整训练曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.8, linewidth=1.5)
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title(f'完整训练曲线 ({total_epochs} 轮)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 对数损失图
    axes[0, 1].semilogy(epochs, train_losses, 'b-', label='训练损失', alpha=0.8)
    if val_losses:
        axes[0, 1].semilogy(epochs, val_losses, 'r-', label='验证损失', alpha=0.8)
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('损失 (对数)')
    axes[0, 1].set_title('对数损失图')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 损失改善率
    if len(train_losses) > 10:
        improvement_rate = []
        window = 10
        for i in range(window, len(train_losses)):
            old_avg = np.mean(train_losses[i-window:i])
            new_avg = np.mean(train_losses[i-window//2:i])
            rate = (old_avg - new_avg) / old_avg * 100
            improvement_rate.append(rate)
        
        axes[0, 2].plot(range(window+1, len(train_losses)+1), improvement_rate, 'g-', alpha=0.8)
        axes[0, 2].set_xlabel('轮次')
        axes[0, 2].set_ylabel('改善率 (%)')
        axes[0, 2].set_title('损失改善率')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. 滑动平均
    window_sizes = [5, 10, 20]
    colors = ['orange', 'green', 'purple']
    for window, color in zip(window_sizes, colors):
        if len(train_losses) > window:
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            smooth_epochs = range(window, len(train_losses) + 1)
            axes[1, 0].plot(smooth_epochs, smoothed, color=color, 
                          label=f'{window}轮滑动平均', alpha=0.8, linewidth=2)
    
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].set_title('滑动平均趋势')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 最后100轮详细图
    if len(train_losses) > 100:
        recent_epochs = epochs[-100:]
        recent_train = train_losses[-100:]
        axes[1, 1].plot(recent_epochs, recent_train, 'b-', label='训练损失', alpha=0.8)
        if len(val_losses) >= 100:
            recent_val = val_losses[-100:]
            axes[1, 1].plot(recent_epochs, recent_val, 'r-', label='验证损失', alpha=0.8)
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].set_title('最后100轮详细图')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 收敛分析
    if len(train_losses) > 20:
        # 计算局部方差
        variance_window = 20
        variances = []
        for i in range(variance_window, len(train_losses)):
            var = np.var(train_losses[i-variance_window:i])
            variances.append(var)
        
        var_epochs = range(variance_window+1, len(train_losses)+1)
        axes[1, 2].semilogy(var_epochs, variances, 'purple', alpha=0.8)
        axes[1, 2].set_xlabel('轮次')
        axes[1, 2].set_ylabel('局部方差 (对数)')
        axes[1, 2].set_title('收敛稳定性分析')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("runs/temporal_nar_300epochs/TemporalNAR-DR2D-128-300epochs-s2025/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "final_training_report.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 最终训练报告已保存: {output_dir}/final_training_report.png")

def analyze_convergence(train_losses, val_losses):
    """分析收敛情况"""
    
    print("\n🎯 收敛分析:")
    
    if len(train_losses) < 20:
        print("   ⏳ 训练轮次不足，无法进行收敛分析")
        return
    
    # 检查收敛状态
    final_variance = np.var(train_losses[-20:])
    
    if final_variance < 1e-8:
        convergence_status = "完全收敛"
        status_emoji = "✅"
    elif final_variance < 1e-6:
        convergence_status = "高度收敛"
        status_emoji = "🎯"
    elif final_variance < 1e-4:
        convergence_status = "基本收敛"
        status_emoji = "📈"
    else:
        convergence_status = "仍在收敛"
        status_emoji = "⏳"
    
    print(f"   {status_emoji} 收敛状态: {convergence_status}")
    print(f"   📊 最后20轮方差: {final_variance:.2e}")
    
    # 估计收敛轮次
    threshold = 1e-6
    convergence_epoch = None
    
    for i in range(20, len(train_losses)):
        if np.var(train_losses[i-20:i]) < threshold:
            convergence_epoch = i
            break
    
    if convergence_epoch:
        print(f"   🏁 估计收敛轮次: 第{convergence_epoch}轮")
        remaining_epochs = len(train_losses) - convergence_epoch
        print(f"   ⏰ 收敛后继续训练: {remaining_epochs}轮")
    else:
        print("   📈 训练期间未达到收敛阈值")
    
    # 过拟合检查
    if val_losses and len(val_losses) >= 20:
        train_trend = np.polyfit(range(len(train_losses[-20:])), train_losses[-20:], 1)[0]
        val_trend = np.polyfit(range(len(val_losses[-20:])), val_losses[-20:], 1)[0]
        
        if train_trend < 0 and val_trend > 0 and abs(val_trend) > abs(train_trend):
            print("   ⚠️  检测到过拟合迹象")
        else:
            print("   ✅ 无明显过拟合")
    
    # 训练效率评估
    total_improvement = train_losses[0] - train_losses[-1]
    training_efficiency = total_improvement / len(train_losses)
    
    print(f"   📈 总体改善: {total_improvement:.6f}")
    print(f"   ⚡ 训练效率: {training_efficiency:.2e} 每轮")

if __name__ == "__main__":
    create_final_report()