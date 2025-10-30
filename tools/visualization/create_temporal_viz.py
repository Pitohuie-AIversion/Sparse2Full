#!/usr/bin/env python3
"""
时序NAR训练结果可视化脚本
生成专业的训练可视化报告，包括训练曲线、收敛分析等
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# 设置matplotlib后端和中文字体
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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
        print("📊 正在加载训练数据...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_losses = data['train_losses']
        val_losses = data.get('val_losses', [])
        best_val_loss = data.get('best_val_loss', min(val_losses) if val_losses else None)
        
        print(f"✅ 成功加载 {len(train_losses)} 轮训练数据")
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 创建综合训练曲线图
        print("\n🎨 正在生成综合训练曲线...")
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('时序NAR模型训练可视化报告', fontsize=20, fontweight='bold')
        
        # 训练损失曲线
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            ax1.plot(val_epochs, val_losses, 'r-', linewidth=2, label='验证损失', alpha=0.8)
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 对数尺度损失
        ax2.semilogy(epochs, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
        if val_losses:
            ax2.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='验证损失', alpha=0.8)
        
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失值 (对数尺度)')
        ax2.set_title('对数尺度损失曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 损失分布直方图
        ax3.hist(train_losses, bins=30, alpha=0.7, color='blue', label='训练损失分布', density=True)
        if val_losses:
            ax3.hist(val_losses, bins=30, alpha=0.7, color='red', label='验证损失分布', density=True)
        
        ax3.set_xlabel('损失值')
        ax3.set_ylabel('密度')
        ax3.set_title('损失分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 改善率分析
        if len(train_losses) > 1:
            train_improvement = np.diff(train_losses)
            improvement_epochs = range(2, len(train_losses) + 1)
            ax4.plot(improvement_epochs, train_improvement, 'g-', linewidth=2, alpha=0.8, label='训练损失改善率')
            
            if val_losses and len(val_losses) > 1:
                val_improvement = np.diff(val_losses)
                val_improvement_epochs = range(2, len(val_losses) + 1)
                ax4.plot(val_improvement_epochs, val_improvement, 'orange', linewidth=2, alpha=0.8, label='验证损失改善率')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('损失改善率')
        ax4.set_title('损失改善率')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path1 = save_dir / "training_curves_comprehensive.png"
        plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        print(f"✅ 综合训练曲线图已保存: {save_path1}")
        
        # 2. 创建收敛分析图
        print("\n📈 正在生成收敛分析...")
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('收敛分析报告', fontsize=20, fontweight='bold')
        
        # 对数尺度损失趋势
        ax1.semilogy(epochs, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
        if val_losses:
            ax1.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='验证损失', alpha=0.8)
        
        # 添加趋势线
        if len(train_losses) > 10:
            z = np.polyfit(epochs, np.log(train_losses), 1)
            trend_line = np.exp(np.poly1d(z)(epochs))
            ax1.semilogy(epochs, trend_line, 'b--', alpha=0.5, label='训练趋势线')
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值 (对数尺度)')
        ax1.set_title('对数尺度损失趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 滑动窗口改善率
        window_size = max(5, len(train_losses) // 20)
        if len(train_losses) > window_size:
            moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            moving_epochs = range(window_size, len(train_losses) + 1)
            if len(moving_avg) > 1:
                improvement_rate = np.diff(moving_avg) / moving_avg[:-1] * 100
                
                ax2.plot(moving_epochs[1:], improvement_rate, 'g-', linewidth=2, alpha=0.8)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_xlabel('训练轮次')
                ax2.set_ylabel('改善率 (%)')
                ax2.set_title(f'滑动窗口改善率 (窗口大小: {window_size})')
                ax2.grid(True, alpha=0.3)
        
        # 稳定性分析
        if len(train_losses) > 20:
            rolling_std = []
            window = 10
            for i in range(window, len(train_losses)):
                rolling_std.append(np.std(train_losses[i-window:i]))
            
            std_epochs = range(window + 1, len(train_losses) + 1)
            ax3.plot(std_epochs, rolling_std, 'purple', linewidth=2, alpha=0.8)
            ax3.set_xlabel('训练轮次')
            ax3.set_ylabel('滑动标准差')
            ax3.set_title('训练稳定性分析')
            ax3.grid(True, alpha=0.3)
        
        # 最佳模型标记
        if val_losses:
            best_epoch = np.argmin(val_losses) + 1
            best_loss = min(val_losses)
            
            ax4.plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2, alpha=0.8, label='验证损失')
            ax4.scatter([best_epoch], [best_loss], color='gold', s=100, zorder=5, label=f'最佳模型 (轮次 {best_epoch})')
            ax4.axvline(x=best_epoch, color='gold', linestyle='--', alpha=0.5)
            
            ax4.set_xlabel('训练轮次')
            ax4.set_ylabel('验证损失')
            ax4.set_title('最佳模型识别')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path2 = save_dir / "convergence_analysis.png"
        plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"✅ 收敛分析图已保存: {save_path2}")
        
        # 3. 生成训练报告摘要
        print("\n📋 正在生成训练报告...")
        
        # 计算统计信息
        stats = {
            'total_epochs': len(train_losses),
            'final_train_loss': train_losses[-1],
            'best_train_loss': min(train_losses),
            'train_loss_reduction': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100,
            'best_val_loss': best_val_loss,
            'val_loss_reduction': (val_losses[0] - val_losses[-1]) / val_losses[0] * 100 if val_losses else None,
            'convergence_epoch': None
        }
        
        # 寻找收敛点
        if len(train_losses) > 10:
            for i in range(10, len(train_losses)):
                recent_losses = train_losses[i-10:i]
                if max(recent_losses) - min(recent_losses) < 0.001:
                    stats['convergence_epoch'] = i
                    break
        
        # 创建报告图
        fig3, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig3.suptitle('训练报告摘要', fontsize=20, fontweight='bold')
        ax.axis('off')
        
        # 创建报告文本
        val_loss_text = f"{stats['val_loss_reduction']:.2f}%" if stats['val_loss_reduction'] else 'N/A'
        convergence_text = stats['convergence_epoch'] if stats['convergence_epoch'] else '未检测到明显收敛点'
        training_status = '已收敛' if stats['convergence_epoch'] else '可能需要更多训练'
        
        overfitting_risk = '低' if val_losses and abs(train_losses[-1] - val_losses[-1]) < 0.01 else '中等' if val_losses else '无法评估'
        stability = '良好' if len(train_losses) > 50 and np.std(train_losses[-20:]) < 0.01 else '一般'
        recommendation = '模型训练良好，可以使用' if stats['best_val_loss'] and stats['best_val_loss'] < 1.0 else '建议继续训练或调整超参数'
        
        report_text = f"""训练配置与结果摘要
{'='*50}

📊 训练统计
• 总训练轮次: {stats['total_epochs']}
• 最终训练损失: {stats['final_train_loss']:.6f}
• 最佳训练损失: {stats['best_train_loss']:.6f}
• 训练损失降幅: {stats['train_loss_reduction']:.2f}%

📈 验证统计
• 最佳验证损失: {stats['best_val_loss']:.6f if stats['best_val_loss'] else 'N/A'}
• 验证损失降幅: {val_loss_text}

🎯 收敛分析
• 收敛轮次: {convergence_text}
• 训练状态: {training_status}

📋 模型性能评估
• 过拟合风险: {overfitting_risk}
• 训练稳定性: {stability}
• 建议: {recommendation}
"""
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        save_path3 = save_dir / "training_report_summary.png"
        plt.savefig(save_path3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        print(f"✅ 训练报告摘要已保存: {save_path3}")
        
        print(f"\n🎉 所有可视化图表已成功生成！")
        print(f"📁 保存位置: {save_dir}")
        print("\n生成的文件:")
        print("  - training_curves_comprehensive.png (综合训练曲线)")
        print("  - convergence_analysis.png (收敛分析)")
        print("  - training_report_summary.png (训练报告摘要)")
        
    except FileNotFoundError:
        print(f"❌ 错误: 未找到训练数据文件 {data_path}")
    except Exception as e:
        print(f"❌ 生成可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()