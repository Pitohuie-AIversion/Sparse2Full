#!/usr/bin/env python3
"""
简单的时序NAR训练结果可视化脚本
使用基础的matplotlib功能生成训练可视化报告
"""

import json
import numpy as np
from pathlib import Path

def create_simple_plot():
    """创建简单的训练可视化"""
    # 设置路径
    base_dir = Path(".")
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
        
        # 尝试导入matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            
            # 设置字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            epochs = range(1, len(train_losses) + 1)
            
            # 1. 创建综合训练曲线图
            print("\n🎨 正在生成综合训练曲线...")
            fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig1.suptitle('Temporal NAR Model Training Visualization Report', fontsize=20, fontweight='bold')
            
            # 训练损失曲线
            ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                val_epochs = range(1, len(val_losses) + 1)
                ax1.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            ax1.set_xlabel('Training Epochs')
            ax1.set_ylabel('Loss Value')
            ax1.set_title('Training Loss Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 对数尺度损失
            ax2.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                ax2.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            ax2.set_xlabel('Training Epochs')
            ax2.set_ylabel('Loss Value (Log Scale)')
            ax2.set_title('Log Scale Loss Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 损失分布直方图
            ax3.hist(train_losses, bins=30, alpha=0.7, color='blue', label='Training Loss Distribution', density=True)
            if val_losses:
                ax3.hist(val_losses, bins=30, alpha=0.7, color='red', label='Validation Loss Distribution', density=True)
            
            ax3.set_xlabel('Loss Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Loss Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 改善率分析
            if len(train_losses) > 1:
                train_improvement = np.diff(train_losses)
                improvement_epochs = range(2, len(train_losses) + 1)
                ax4.plot(improvement_epochs, train_improvement, 'g-', linewidth=2, alpha=0.8, label='Training Loss Improvement')
                
                if val_losses and len(val_losses) > 1:
                    val_improvement = np.diff(val_losses)
                    val_improvement_epochs = range(2, len(val_losses) + 1)
                    ax4.plot(val_improvement_epochs, val_improvement, 'orange', linewidth=2, alpha=0.8, label='Validation Loss Improvement')
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Training Epochs')
            ax4.set_ylabel('Loss Improvement Rate')
            ax4.set_title('Loss Improvement Rate')
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
            fig2.suptitle('Convergence Analysis Report', fontsize=20, fontweight='bold')
            
            # 对数尺度损失趋势
            ax1.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses:
                ax1.semilogy(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            # 添加趋势线
            if len(train_losses) > 10:
                z = np.polyfit(epochs, np.log(train_losses), 1)
                trend_line = np.exp(np.poly1d(z)(epochs))
                ax1.semilogy(epochs, trend_line, 'b--', alpha=0.5, label='Training Trend Line')
            
            ax1.set_xlabel('Training Epochs')
            ax1.set_ylabel('Loss Value (Log Scale)')
            ax1.set_title('Log Scale Loss Trend')
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
                    ax2.set_xlabel('Training Epochs')
                    ax2.set_ylabel('Improvement Rate (%)')
                    ax2.set_title(f'Moving Window Improvement Rate (Window Size: {window_size})')
                    ax2.grid(True, alpha=0.3)
            
            # 稳定性分析
            if len(train_losses) > 20:
                rolling_std = []
                window = 10
                for i in range(window, len(train_losses)):
                    rolling_std.append(np.std(train_losses[i-window:i]))
                
                std_epochs = range(window + 1, len(train_losses) + 1)
                ax3.plot(std_epochs, rolling_std, 'purple', linewidth=2, alpha=0.8)
                ax3.set_xlabel('Training Epochs')
                ax3.set_ylabel('Rolling Standard Deviation')
                ax3.set_title('Training Stability Analysis')
                ax3.grid(True, alpha=0.3)
            
            # 最佳模型标记
            if val_losses:
                best_epoch = np.argmin(val_losses) + 1
                best_loss = min(val_losses)
                
                ax4.plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2, alpha=0.8, label='Validation Loss')
                ax4.scatter([best_epoch], [best_loss], color='gold', s=100, zorder=5, label=f'Best Model (Epoch {best_epoch})')
                ax4.axvline(x=best_epoch, color='gold', linestyle='--', alpha=0.5)
                
                ax4.set_xlabel('Training Epochs')
                ax4.set_ylabel('Validation Loss')
                ax4.set_title('Best Model Identification')
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
            fig3.suptitle('Training Report Summary', fontsize=20, fontweight='bold')
            ax.axis('off')
            
            # 创建报告文本
            val_loss_text = f"{stats['val_loss_reduction']:.2f}%" if stats['val_loss_reduction'] else 'N/A'
            convergence_text = stats['convergence_epoch'] if stats['convergence_epoch'] else 'No clear convergence detected'
            training_status = 'Converged' if stats['convergence_epoch'] else 'May need more training'
            
            overfitting_risk = 'Low' if val_losses and abs(train_losses[-1] - val_losses[-1]) < 0.01 else 'Medium' if val_losses else 'Cannot assess'
            stability = 'Good' if len(train_losses) > 50 and np.std(train_losses[-20:]) < 0.01 else 'Fair'
            recommendation = 'Model training is good, ready to use' if stats['best_val_loss'] and stats['best_val_loss'] < 1.0 else 'Consider more training or hyperparameter tuning'
            
            report_text = f"""Training Configuration and Results Summary
{'='*50}

📊 Training Statistics
• Total Training Epochs: {stats['total_epochs']}
• Final Training Loss: {stats['final_train_loss']:.6f}
• Best Training Loss: {stats['best_train_loss']:.6f}
• Training Loss Reduction: {stats['train_loss_reduction']:.2f}%

📈 Validation Statistics
• Best Validation Loss: {stats['best_val_loss']:.6f if stats['best_val_loss'] else 'N/A'}
• Validation Loss Reduction: {val_loss_text}

🎯 Convergence Analysis
• Convergence Epoch: {convergence_text}
• Training Status: {training_status}

📋 Model Performance Assessment
• Overfitting Risk: {overfitting_risk}
• Training Stability: {stability}
• Recommendation: {recommendation}
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
            
        except ImportError as e:
            print(f"❌ matplotlib导入失败: {e}")
            print("正在生成文本报告...")
            
            # 生成文本报告作为备选方案
            report_path = save_dir / "training_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("时序NAR模型训练报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"总训练轮次: {len(train_losses)}\n")
                f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
                f.write(f"最佳训练损失: {min(train_losses):.6f}\n")
                f.write(f"训练损失降幅: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.2f}%\n")
                
                if val_losses:
                    f.write(f"最佳验证损失: {min(val_losses):.6f}\n")
                    f.write(f"验证损失降幅: {(val_losses[0] - val_losses[-1]) / val_losses[0] * 100:.2f}%\n")
                
                f.write(f"\n训练损失序列:\n{train_losses}\n")
                if val_losses:
                    f.write(f"\n验证损失序列:\n{val_losses}\n")
            
            print(f"✅ 文本报告已保存: {report_path}")
        
    except FileNotFoundError:
        print(f"❌ 错误: 未找到训练数据文件 {data_path}")
    except Exception as e:
        print(f"❌ 生成可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_simple_plot()